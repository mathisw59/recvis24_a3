from dataclasses import dataclass
from pathlib import Path, WindowsPath
from typing import Optional, List, Union, Dict, Any
import yaml
from pydantic import BaseModel, Field, field_validator
import torch

# Register a representer for Path objects
yaml.add_representer(Path, lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', str(data)))
yaml.add_representer(WindowsPath, lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', str(data)))

class OptimizerConfig(BaseModel):
    name: str = Field(..., description="Name of the optimizer (adam, sgd, etc)")
    lr: float = Field(..., description="Learning rate")
    weight_decay: float = 0.0
    momentum: Optional[float] = None
    betas: Optional[tuple] = None

    @field_validator('name')
    def validate_optimizer(cls, v):
        allowed = ['adam', 'sgd', 'adamw']
        if v.lower() not in allowed:
            raise ValueError(f'Optimizer must be one of {allowed}')
        return v.lower()
    
class CriterionConfig(BaseModel):
    name: str = Field(..., description="Name of the criterion")
    params: Optional[Dict[str, Any]] = {'temperature': 0.1}

    @field_validator('name')
    def validate_criterion(cls, v):
        allowed = ['crossentropy', 'simclr']
        if v.lower() not in allowed:
            raise ValueError(f'Criterion must be one of {allowed}')
        return v.lower()
                    

class SchedulerConfig(BaseModel):
    name: str = Field(..., description="Name of the scheduler")
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('name')
    def validate_scheduler(cls, v):
        allowed = ['step', 'cosine', 'reduce_on_plateau', 'linear', 'constant']
        if v.lower() not in allowed:
            raise ValueError(f'Scheduler must be one of {allowed}')
        return v.lower()

class DataConfig(BaseModel):
    directory: Path = Field(..., description="Path to training data")
    batch_size: int = Field(..., description="Batch size for training")
    num_workers: int = 8
    shuffle: bool = True
    transform: Optional[str] = None

    @field_validator('batch_size')
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError('Batch size must be positive')
        return v
    
    @field_validator('directory')
    def validate_directory(cls, v):
        if not v.exists():
            raise ValueError(f'Directory {v} does not exist')
        return v
    
    @field_validator('transform')
    def validate_transform(cls, v):
        if v is not None:
            allowed = ['simclr', 'basic']
            if v.lower() not in allowed:
                raise ValueError(f'Scheduler must be one of {allowed}')
        return v
    
class ModelConfig(BaseModel):
    name: str = Field(..., description="Model architecture name")
    pretrained: bool = False
    supervised: bool = True
    weights_path: Optional[Path] = None
    fix_encoder: bool = False
    head: Optional[str] = "linear"

    @field_validator('name')
    def validate_model(cls, v):
        allowed = ['resnet18', 'vit', 'basic_cnn',
                   'mobilenetv3', 'mobilenetv3_small', 'mobilenetv3_large',
                   'squeezenet1_1', 'squeezenet1_0', 'squeezenet1',
                   'efficientnetv2', 'efficientnet']
        if v.lower() not in allowed:
            raise ValueError(f'Model must be one of {allowed}')
        return v.lower()

class EarlyStoppingConfig(BaseModel):
    min_delta: float = 0.0
    patience: int = 10
    mode: str = 'min'


class TrainingConfig(BaseModel):
    # Basic training parameters
    experiment_name: str = Field(..., description="Name of the experiment")
    seed: int = 42
    epochs: int = Field(..., description="Number of training epochs")
    mixed_precision: bool = False
    mixed_precision_level: Optional[str] = 'float32'
    checkpoint_dir: Optional[Path] = Path("checkpoints")
    config_dir: Optional[Path] = Path("configuration_files/automated_saves/")
    log_dir: Optional[Path] = Path("experiments_logs/default/")
    use_cuda: bool = torch.cuda.is_available()
    
    # Components
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    data: DataConfig
    criterion: CriterionConfig
    early_stopping: Optional[EarlyStoppingConfig] = None

    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 1
    
    # Training features
    gradient_clipping: Optional[float] = None
    gradient_checkpointing: bool = False
    
    class Config:
        arbitrary_types_allowed = True

    def save(self, path: Union[str, Path]):
        with open(path, 'w+') as f:
            yaml.dump(self.model_dump(), f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingConfig':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

def get_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Create optimizer based on config."""
    if config.name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas or (0.9, 0.999)
        )
    elif config.name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum or 0.0,
            weight_decay=config.weight_decay
        )
    elif config.name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas or (0.9, 0.999)
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")

def get_scheduler(optimizer: torch.optim.Optimizer, config: SchedulerConfig):
    """Create learning rate scheduler based on config."""
    if config.name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.params.get('step_size', 30),
            gamma=config.params.get('gamma', 0.1)
        )
    elif config.name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.params.get('T_max', 100),
            eta_min=config.params.get('eta_min', 0)
        )
    elif config.name == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.params.get('mode', 'min'),
            factor=config.params.get('factor', 0.1),
            patience=config.params.get('patience', 10)
        )
    raise ValueError(f"Unsupported scheduler: {config.name}")
