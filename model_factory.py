"""Python file to instantite the model and the transform that goes with it."""

from typing import Optional, List, Union, Dict, Any
from pathlib import Path
import yaml

import model as projectModels
import torch.nn as nn
import torch

from yaml_parser import TrainingConfig, get_optimizer, get_scheduler
from losses import get_criterion
from data import get_dataloaders

import os

class ModelFactory:

    def __init__(self, config) -> None:

        self.config = config
        self.config.use_cuda |= torch.cuda.is_available()
        torch.manual_seed(config.seed)

        self.model = projectModels.get_complete_model(config=config)
        if self.config.use_cuda:
            self.model.cuda()
        
        self.criterion = get_criterion(config=config.criterion)
        self.optimizer = get_optimizer(self.model, config.optimizer)
        self.scheduler = get_scheduler(self.optimizer, config.scheduler)
        self.train_loader, self.val_loader = get_dataloaders(config)

        self.log_dir, save_name = self._get_log_directory()
        self.checkpoint_path = os.path.join(config.checkpoint_dir, save_name)
        self.config_path = os.path.join(config.config_dir, save_name + '.yml')
        self.save_name = save_name

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.isdir(os.path.dirname(self.config_path)):
            os.makedirs(os.path.dirname(self.config_path))

        if not os.path.isdir(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.supervised = self.config.model.supervised # allias
        


        self.config.save(self.config_path)

        if self.config.mixed_precision:
            dtypes = {
                'float32': torch.float32,
                'float16': torch.float16
            }
            self.auto_cast = torch.cuda.amp.autocast(dtype=dtypes[self.config.mixed_precision_level])
            self.scaler = torch.cuda.amp.GradScaler()
        

    def __getattr__(self, name): # Called if the variable is not in the model factory instance
        if hasattr(self.config, name):
            return getattr(self.config, name)
        else:
            raise ValueError(f"ModelFactory has no attribute {name}")

    @classmethod
    def from_config(cls, config: TrainingConfig) -> 'ModelFactory':
        return cls(config=config)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ModelFactory':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_config(TrainingConfig(**config_dict))

    def _get_log_directory(self):
        # Automatic renaming to prevent overwriting
        # exemple: experiments/basic_cnn -> experiments/basic_cnn_1 if folder already exists
        
        log_directory = self.config.log_dir
        log_directory = os.path.normpath(log_directory)
        subdirectory = log_directory.split(os.sep)[-1]
        directory = os.sep.join(log_directory.split(os.sep)[:-1])

        if not os.path.exists(log_directory):
            return log_directory, subdirectory
        
        folders = [os.path.join(directory, f) for f in os.listdir(directory)]
        run_num = 1
        while os.path.join(directory, subdirectory) + f"_{run_num}" in folders:
            run_num += 1
        
        return os.path.join(directory, subdirectory) + f"_{run_num}", subdirectory + f"_{run_num}"