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


# class ModelFactory:
#     def __init__(self, config: TrainingConfig):

#         self.model = self.init_model(config)
#         self.transform = self.init_transform(config)
#         self.loss = self.init_loss(config)

#     def init_model(self):
#         if self.model_name == "basic_cnn":
#             return Net()
#         elif self.model_name == "SimCLR":
#             # I am using a Mobile Net V3 model, as:
#             # * I am training locally with a GTX 1050 Ti, and am limited in RAM
#             # * SSL requires huge batch sizes (~>1024)
#             # * It is optimized for small resolution images
#             # * Faster computation
#             # * Accuracy almost equal to that of ResNet18 after SSL
#             # * The data set is relatively small for nowadays standards (20k images) and Mobile Net V3 might less overfit
#             # * I do not want to waste all my money in electric consumption
#             base_encoder = models.mobilenet_v3_small
#             return SimCLRModel(base_encoder, 128, weights=models.MobileNet_V3_Small_Weights(models.MobileNet_V3_Small_Weights.IMAGENET1K_V1))
#         elif self.model_name == "trained_SimCLR":
#             base_encoder = models.mobilenet_v3_small
#             trained_model = SimCLRModel(base_encoder, 128)
#             if self.model_path is not None:
#                 trained_model.load_state_dict(torch.load(self.model_path))
#             model = SimCLRTrainedModel(trained_model)
#             return model
#         else:
#             raise NotImplementedError("Model not implemented")

#     def init_transform(self):
#         if self.model_name == "basic_cnn":
#             return data_transforms
#         elif self.model_name == "SimCLR":
#             return SimCLRTransform()
#         elif self.model_name == "trained_SimCLR":
#             return SimCLRTransformSingle()
#         else:
#             raise NotImplementedError("Transform not implemented")
        
#     def init_loss(self):
#         if self.model_name == "basic_cnn":
#             return nn.CrossEntropyLoss()
#         elif self.model_name == "SimCLR":
#             return SimCLRLoss(self.temperature)
#         elif self.model_name == "trained_SimCLR":
#             return nn.CrossEntropyLoss()
#         else:
#             raise NotImplementedError("Loss not implemented")


#     def get_model(self):
#         return self.model

#     def get_transform(self):
#         return self.transform
    
#     def get_loss(self):
#         return self.loss

#     def get_all(self):
#         return self.model, self.transform, self.loss


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