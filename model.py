import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from collections import OrderedDict

from yaml_parser import ModelConfig, TrainingConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

nclasses = 500

import torch.utils.checkpoint as checkpoint

def checkpoint_hook(module, input, output):
    # Apply checkpointing to the module itself
    return checkpoint.checkpoint(module, *input)

def apply_checkpoint_hooks(model):
    # Iterate over all named modules in the model
    for name, module in model.named_modules():
        # Apply checkpointing to convolutional layers (Conv2d and Conv1d)
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            module.register_forward_hook(checkpoint_hook)
        
        # Apply checkpointing to fully connected layers (Linear)
        elif isinstance(module, nn.Linear):
            module.register_forward_hook(checkpoint_hook)

        elif isinstance(module, nn.MultiheadAttention):
            module.register_forward_hook(checkpoint_hook)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv3 = nn.Conv2d(20, 20, kernel_size=5)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x), 2))
#         x = x.view(-1, 320)
#         return x

class Net(nn.Module):
    def __init__(self, output_dim=512):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc = nn.Linear(32 * 7 * 7, output_dim)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = F.relu(self.conv5(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x        


class SelfSupervisedModel(nn.Module):
    def __init__(self, model):
        super(SelfSupervisedModel, self).__init__()
        self.model = model
    
    def forward(self, x):
        batch_shape = x.shape
        out = x.view(-1, *x.shape[2:])
        out = self.model(out)
        new_batch_shape = batch_shape[:2] + out.shape[1:]
        out = out.view(new_batch_shape)
        out = F.normalize(out, dim=-1)
        return out


def getBackboneModel(model_config: ModelConfig):
    if model_config.name == "basic_cnn":
        in_features = 512
        encoder = Net()
    elif model_config.name == "mobilenetv3" or model_config.name == "mobilenetv3_small":
        if model_config.pretrained:
            encoder = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        else:
            encoder = models.mobilenet_v3_small()

        in_features = encoder.classifier[0].in_features
        encoder.classifier = nn.Identity() # remove the classifier head
    
    elif model_config.name == "mobilenetv3_large":
        if model_config.pretrained:
            encoder = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        else:
            encoder = models.mobilenet_v3_large()

        in_features = encoder.classifier[0].in_features
        encoder.classifier = nn.Identity()

    elif model_config.name in ["resnet50", "resnet18"]:
        if model_config.pretrained:
            encoder = models.resnet18(models.ResNet18_Weights.DEFAULT) if model_config.name == "resnet18" else models.resnet50(models.ResNet50_Weights)
        else:
            encoder = models.resnet18() if model_config.name == "resnet18" else models.resnet50()
        in_features = encoder.fc.in_features
        encoder.fc = nn.Identity()

    elif model_config.name == 'vit':
        if model_config.pretrained:
            encoder = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT, image_size=224)
        else:
            encoder = models.vit_b_16()
        in_features = encoder.heads[0].in_features
        encoder.heads = nn.Identity()

    elif model_config.name == 'squeezenet' or model_config.name == 'squeezenet1_1':
        if model_config.pretrained:
            # i don't find it necessary to remove the classifier head for squeenet, as its just conv + relu + pooling
            # maybe i'm wrong, and it would be wise to look at linear evaluation results keeping the classification head
            encoder = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
        else:
            encoder = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1')
        in_features = 1000

    elif model_config.name == 'squeezenet1_0':
        if model_config.pretrained:
            # i don't find it necessary to remove the classifier head for squeenet, as its just conv + relu + pooling
            # maybe i'm wrong, and it would be wise to look at linear evaluation results keeping the classification head
            encoder = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
        else:
            encoder = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0')
        in_features = 1000

    elif model_config.name == "efficientnet":
        if model_config.pretrained:
            encoder = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
        else:
            encoder = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0')
        in_features = encoder.classifier.in_features
        encoder.classifier = nn.Identity()

    elif model_config.name == "efficientnetv2":
        if model_config.pretrained:
            encoder = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        else:
            encoder = models.efficientnet_v2_s()
        in_features = encoder.classifier[1].in_features
        encoder.classifier = nn.Identity()
    
    return encoder, in_features

def get_CLRHead(in_features, out_features):
    CLR_layers: OrderedDict[str, nn.Module] = OrderedDict()

    CLR_layers["head"] = nn.Linear(in_features, in_features)
    CLR_layers["activation"]  = nn.ReLU()
    CLR_layers["head"] == nn.Linear(in_features, out_features)

    return nn.Sequential(CLR_layers)

def get_CLSHead(in_features, out_features):
    return nn.Linear(in_features, out_features)


def get_complete_model(config: TrainingConfig):
    backbone, in_features = getBackboneModel(config.model)
    if config.model.fix_encoder:
        for param in backbone.parameters():
            param.requires_grad = False
    out_features = nclasses if config.model.supervised else 128
    if config.model.head == "linear":
        head = get_CLSHead(in_features, out_features)
    elif config.model.head == "simclr":
        head = get_CLRHead(in_features, out_features)
    else:
        raise ValueError(f"Invalid head type {config.model.head}")

    model = nn.Sequential(OrderedDict([
        ['backbone', backbone],
        ['head', head]
    ]))

    if config.model.weights_path:
        model[0].load_state_dict(torch.load(config.model.weights_path))

    if not config.model.supervised:
        model = SelfSupervisedModel(model)

    if config.gradient_checkpointing:
        apply_checkpoint_hooks(model)
    
    return model




class SimCLRModel(nn.Module):
    """
    Model base for SimCLR.
    """
    def __init__(self, base_encoder, projection_dim=128, **kwargs):
        super(SimCLRModel, self).__init__()
        self.encoder = base_encoder(**kwargs)
        try:
            in_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        except AttributeError:
            # self.encoder = 
            in_features = self.encoder.classifier[0].in_features

        self.projection_head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, projection_dim)
        )
    
    def forward(self, x):
        # Args:
        #  x, shape: (batch_size, 2, 3, 64, 64)
        x1, x2 = x[:, 0], x[:, 1]
        h1 = self.encoder(x1)
        z1 = self.projection_head(h1)
        z1 = F.normalize(z1, dim=-1)
        h2 = self.encoder(x2)
        z2 = self.projection_head(h2)
        z2 = F.normalize(z2, dim=-1)
        # return shape: (batch_size, 2, projection_dim)
        output = torch.stack([z1, z2], dim=1)
        return output


class SimCLRTrainedModel(nn.Module):
    """
    Model for SimCLR.
    """
    def __init__(self, trained_model):
        super(SimCLRTrainedModel, self).__init__()
        self.encoder = trained_model.encoder
        self.encoder.eval()
        self.projection_head = trained_model.projection_head

        # freeze the encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.fc = nn.Linear(128, nclasses)
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        output = self.fc(z)
        return output