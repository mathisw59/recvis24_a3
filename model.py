import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from collections import OrderedDict

from yaml_parser import ModelConfig, TrainingConfig

nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
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
        in_features = 320
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
    out_features = nclasses if config.model.supervised else 128
    head = get_CLSHead(in_features, out_features) if config.model.supervised else get_CLRHead(in_features, out_features)

    model = nn.Sequential(OrderedDict([
        ['backbone', backbone],
        ['head', head]
    ]))

    if config.model.weights_path:
        model['backbone'].load_state_dict(torch.load(config.model.weights_path))

    if not config.model.supervised:
        model = SelfSupervisedModel(model)
    
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