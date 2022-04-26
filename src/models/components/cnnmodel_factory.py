import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch import Tensor

from abc import abstractmethod

class CnnModelFactory:
    """
    Reference:
    https://github.com/vrvlive/knowlege-distillation/blob/master/cnnmodels.py
    """
    def __init__(self):
        self.model_builders = {
            'resnet18': Resnet18Classifier,
            'vgg16': Vgg16Classifier,
            'densenet121': DenseNet121Classifier,
            'simple': SimpleClassifier,
        }

    def build_model(self, model_name, **kwargs):
        if model_name not in self.model_builders:
            raise ValueError(f"Model {model_name} is not supported")

        return self.model_builders[model_name](**kwargs)


class CnnClassifier(nn.Module):
    def __init__(
        self, 
        image_size, 
        num_classes, 
        pretrained=False, 
        name="student"
    ):
        super().__init__()
        self.state_dict_file = f"{name}_{num_classes}_state_dict.pt"
        self.image_size = image_size
        self.num_classes = num_classes
        self.pretrained = pretrained

    def get_fclayer_list(self, hidden_layers):
        """
        Get the list of fully connected layers.
        """
        input_layers, output_layers = hidden_layers[:-1], hidden_layers[1:]
        layers = []
        for i, (l1, l2) in enumerate(zip(input_layers, output_layers)):
            layers.append((f"fc{i}", nn.Linear(l1, l2)))
            layers.append((f"relu{i}", nn.ReLU()))
        layers.append((f"fc_out", nn.Linear(output_layers[-1], self.num_classes)))
        return layers

    def forward(self, x):
        return self.get_model().forward(x)

    @abstractmethod
    def get_model(self):
        pass

    def save_model(self):
        torch.save(self.get_model().state_dict(), self.state_dict_file)

    def load_model(self):
        state_dict = torch.load(self.state_dict_file)
        self.get_model().load_state_dict(state_dict)

class SimpleClassifier(CnnClassifier):
    """
    Simple CNN classifier model
    """
    def __init__(
        self, 
        image_size, 
        num_classes, 
        pretrained=False, 
    ):
        super().__init__(image_size, num_classes, pretrained, name="simple")

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512) # it is supposed that image_size parameter to calculate input of linear layer
        self.fc2 = nn.Linear(512, self.num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # dropout
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_model(self):
        return self

class Resnet18Classifier(CnnClassifier):
    """
    Resnet18 CNN classifier model
    """
    def __init__(
        self, 
        image_size, 
        num_classes, 
        pretrained=False, 
    ):
        super().__init__(image_size, num_classes, pretrained, name="resnet18")
        self.model = torchvision.models.resnet18(pretrained=self.pretrained)
        self.model.fc = nn.Sequential(OrderedDict(self.get_fclayer_list([512, 100]))) # resnet18 final classification layer are modified

    def get_model(self):
        return self.model

class Vgg16Classifier(CnnClassifier):
    """
    Vgg16 CNN classifier model
    """
    def __init__(
        self, 
        image_size, 
        num_classes, 
        pretrained=False, 
    ):
        super().__init__(image_size, num_classes, pretrained, name="vgg16")
        self.model = torchvision.models.vgg16(pretrained=self.CnnClassifier__pretrained)
        self.model.classifier = nn.Sequential(OrderedDict(self.get_fclayer_list([512*7*7, 4096, 4096, 2048, 512]))) # vgg16 final classification layer are modified

    def get_model(self):
        return self.model

class DenseNet121Classifier(CnnClassifier):
    """
    Densenet121 CNN classifier model
    """
    def __init__(
        self, 
        image_size, 
        num_classes, 
        pretrained=False, 
    ):
        super().__init__(image_size, num_classes, pretrained, name="densenet121")
        self.model = torchvision.models.densenet121(pretrained=self.pretrained)
        self.model.classifier = nn.Sequential(OrderedDict(self.get_fclayer_list([1024, 500]))) # densenet121 final classification layer are modified

    def get_model(self):
        return self.model