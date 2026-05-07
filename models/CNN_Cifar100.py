import torch.nn as nn
import torchvision
from models.Model_base import MyModel


class Model(MyModel):

    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        return self.model(x)
