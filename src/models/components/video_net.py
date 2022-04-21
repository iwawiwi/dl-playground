import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18


class SimpleResNet3D_18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = nn.Sequential(*list(r3d_18(pretrained=False).children())[:-1])
        self.fc1 = nn.Linear(512, 51)
        self.fc2 = nn.Linear(51, 51)
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        x = self.base_model(x).squeeze(4).squeeze(3).squeeze(2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x
