import torch
from torch import nn
from torch.nn import functional as F


class SpatialTransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # using MNIST dataset
        self.conv1 = nn.Conv2d(
            1, 10, kernel_size=5
        )  # conv1: 1x28x28 -> 10x24x24 - pool -> 10x12x12
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # conv2: 10x12x12 -> 20x8x8 - pool -> 20x4x4
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)  # flatten in 20x4x4 -> 320
        self.fc2 = nn.Linear(50, 10)  # 10 classes

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),  # conv1: 1x28x28 -> 8x22x22
            nn.MaxPool2d(2, stride=2),  # pool -> 8x11x11
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),  # conv2: 8x11x11 -> 10x7x7
            nn.MaxPool2d(2, stride=2),  # pool -> 10x3x3
            nn.ReLU(True),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),  # flatten in 10x3x3 -> 90
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),  # 3x2 affine matrix
        )

        # initialize weights and bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )  # identity matrix

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform input
        x = self.stn(x)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
