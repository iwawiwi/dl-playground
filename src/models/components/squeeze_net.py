import torch
import torch.nn as nn


class FireBlock(nn.Module):
    def __init__(
        self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.bn_squeeze = nn.BatchNorm2d(squeeze_planes)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.bn_expand1x1 = nn.BatchNorm2d(expand1x1_planes)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.bn_expand3x3 = nn.BatchNorm2d(expand3x3_planes)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.bn_squeeze(self.squeeze(x)))
        return torch.cat(
            [
                self.expand1x1_activation(self.bn_expand1x1(self.expand1x1(x))),
                self.expand3x3_activation(self.bn_expand3x3(self.expand3x3(x))),
            ],
            1,
        )


class ModifiedSqueezenet(nn.Module):
    def __init__(self, num_classes, dropout=0.2, full_conv=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Dropout(dropout),
            FireBlock(64, 16, 64, 64),
            FireBlock(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Dropout(dropout),
            FireBlock(128, 32, 128, 128),
            FireBlock(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Dropout(dropout),
        )
        self.full_conv = full_conv

        if full_conv:
            # convolutional classfier from squeezenet
            final_conv = nn.Conv2d(256, num_classes, kernel_size=1)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        else:
            # fully connected layer for classification
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):
        x = self.features(x)  # (1, 256, 1, 1)

        if not self.full_conv:
            x = x.view(x.size(0), -1)
        x = self.classifier(x)  # output size: (batch_size, num_classes, 1, 1)

        if self.full_conv:
            x = torch.flatten(x, 1)  # flatten non-batch dim

        return x
