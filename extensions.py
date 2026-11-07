"""Fixed-input PyTorch ports for the next two pilot classifiers.

DS-CNN-S follows ARM-software/ML-KWS-for-MCU models.py and train_commands.txt.
MatchboxNet follows the encoder and decoder configuration in NVIDIA's NGC
commandrecognition_en_matchboxnet3x2x64_v2_subset_task 1.0.0rc1 artifact.
"""
from torch import nn


def ds_cnn_small():
    layers = [
        nn.ZeroPad2d((1, 1, 4, 5)),
        nn.Conv2d(1, 64, (10, 4), stride=2),
        nn.ReLU(),
        nn.BatchNorm2d(64),
    ]
    for _ in range(4):
        layers += [
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        ]
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 12))


def separable(inputs, outputs, kernel, dilation=1):
    return nn.Sequential(
        nn.Conv1d(inputs, inputs, kernel, padding=dilation * (kernel // 2),
                  dilation=dilation, groups=inputs, bias=False),
        nn.Conv1d(inputs, outputs, 1, bias=False),
        nn.BatchNorm1d(outputs),
    )


class MatchboxBlock(nn.Module):
    def __init__(self, inputs, kernel):
        super().__init__()
        self.convs = nn.Sequential(
            separable(inputs, 64, kernel), nn.ReLU(), separable(64, 64, kernel)
        )
        self.skip = nn.Sequential(nn.Conv1d(inputs, 64, 1, bias=False), nn.BatchNorm1d(64))

    def forward(self, x):
        return (self.convs(x) + self.skip(x)).relu()


def matchboxnet():
    return nn.Sequential(
        separable(64, 128, 11), nn.ReLU(),
        MatchboxBlock(128, 13), MatchboxBlock(64, 15), MatchboxBlock(64, 17),
        separable(64, 128, 29, dilation=2), nn.ReLU(),
        nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 12),
    )
