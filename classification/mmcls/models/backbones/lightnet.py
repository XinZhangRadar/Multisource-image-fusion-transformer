import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class LightNet(BaseBackbone):
    """`AlexNet <https://en.wikipedia.org/wiki/AlexNet>`_ backbone.

    The input for AlexNet is a 224x224 RGB image.

    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1,input_dim=3,out_indices = (2,6,13,20)):
        super(LightNet, self).__init__()
        self.num_classes = num_classes
        self.input_dim=input_dim
        self.out_indices = out_indices 
        self.features = nn.Sequential(
            nn.Conv2d(self.input_dim, 16, kernel_size=3, stride=1, padding=1), #0
            nn.BatchNorm2d(16,momentum = 0.9),#1
            nn.ReLU(inplace=True),#2  x1
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=1),#3  
            nn.BatchNorm2d(32,momentum = 0.9),#4
            nn.MaxPool2d(kernel_size=2, stride=2),      #5  
            nn.ReLU(inplace=True),#6 x2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),#7
            nn.BatchNorm2d(64,momentum = 0.9),#8
            nn.ReLU(inplace=True),#9 x2
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=1),#10
            nn.BatchNorm2d(128,momentum = 0.9),#11
            nn.MaxPool2d(kernel_size=2, stride=2),#12
            nn.ReLU(inplace=True),#13 x4
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=1),#14
            nn.BatchNorm2d(128,momentum = 0.9),#15
            nn.ReLU(inplace=True),#16 x4
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=1),#17
            nn.BatchNorm2d(64,momentum = 0.9),#18
            nn.AvgPool2d(kernel_size=2, stride=2),#19
            nn.ReLU(inplace=True),#20 x8
        )


    def forward(self, x):
        outs = []
        #import pdb;pdb.set_trace()
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

