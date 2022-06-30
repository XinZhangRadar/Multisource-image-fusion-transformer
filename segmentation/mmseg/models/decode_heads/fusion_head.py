import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


from mmcv.cnn import ConvModule, xavier_init

from ..builder import HEADS
import cv2
from .decode_head import BaseDecodeHead
MIN_NUM_PATCHES = 16

class Fusion(nn.Module):
    def __init__(self,  input_dim ,fusion_style):
        super().__init__()
        self.fusion_style = fusion_style
        self.input_dim = input_dim
        if self.fusion_style == 'concat_CNN':   
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(self.input_dim*2, 256, kernel_size=1, stride=1, padding=0), #0
                nn.BatchNorm2d(256,momentum = 0.9),#1
                nn.ReLU(inplace=True),#2  x1
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),#3  
                nn.BatchNorm2d(256,momentum = 0.9),#4
                nn.ReLU(inplace=True),#6 x2
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256,momentum = 0.9),#8
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256,momentum = 0.9),#8
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256,momentum = 0.9),#8
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256,momentum = 0.9),#8
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256,momentum = 0.9),#8
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 2048, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(2048,momentum = 0.9),#8
                nn.ReLU(inplace=True)
            )
        if self.fusion_style == 'concat':   
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(self.input_dim*2, 2048, kernel_size=1, stride=1, padding=0), #0
                nn.BatchNorm2d(2048,momentum = 0.9),
                nn.ReLU(inplace=True)
            )

    def forward(self, fea_A, fea_B, mask = None):
        #import pdb;pdb.set_trace()
        if self.fusion_style == 'concat_CNN':  
            x = torch.cat((fea_A,fea_B),1)
            x = self.fusion_conv(x)
        if self.fusion_style == 'add':  
            x = fea_A + fea_B
        if self.fusion_style == 'concat':
            x = torch.cat((fea_A,fea_B),1)
            x = self.fusion_conv(x)
        return x

     
               



@HEADS.register_module()
class MSF(nn.Module):
    def __init__(self,  input_dim, fusion_style = 'concat',input_transform = None, in_index= 3):
        super().__init__()

        self.input_transform = input_transform
        self.in_index = in_index
        self.fusion_style = fusion_style
        self.input_dim = input_dim
        self.fusion = Fusion(self.input_dim, self.fusion_style)               

    def init_weights(self):
        pass
    def forward(self, fea_A, fea_B, mask = None):
        #import pdb;pdb.set_trace()
        fea_A = self._transform_inputs(fea_A)
        fea_B = self._transform_inputs(fea_B)


        out = self.fusion(fea_A,fea_B)

        return out


    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs