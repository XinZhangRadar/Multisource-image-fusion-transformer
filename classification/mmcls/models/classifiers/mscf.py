import torch.nn as nn

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier
import numpy as np
import cv2

@CLASSIFIERS.register_module()
class MultiSourceFusionClassifier(BaseClassifier):

    def __init__(self, backbone=None, neck=None, head=None, pretrained=None):
        super(MultiSourceFusionClassifier, self).__init__()
        if backbone is not None:  
            self.with_backbone = True 
            #import pdb;pdb.set_trace()
            if isinstance(backbone,list):
                self.backbone_A = build_backbone(backbone[0])
                self.backbone_B = build_backbone(backbone[1])
            else:
                self.backbone_A = build_backbone(backbone)
                self.backbone_B = build_backbone(backbone)               
        else :
            self.with_backbone = False
        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(MultiSourceFusionClassifier, self).init_weights(pretrained)
        if self.with_backbone:
            self.backbone_A.init_weights(pretrained=pretrained)
            self.backbone_B.init_weights(pretrained=pretrained)
        
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img_A,img_B):
        """Directly extract features from the backbone + neck
        """
        #import pdb;pdb.set_trace()
        save_images = True
        if save_images:
            mean_EO=np.array([123.675, 116.28, 103.53]).reshape(1,1,3)
            std_EO=np.array([58.395, 57.12, 57.375]).reshape(1,1,3)
            cv2.imwrite('/home/zhangxin/mmclassification/vis/image_EO.jpg',(img_A.detach().cpu().numpy()[0].transpose(1,2,0)*std_EO + mean_EO))
            cv2.imwrite('/home/zhangxin/mmclassification/vis/image_SAR.jpg',(img_B.detach().cpu().numpy()[0].transpose(1,2,0)*std_EO + mean_EO))

        #print(self.with_neck)
        if self.with_backbone:
            img_A = self.backbone_A(img_A)
            img_B = self.backbone_B(img_B)
        if self.with_neck:
            #print(img_A.shape)
            #print(img_B.shape)
            out = self.neck(img_A,img_B)
        return out

    def forward_train(self, img_A, img_B, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #import pdb;pdb.set_trace()
        out = self.extract_feat(img_A, img_B)
        # import pdb;pdb.set_trace()
        # aa = img[0].permute(1,2,0).detach().cpu().numpy()
        # aa = (aa*np.array([58.395, 57.12, 57.375]))+np.array([123.675, 116.28, 103.53])
        # cv2.imwrite('process_img.jpg',aa)


        losses = dict()
        loss = self.head.forward_train(out, **kwargs)
        losses.update(loss)

        return losses

    def simple_test(self, img_A,img_B):
        """Test without augmentation."""
        #import pdb;pdb.set_trace()
        out = self.extract_feat(img_A,img_B)

        return self.head.simple_test(out)
