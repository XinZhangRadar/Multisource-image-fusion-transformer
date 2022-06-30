import torch.nn as nn
import torch
from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier
import numpy as np
import cv2

@CLASSIFIERS.register_module()
class Proto_MultiSourceFusionClassifier(BaseClassifier):

    def __init__(self, backbone, neck=None, head=None, pretrained=None, class_num = 10, protos_dim = 64):
        super(Proto_MultiSourceFusionClassifier, self).__init__()
        self.backbone_A = build_backbone(backbone)
        self.backbone_B = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained)
        protos = torch.rand((class_num,protos_dim), requires_grad=True)
        self.protos = torch.nn.Parameter(protos)
    def init_weights(self, pretrained=None):
        super(Proto_MultiSourceFusionClassifier, self).init_weights(pretrained)
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
        x_A = self.backbone_A(img_A)
        x_B = self.backbone_A(img_B)
        if self.with_neck:
            out = self.neck(x_A,x_B)
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
        loss = self.head.forward_train(out, self.protos, **kwargs)
        losses.update(loss)

        return losses

    def simple_test(self, img_A,img_B):
        """Test without augmentation."""
        #import pdb;pdb.set_trace()
        out = self.extract_feat(img_A,img_B)

        return self.head.simple_test(out,self.protos)
