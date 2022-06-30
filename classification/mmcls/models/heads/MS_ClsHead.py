import torch
from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead
import torch.nn.functional as F


@HEADS.register_module()
class MS_ClsHead(BaseHead):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
    """  # noqa: W605

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 loss_out = 2):
        super(MS_ClsHead, self).__init__()

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.loss_out = loss_out

    def loss(self, cls_score, gt_label):
        #import pdb;pdb.set_trace()
        #cls_score_A,cls_score_B,cls_score_A_B = cls_score
        if self.loss_out == 2:
            cls_score_encoder,cls_score_decoder = cls_score 
            num_samples = len(cls_score_decoder)
        elif self.loss_out ==3:
            cls_A,cls_B,cls_score_encoder,cls_score_decoder = cls_score 
            num_samples = len(cls_score_decoder)
            #import pdb;pdb.set_trace()

            loss_A = self.compute_loss(cls_A, gt_label, avg_factor=num_samples)
            loss_B = self.compute_loss(cls_B, gt_label, avg_factor=num_samples)

        
        losses = dict()
        # compute loss
        #loss_A = self.compute_loss(cls_score_A, gt_label, avg_factor=num_samples)
        #loss_B = self.compute_loss(cls_score_B, gt_label, avg_factor=num_samples)
        #loss_A_B = self.compute_loss(cls_score_A_B, gt_label, avg_factor=num_samples)
        if cls_score_encoder is not None:
            loss_encoder = self.compute_loss(cls_score_encoder, gt_label, avg_factor=num_samples)
        else:
            loss_encoder = 0
        loss_decoder = self.compute_loss(cls_score_decoder, gt_label, avg_factor=num_samples)

        # compute accuracy
        acc = self.compute_accuracy(cls_score_decoder, gt_label)
        assert len(acc) == len(self.topk)
        #losses['loss'] = loss_A + loss_B + loss_A_B
        if self.loss_out ==2:
            losses['loss'] = loss_encoder + loss_decoder
        elif self.loss_out ==3:
            losses['loss'] = loss_encoder + loss_decoder + loss_A + loss_B
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        return losses

    def forward_train(self, cls_score, gt_label):
        losses = self.loss(cls_score, gt_label)
        return losses
    def simple_test(self, cls_score):
        """Test without augmentation."""
        #import pdb;pdb.set_trace()
        #cls_score_A,cls_score_B,cls_score_A_B = cls_score
        #cls_score = cls_score_A

        if self.loss_out == 2:
            cls_score_encoder,cls_score_decoder = cls_score 
        elif self.loss_out ==3:
            cls_A,cls_B,cls_score_encoder,cls_score_decoder = cls_score 

        #cls_score_encoder,cls_score_decoder = cls_score
        cls_score = cls_score_decoder
        #cls_score = cls_score_encoder


        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred