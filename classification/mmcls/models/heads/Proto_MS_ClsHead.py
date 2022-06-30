import torch
from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead
import torch.nn.functional as F



@HEADS.register_module()
class Proto_MS_ClsHead(BaseHead):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
    """  # noqa: W605

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(Proto_MS_ClsHead, self).__init__()

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        #self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)

    def loss(self, cls_score, protos, gt_label):
        #import pdb;pdb.set_trace()
        #cls_score_A,cls_score_B,cls_score_A_B = cls_score
        cls_score_encoder,cls_score_decoder = cls_score 
        
        num_samples = len(cls_score_encoder)
        losses = dict()
        # compute loss
        #loss_A = self.compute_loss(cls_score_A, gt_label, avg_factor=num_samples)
        #loss_B = self.compute_loss(cls_score_B, gt_label, avg_factor=num_samples)
        #loss_A_B = self.compute_loss(cls_score_A_B, gt_label, avg_factor=num_samples)
        loss_encoder,acc_encoder = self.loss_acc(cls_score_encoder, gt_label, protos)
        loss_decoder,acc_dencoder = self.loss_acc(cls_score_decoder, gt_label, protos)

        # compute accuracy
        #import pdb;pdb.set_trace()
        acc = acc_dencoder
        #assert len(acc) == len(self.topk)
        #losses['loss'] = loss_A + loss_B + loss_A_B
        losses['loss'] = loss_encoder + loss_decoder
        losses['accuracy'] = {'top-1': acc}
        return losses


    def loss_acc(self, cls_score, gt_label, protos):
        #import pdb;pdb.set_trace()

        dists = self.euclidean_dist(cls_score, protos)

        log_p_y = F.log_softmax(-dists, dim=1)

        loss = -log_p_y.gather(1, gt_label.view(-1,1)).mean()

        _, y_hat = log_p_y.max(1)
        acc = torch.eq(y_hat, gt_label).float().mean()

        return loss, acc
        
    def forward_train(self, cls_score, protos, gt_label):
        losses = self.loss(cls_score, protos, gt_label)
        return losses
    def simple_test(self, cls_score, protos):
        """Test without augmentation."""
        #import pdb;pdb.set_trace()
        #cls_score_A,cls_score_B,cls_score_A_B = cls_score
        #cls_score = cls_score_A
        cls_score_encoder,cls_score_decoder = cls_score
        cls_score = cls_score_decoder


        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        dists = self.euclidean_dist(cls_score, protos)
        pred = F.softmax(-dists, dim=1)
        #pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
    def euclidean_dist(self,x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)