import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ..builder import NECKS


from mmcv.cnn import ConvModule, xavier_init

from mmcls.core import auto_fp16
from ..builder import NECKS
import cv2

MIN_NUM_PATCHES = 16

class HRFPN(nn.Module):
    """HRFPN (High Resolution Feature Pyrmamids)

    paper: `High-Resolution Representations for Labeling Pixels and Regions
    <https://arxiv.org/abs/1904.04514>`_.

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        stride (int): stride of 3x3 convolutional layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False,
                 stride=1):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.reduction_conv = ConvModule(
            sum(in_channels),
            out_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            act_cfg=None)

        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    act_cfg=None))

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs,out_big_map = True):
        """Forward function."""
        assert len(inputs) == self.num_ins
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            outs.append(
                F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        if out.requires_grad and self.with_cp:
            out = checkpoint(self.reduction_conv, out)
        else:
            out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        outputs = []

        for i in range(self.num_outs):
            if outs[i].requires_grad and self.with_cp:
                tmp_out = checkpoint(self.fpn_convs[i], outs[i])
            else:
                tmp_out = self.fpn_convs[i](outs[i])
            outputs.append(tmp_out)
        if out_big_map:
            return outputs[0]        
        return tuple(outputs)





class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x_A, x_B, **kwargs):
        #import pdb;pdb.set_trace()
        return self.fn(x_A, x_B, **kwargs) + x_A

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x_A, x_B, **kwargs):
        #import pdb;pdb.set_trace()
        if x_B == None:
            return self.fn(self.norm(x_A), **kwargs)  
        else:    
            return self.fn(self.norm(x_A),self.norm(x_B), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class MS_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_q = nn.Linear(dim, inner_dim , bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_A,x_B, mask = None):
        #import pdb;pdb.set_trace()

        b, n, _, h = *x_A.shape, self.heads
        q =  self.to_q(x_A)
        kv = self.to_kv(x_B).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Dual_encoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers_A = nn.ModuleList([])
        self.layers_B = nn.ModuleList([])
        for _ in range(depth):
            self.layers_A.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
            self.layers_B.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x_A, x_B, mask = None):
        for layer_A,layer_B in zip(self.layers_A,self.layers_B):
            #import pdb;pdb.set_trace()
            x_A = layer_A[0](x_A,None, mask = mask) #self_attention
            x_B = layer_B[0](x_B,None, mask = mask)

            x_A = layer_A[1](x_A,None, mask = mask) #multi_sourse_attention
            x_B = layer_B[1](x_B,None, mask = mask)

            x_A = layer_A[2](x_A,None) #FFN
            x_B = layer_B[2](x_B,None)
        return x_A,x_B

class Dual_decoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers_A = nn.ModuleList([])
        self.layers_B = nn.ModuleList([])
        for _ in range(depth):
            self.layers_A.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, MS_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
            self.layers_B.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, MS_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x_A, x_B, mask = None):
        for layer_A,layer_B in zip(self.layers_A,self.layers_B):
            #import pdb;pdb.set_trace()
            x_A = layer_A[0](x_A,None, mask = mask) #self_attention
            x_B = layer_B[0](x_B,None, mask = mask)

            x__A = layer_A[1](x_A,x_B, mask = mask) #multi_sourse_attention
            x_B = layer_B[1](x_B,x_A, mask = mask)

            x_A = layer_A[2](x__A,None) #FFN
            x_B = layer_B[2](x_B,None)
        return x_A,x_B

@NECKS.register_module()
class HRFPN_MSFT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., in_channels=[40, 80, 160, 320],out_channels=256):
        super().__init__()
        #share mate
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        #print(num_patches)
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        #self.in_channels= in_channels
        #self.out_channels= out_channels
        #self.num_outs = num_outs

        self.FPN_neck_A = HRFPN(in_channels = in_channels,out_channels= out_channels)
        self.FPN_neck_B = HRFPN(in_channels = in_channels,out_channels= out_channels)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head_A = nn.Sequential(
            nn.Linear(channels, dim),
            nn.Dropout(dropout)
        )
        self.cls_head_B = nn.Sequential(
            nn.Linear(channels, dim),
            nn.Dropout(dropout)
        )
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)


        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        #self.dual_transformer = Dual_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.dual_encoder = Dual_encoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.dual_decoder = Dual_decoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        #individule mate

        self.patch_to_embedding_A = nn.Linear(patch_dim, dim)
        self.cls_token_A = nn.Parameter(torch.randn(1, 1, dim))
        self.patch_to_embedding_B = nn.Linear(patch_dim, dim)
        self.cls_token_B = nn.Parameter(torch.randn(1, 1, dim))
    def init_weights(self):
        pass
    def forward(self, fea_A, fea_B, mask = None):

        #import pdb;pdb.set_trace()
        fea_A = self.FPN_neck_A(fea_A)
        fea_B = self.FPN_neck_B(fea_B)
        p = self.patch_size

        x_A = rearrange(fea_A, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x_A = self.patch_to_embedding_A(x_A)
        b, n, _ = x_A.shape
        x_B = rearrange(fea_B, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x_B = self.patch_to_embedding_B(x_B)


        #cls_tokens_A = repeat(self.cls_token_A, '() n d -> b n d', b = b)
        #cls_tokens_B = repeat(self.cls_token_B, '() n d -> b n d', b = b)
        cls_tokens_A = self.cls_head_A(self.gap(fea_A).squeeze(-1).squeeze(-1)).unsqueeze(1)
        cls_tokens_B = self.cls_head_B(self.gap(fea_B).squeeze(-1).squeeze(-1)).unsqueeze(1)

        x_A = torch.cat((cls_tokens_A, x_A), dim=1)
        x_A += self.pos_embedding[:, :(n + 1)]
        x_A = self.dropout(x_A)

        x_B = torch.cat((cls_tokens_B, x_B), dim=1)
        x_B += self.pos_embedding[:, :(n + 1)]
        x_B = self.dropout(x_B)


        x_A,x_B = self.dual_encoder(x_A,x_B, mask)
        cls_tokens_A_encoder = x_A[:,0]
        cls_tokens_B_encoder = x_B[:,0]


        x_A_D,x_B_D = self.dual_decoder(x_A,x_B, mask)
        cls_tokens_A_decoder = x_A_D[:,0]
        cls_tokens_B_decoder = x_B_D[:,0]

        #x_A = x_A.mean(dim = 1) if self.pool == 'mean' else x_A[:, 0]
        #x_B = x_B.mean(dim = 1) if self.pool == 'mean' else x_B[:, 0]


        #x_A = self.to_latent(x_A)
        #x_B = self.to_latent(x_B)

        return self.mlp_head(cls_tokens_A_encoder+cls_tokens_B_encoder),self.mlp_head(cls_tokens_A_decoder+cls_tokens_B_decoder)
