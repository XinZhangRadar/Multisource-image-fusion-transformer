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

class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs, out_big_map = True):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        #import pdb;pdb.set_trace()

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        if out_big_map:
            return outs[0]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)



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
class FPN_MSFT_WO_encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., in_channels=[256, 512, 1024, 2048],out_channels=256,num_outs=5,):
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

        self.FPN_neck_A = FPN(in_channels = in_channels,out_channels= out_channels,num_outs = num_outs)
        self.FPN_neck_B = FPN(in_channels = in_channels,out_channels= out_channels,num_outs = num_outs)
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
        #self.dual_encoder = Dual_encoder(dim, depth, heads, dim_head, mlp_dim, dropout)
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


        #x_A,x_B = self.dual_encoder(x_A,x_B, mask)
        #cls_tokens_A_encoder = x_A[:,0]
        #cls_tokens_B_encoder = x_B[:,0]


        x_A_D,x_B_D = self.dual_decoder(x_A,x_B, mask)
        #cls_tokens_A_decoder = x_A_D[:,0]
        #cls_tokens_B_decoder = x_B_D[:,0]
        cls_tokens_A_decoder = x_A_D.mean(dim = 1)
        cls_tokens_B_decoder = x_B_D.mean(dim = 1)
        #x_A = x_A.mean(dim = 1) if self.pool == 'mean' else x_A[:, 0]
        #x_B = x_B.mean(dim = 1) if self.pool == 'mean' else x_B[:, 0]


        #x_A = self.to_latent(x_A)
        #x_B = self.to_latent(x_B)

        return None,self.mlp_head(cls_tokens_A_decoder+cls_tokens_B_decoder)
