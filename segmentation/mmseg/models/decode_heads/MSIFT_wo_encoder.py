import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


from mmcv.cnn import ConvModule, xavier_init

from ..builder import HEADS
import cv2
from .decode_head import BaseDecodeHead
MIN_NUM_PATCHES = 16




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


class FeatureReorganization(nn.Module):
    """ Patch Embedding to Feature
    input : N num_patch P^2*C
    output: N C H W
    """
    def __init__(self, patch_size=1, in_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = None
        self.dim = self.patch_size ** 2 * in_channels

    def forward(self, x, ori_shape):
        N, num_patches, dim = x.shape
        _, C, H, W = ori_shape
        p = self.patch_size
        out = torch.zeros(ori_shape).to(x.device)
        i, j = 0, 0
        for k in range(num_patches):
            if i + p > W:
                i = 0
                j += p
            out[:, :, i:i+p, j:j+p] = x[:, k, :].reshape(N, C, p, p)
            #out[:, k, :] = x[:, :, i:i+p, j:j+p].flatten(1)
            i += p
        return out


@HEADS.register_module()
class MSFT_WE(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., in_index = 3, input_transform = None):
        super().__init__()
        #share mate
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        #print(num_patches)
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        self.input_transform = input_transform
        #self.in_channels= in_channels
        #self.out_channels= out_channels
        #self.num_outs = num_outs

        #self.FPN_neck_A = FPN(in_channels = in_channels,out_channels= out_channels,num_outs = num_outs)
        # self.FPN_neck_B = FPN(in_channels = in_channels,out_channels= out_channels,num_outs = num_outs)
        #self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.cls_head_A = nn.Sequential(
        #     nn.Linear(channels, dim),
        #     nn.Dropout(dropout)
        # )
        # self.cls_head_B = nn.Sequential(
        #     nn.Linear(channels, dim),
        #     nn.Dropout(dropout)
        # )
        self.in_index = in_index
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)


 
        self.to_latent = nn.Identity()
        # self.combine = combine
        # if self.combine == 'add':
        #     self.mlp_head = nn.Sequential(
        #         nn.LayerNorm(dim),
        #         nn.Linear(dim, num_classes)
        #     )
        # elif self.combine == 'concat':
        #     self.mlp_head = nn.Sequential(
        #         nn.LayerNorm(dim*2),
        #         nn.Linear(dim*2, num_classes)
        #     )            
        #self.dual_transformer = Dual_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        #self.dual_encoder = Dual_encoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.dual_decoder = Dual_decoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        #individule mate

        self.patch_to_embedding_A = nn.Linear(patch_dim, dim)

        #self.cls_token_A = nn.Parameter(torch.randn(1, 1, dim))
        self.patch_to_embedding_B = nn.Linear(patch_dim, dim)
        #self.embedding_to_patch_B = nn.Linear(dim,patch_dim)
        self.embedding_to_patch = nn.Linear(dim, patch_dim)


        #self.Fereorg = FeatureReorganization(patch_size,dim)
        #self.cls_token_B = nn.Parameter(torch.randn(1, 1, dim))
        # if self.loss_out ==3:
        #     self.tl_head_A = nn.Linear(dim, num_classes)
        #     self.tl_head_B = nn.Linear(dim, num_classes)
                       

    def init_weights(self):
        pass
    def forward(self, fea_A, fea_B, mask = None,loss_type = None):
        fea_A = self._transform_inputs(fea_A)
        fea_B = self._transform_inputs(fea_B)
        #import pdb;pdb.set_trace()
        #fea_A = self.FPN_neck_A(fea_A)
        #fea_B = self.FPN_neck_B(fea_B)
        p = self.patch_size

        x_A = rearrange(fea_A, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        ori_shape = fea_A.shape
        x_A = self.patch_to_embedding_A(x_A)
        b, n, _ = x_A.shape
        x_B = rearrange(fea_B, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x_B = self.patch_to_embedding_B(x_B)



        #cls_tokens_A = repeat(self.cls_token_A, '() n d -> b n d', b = b)
        #cls_tokens_B = repeat(self.cls_token_B, '() n d -> b n d', b = b)
        #cls_tokens_A = self.cls_head_A(self.gap(fea_A).squeeze(-1).squeeze(-1)).unsqueeze(1)
        #cls_tokens_B = self.cls_head_B(self.gap(fea_B).squeeze(-1).squeeze(-1)).unsqueeze(1)

        #x_A = torch.cat((cls_tokens_A, x_A), dim=1)
        x_A_D = x_A + self.pos_embedding[:, :(n)]
        x_A_D = self.dropout(x_A_D)

        # x_B = torch.cat((cls_tokens_B, x_B), dim=1)
        x_B_D = x_B + self.pos_embedding[:, :(n)]
        x_B_D = self.dropout(x_B_D)




        #x_A_E,x_B_E = self.dual_encoder(x_A_E,x_B_E, mask)
        #cls_tokens_A_encoder = x_A[:,0]
        #cls_tokens_B_encoder = x_B[:,0]


        x_A_D,x_B_D = self.dual_decoder(x_A_D,x_B_D, mask)
        #cls_tokens_A_decoder = x_A_D[:,0]
        #cls_tokens_B_decoder = x_B_D[:,0]

        #x_A = x_A.mean(dim = 1) if self.pool == 'mean' else x_A[:, 0]
        #x_B = x_B.mean(dim = 1) if self.pool == 'mean' else x_B[:, 0]


        #x_A = self.to_latent(x_A)
        #x_B = self.to_latent(x_B)
        #import pdb;pdb.set_trace()
        out_trans_decoder = self.embedding_to_patch(x_A_D + x_B_D)
        h =  int(ori_shape[-2]/p)
        w = int(ori_shape[-1]/p)
        out_trans_decoder = rearrange(out_trans_decoder, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = p, p2 = p, h = h, w = w)

        if loss_type == 'aux':
            out_cnn_encoder = self.embedding_to_patch(x_A + x_B) 
            out_cnn_encoder = rearrange(out_cnn_encoder, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = p, p2 = p, h = h, w = w)

            #out_trans_encoder = self.embedding_to_patch(x_A_E + x_B_E)
            #out_trans_encoder = rearrange(out_trans_encoder, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = p, p2 = p, h = h, w = w)
            return (out_cnn_encoder,out_trans_decoder)


        #out = self.Fereorg(out,ori_shape)


        return out_trans_decoder


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