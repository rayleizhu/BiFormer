"""
BiFormer-STL (Swin-Tiny-Layout) model we used in ablation study.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, LayerNorm2d, to_2tuple, trunc_normal_

from ops.bra_legacy import BiLevelRoutingAttention

from ._common import Attention, AttentionLePE


class BiFormerBlock(nn.Module):
    """
    Attention + FFN
    """
    def __init__(self, dim, drop_path=0., num_heads=8, n_win=7, 
                 qk_dim=None, qk_scale=None, topk=4, mlp_ratio=4, side_dwconv=5):
        super().__init__()
        qk_dim = qk_dim or dim
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(
                dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                qk_scale=qk_scale, topk=topk, side_dwconv=side_dwconv)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'), # compatiability
                                      nn.Conv2d(dim, dim, 1), # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim), # pseudo attention
                                      nn.Conv2d(dim, dim, 1), # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                     )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim)
                                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            

    def forward(self, x):
        """
        Args:
            x: NHWC tensor
        Return:
            NHWC tensor
        """
        # attention & mlp
        x = x + self.drop_path(self.attn(self.norm1(x))) # (N, H, W, C)
        x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, H, W, C)
        return x

class BasicLayer(nn.Module):
    """
    Stack several BiFormer Blocks
    """
    def __init__(self, dim, depth, num_heads, n_win, topk,
                 mlp_ratio=4., drop_path=0., side_dwconv=5):

        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            BiFormerBlock(
                    dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    num_heads=num_heads,
                    n_win=n_win,
                    topk=topk,
                    mlp_ratio=mlp_ratio,
                    side_dwconv=side_dwconv,
                )
            for i in range(depth)
        ])

    def forward(self, x:torch.Tensor):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # TODO: use fixed window size instead of fixed number of windows
        x = x.permute(0, 2, 3, 1) # NHWC
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 3, 1, 2) # NCHW
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class BiFormerSTL(nn.Module):
    """
    Replace WindowAttn-ShiftWindowAttn in Swin-T model with Bi-Level Routing Attention
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depth=[2, 2, 6, 2],
                 embed_dim=[96, 192, 384, 768],
                 head_dim=32, qk_scale=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 # before_attn_dwconv=3,
                 mlp_ratios=[4, 4, 4, 4],
                 norm_layer=LayerNorm2d,
                 pre_head_norm_layer=None,
                 ######## biformer specific ############
                 n_wins:Union[int, Tuple[int]]=(7, 7, 7, 7),
                 topks:Union[int, Tuple[int]]=(1, 4, 16, -2),
                 side_dwconv:int=5,
                 #######################################
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # patch embedding: conv-norm
        stem = nn.Sequential(nn.Conv2d(in_chans, embed_dim[0], kernel_size=(4, 4), stride=(4, 4)),
                             norm_layer(embed_dim[0])
                            )
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            # patch merging: norm-conv
            downsample_layer = nn.Sequential(
                        norm_layer(embed_dim[i]), 
                        nn.Conv2d(embed_dim[i], embed_dim[i+1], kernel_size=(2, 2), stride=(2, 2)),
                    )
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)

        ##########################################################################
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads= [dim // head_dim for dim in embed_dim]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        for i in range(4):
            stage = BasicLayer(dim=embed_dim[i],
                               depth=depth[i],
                               num_heads=nheads[i], 
                               mlp_ratio=mlp_ratios[i],
                               drop_path=dp_rates[sum(depth[:i]):sum(depth[:i+1])],
                               ####### biformer specific ########
                               n_win=n_wins[i], topk=topks[i], side_dwconv=side_dwconv
                               ##################################
                               )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)

        ##########################################################################
        pre_head_norm = pre_head_norm_layer or norm_layer 
        self.norm = pre_head_norm(embed_dim[-1])
        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x:torch.Tensor):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        return x

    def forward(self, x:torch.Tensor):
        x = self.forward_features(x)
        # x = x.flatten(2).mean(-1)
        x = x.mean([2, 3])
        x = self.head(x)
        return x


model_urls = {
    "biformer_stl_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChSf-m7ujkvx9lIQ1/root/content",
}

@register_model
def biformer_stl(pretrained=False, pretrained_cfg=None,
                 pretrained_cfg_overlay=None, **kwargs):
    model = BiFormerSTL(depth=[2, 2, 6, 2],
                        embed_dim=[96, 192, 384, 768],
                        mlp_ratios=[4, 4, 4, 4],
                        head_dim=32,
                        norm_layer=nn.BatchNorm2d,
                        ######## biformer specific ############
                        n_wins=(7, 7, 7, 7),
                        topks=(1, 4, 16, -2),
                        side_dwconv=5,
                        #######################################
                        **kwargs)
    if pretrained:
        model_key = 'biformer_stl_in1k'
        url = model_urls[model_key]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True, file_name=f"{model_key}.pth")
        model.load_state_dict(checkpoint["model"])

    return model
