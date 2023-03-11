"""
MaxViT-STL (Swin-Tiny-Layout) model we used in ablation study.
The block-grid attention is proposed in  "MaxViT: Multi-Axis Vision Transformer (ECCV 2022)"

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import LayerNorm2d, to_2tuple, trunc_normal_
from timm.models.maxxvit import MaxxVitTransformerCfg, PartitionAttention2d


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, grid_window_size=7,
                 mlp_ratio=4., drop_path=0.):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # copied from timm.models.maxxvit._tf_cfg
        transformer_cfg = MaxxVitTransformerCfg(
            norm_eps=1e-5,
            # act_layer='gelu_tanh', # NOTE: default in _tf_cfg, but causes error
            act_layer='gelu',
            head_first=False,
            rel_pos_type='bias_tf',
            window_size=to_2tuple(grid_window_size),
            grid_size=to_2tuple(grid_window_size)
        )

        self.blocks = nn.ModuleList([
            PartitionAttention2d(
                dim=dim,
                partition_type='block' if (i % 2 == 0) else 'grid',
                cfg=transformer_cfg,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class MaxViTSTL(nn.Module):
    """
    Replace WindowAttn-ShiftWindowAttn in Swin-T model with WindowAttn-GridAttn proposed in MaxViT 
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depth=[2, 2, 6, 2],
                 embed_dim=[96, 192, 384, 768],
                 head_dim=32, qk_scale=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 grid_window_size=7,
                 # before_attn_dwconv=3,
                 mlp_ratios=[4, 4, 4, 4],
                 norm_layer=LayerNorm2d,
                 pre_head_norm_layer=None):
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
                               grid_window_size=grid_window_size,
                               mlp_ratio=mlp_ratios[i],
                               drop_path=dp_rates[sum(depth[:i]):sum(depth[:i+1])])
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
        x = x.contiguous()
        return x

    def forward(self, x:torch.Tensor):
        x = self.forward_features(x)
        # x = x.flatten(2).mean(-1)
        x = x.mean([2, 3])
        x = self.head(x)
        return x


@register_model
def maxvit_stl(pretrained=False, pretrained_cfg=None,
               pretrained_cfg_overlay=None, **kwargs):
    model = MaxViTSTL(depth=[2, 2, 6, 2],
                      embed_dim=[96, 192, 384, 768],
                      mlp_ratios=[4, 4, 4, 4],
                      grid_window_size=7,
                      head_dim=32,
                      norm_layer=LayerNorm2d,
                      **kwargs)
    return model
