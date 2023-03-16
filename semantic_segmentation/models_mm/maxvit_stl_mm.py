
import os.path as osp
from logging import Logger
from typing import Optional

import torch
import torch.nn as nn
# from mmcv_custom import load_checkpoint
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from models_cls.maxvit_stl import MaxViTSTL
from timm.models.layers import LayerNorm2d


def load_checkpoint(model:nn.Module, filename:str, 
    map_location='cpu', strict=False, logger:Optional[Logger]=None):

    if filename.startswith('https://'):
        checkpoint = torch.hub.load_state_dict_from_url(url=filename,
            map_location=map_location, check_hash=True)
    else:
        assert osp.isfile(filename)
        checkpoint = torch.load(filename, map_location=map_location)
    msg = model.load_state_dict(checkpoint["model"], strict=strict)
    if logger is not None and len(msg) > 0:
        logger.info(msg)

    return checkpoint


@BACKBONES.register_module()  
class MaxViTSTL_mm(MaxViTSTL):
    def __init__(self, pretrained, **kwargs):
        super().__init__(norm_layer=LayerNorm2d, **kwargs)

        # step 1: remove unused segmentation head & norm
        del self.head # classification head
        del self.norm # head norm

        # step 2: add extra norms for dense tasks
        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
        
        # step 3: initialization & load ckpt
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)

        # step 4: convert sync bn, as the batch size is too small in segmentation
        # TODO: check if this is correct
        nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            print(f'Load pretrained model from {pretrained}')   
    
    def forward_features(self, x: torch.Tensor):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out.append(self.extra_norms[i](x))
        return tuple(out)
    
    def forward(self, x:torch.Tensor):
        return self.forward_features(x)
