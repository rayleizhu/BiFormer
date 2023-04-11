
import torch
import torch.nn as nn
from mmcv_custom import load_checkpoint
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from models_cls.biformer import BiFormer
from timm.models.layers import LayerNorm2d


@BACKBONES.register_module()  
class BiFormer_mm(BiFormer):
    def __init__(self, pretrained=None, norm_eval=True, disable_bn_grad=False, **kwargs):
        super().__init__(**kwargs)
        
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

        # step 4: freeze bn
        self.norm_eval = norm_eval
        if disable_bn_grad:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for param in m.parameters():
                        param.requires_grad = False


    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info(f'Load pretrained model from {pretrained}') 
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
    
              
    def forward_features(self, x: torch.Tensor):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # NOTE: in the version before submission:
            # x = self.extra_norms[i](x)
            # out.append(x)
            out.append(self.extra_norms[i](x).contiguous())
        return tuple(out)
    
    def forward(self, x:torch.Tensor):
        return self.forward_features(x)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
