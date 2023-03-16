_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k_sfpn.py',
    '../_base_/default_runtime.py',
]


# model.pretrained is actually loaded by backbone, see
# https://github.com/open-mmlab/mmsegmentation/blob/186572a3ce64ac9b6b37e66d58c76515000c3280/mmseg/models/segmentors/encoder_decoder.py#L32

model=dict(
    pretrained=None, 
    backbone=dict(
        # use _delete_=True to delete all content in the same section of base config
        # otherwise, configs are merged with the same section of base config
        _delete_=True,
        type='BiFormer_mm',
        #--------------------------
        depth=[4, 4, 18, 4],
        embed_dim=[96, 192, 384, 768],
        mlp_ratios=[3, 3, 3, 3],
        n_win=8, # training resolution is 512
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[96, 192, 384, 768],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        #--------------------------
        # it seems whole_eval takes raw-resolution input
        # use auto_pad to allow any-size input
        auto_pad=True,
        # use grad ckpt to save memory on old gpus
        use_checkpoint_stages=[],
        drop_path_rate=0.3),
    neck=dict(in_channels=[96, 192, 384, 768]),
    decode_head=dict(num_classes=150))

############## below we strictly follow uniformer ####################################
# https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/fpn_seg/exp/fpn_global_small/config.py
#############################################################################
gpu_multiples=2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-8)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples, max_keep_ckpts=1)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU', save_best='mIoU')
#############################################################################

# NOTE: True is conflict with checkpoint 
# https://github.com/allenai/longformer/issues/63#issuecomment-648861503
find_unused_parameters=False

# place holder for new verison mmseg compatiability
resume_from=None
device='cuda'

# fp32 training ->
# optimizer_config = dict()

# AMP ->
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
fp16 = dict()

