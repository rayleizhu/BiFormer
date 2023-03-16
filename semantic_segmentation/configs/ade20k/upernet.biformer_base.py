_base_ = [
    '../configs/_base_/models/upernet_r50.py', 
    '../configs/_base_/datasets/ade20k_upernet.py',
    '../configs/_base_/default_runtime.py', 
    '../configs/_base_/schedules/schedule_160k.py'
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
        drop_path_rate=0.4),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=150),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=150))


############## below we strictly follow uniformer & cswin ####################################
# https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/exp/upernet_global_small/config.py
# https://github.com/microsoft/CSWin-Transformer/blob/main/segmentation/configs/cswin/upernet_cswin_tiny.py
##############################################################################################
# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data=dict(samples_per_gpu=2)
#############################################################################

checkpoint_config = dict(max_keep_ckpts=1)
evaluation = dict(save_best='mIoU')

# NOTE: True is conflict with checkpoint 
# https://github.com/allenai/longformer/issues/63#issuecomment-648861503
find_unused_parameters=False

# place holder for new verison mmseg compatiability
resume_from=None
device='cuda'

# fp32 training (choose this if nan loss occurs)->
# optimizer_config = dict()

# AMP (faster but may meet nan loss) ->
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
fp16 = dict()
