_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# model.pretrained is actually loaded by backbone, see
# https://github.com/open-mmlab/mmsegmentation/blob/186572a3ce64ac9b6b37e66d58c76515000c3280/mmseg/models/segmentors/encoder_decoder.py#L32

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained=None, 
    backbone=dict(
        # use _delete_=True to delete all content in the same section of base config
        # otherwise, configs are merged with the same section of base config
        _delete_=True,
        type='BiFormer_mm',
        num_classes=80,
        #--------------------------
        depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512],
        mlp_ratios=[3, 3, 3, 3],
        n_win=16,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
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
        drop_path_rate=0.2),
    neck=dict(in_channels=[64, 128, 256, 512]))

###########################################################################################################
# https://github.com/Sense-X/UniFormer/blob/main/object_detection/exp/mask_rcnn_1x_hybrid_small/config.py
# We follow uniformer's optimizer and lr schedule
# but I do not like apex which requires extra MANUAL installation, hence we use pytorch's native amp instead

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[8, 11])

# runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
# # do not use mmdet version fp16 -> WHY?
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

fp16 = dict()
###########################################################################################################

# place holder for new verison mmdet compatiability
resume_from=None

# custom
checkpoint_config = dict(max_keep_ckpts=1)
