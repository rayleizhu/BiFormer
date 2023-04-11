_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

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
        embed_dim=[96, 192, 384, 768],
        mlp_ratios=[3, 3, 3, 3],
        n_win=16,
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
        drop_path_rate=0.3,
        disable_bn_grad=False),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))

###########################################################################################################
# Below we follow WaveViT
# https://github.com/YehLi/ImageNetModel/blob/main/object_detection/configs/wavevit/retinanet_wavevit_s_fpn_1x_coco.py

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

runner = dict(type='EpochBasedRunner', max_epochs=12)
fp16 = dict(loss_scale=512.0)
find_unused_parameters=True
###########################################################################################################

# place holder for new verison mmdet compatiability
resume_from=None

# custom
checkpoint_config = dict(max_keep_ckpts=1)
