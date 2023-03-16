



1. Differences between `ade20k_sfpn.py` and `ade20k_uppernet.py`

We follow Uniformer's experimental settings. It seems that, the major difference is how images are resized. 

```bash

$ diff semantic_segmentation/configs/_base_/datasets/ade20k_sfpn.py semantic_segmentation/configs/_base_/datasets/ade20k_uppernet.py 
1,3c1,2
< # copied from uniformer
< # https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/fpn_seg/configs/_base_/datasets/ade20k.py
< #
---
> # copied from Uniformer
> # https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/configs/_base_/datasets/ade20k.py
31c30
<             dict(type='AlignResize', keep_ratio=True, size_divisor=32),
---
>             dict(type='Resize', keep_ratio=True),
42,49c41,45
<         type='RepeatDataset',
<         times=50,
<         dataset=dict(
<             type=dataset_type,
<             data_root=data_root,
<             img_dir='images/training',
<             ann_dir='annotations/training',
<             pipeline=train_pipeline)),
---
>         type=dataset_type,
>         data_root=data_root,
>         img_dir='images/training',
>         ann_dir='annotations/training',
>         pipeline=train_pipeline),

```
