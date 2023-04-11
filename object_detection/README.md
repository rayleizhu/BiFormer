# COCO Object detection 

## How to use

The environment for object detetction has been included in [../environment.yaml](../environment.yaml). Typically, You do not need to take care of it if you create
the environment as specified in [../INSTALL.md](../INSTALL.md). In case there are problems with mmcv or mmdetection, you may uninstall the package and then reinstall it mannually, e.g. 

```bash
pip uninstall mmcv
pip install --no-cache-dir mmcv==1.7.0
```

* STEP 0: prepare data 

```bash
$ mkdir data && ln -s /your/path/to/coco data/coco # prepare data
```

* STEP 1: run experiments

```bash
$ vim slurm_train.sh # change config file, slurm partition, etc.
$ bash slurm_train.sh
```

See [`slurm_train.sh`](./slurm_train.sh) for details.


## Results

| name | Pretrained Model | Method | Lr Schd | mAP_box | mAP_mask | log | mAP_box<sup>*</sup> |  mAP_mask<sup>*</sup> |  tensorboard log<sup>*</sup> | config |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BiFormer-S | IN1k | MaskRCNN | 1x | 47.8 | 43.2 | [log](https://1drv.ms/u/s!AkBbczdRlZvChgmyNozEQrrsfOdG?e=aOCj2A) | 48.1 | 43.6 | [tensorboard.dev](https://tensorboard.dev/experiment/EvZZMPPRTA29oL5m5olPNw/#scalars&tagFilter=mAP&_smoothingWeight=0) | [config](./configs/coco/maskrcnn.1x.biformer_small.py) |
| BiFormer-B | IN1k | MaskRCNN | 1x | 48.6 | 43.7 | [log](https://1drv.ms/u/s!AkBbczdRlZvChhF-itieos4fg28D?e=Gor6oV) | - | - | - | [config](./configs/coco/maskrcnn.1x.biformer_base.py) |
| BiFormer-S | IN1k | RetinaNet | 1x | 45.9 | - | [log](https://1drv.ms/u/s!AkBbczdRlZvChhKipB3XMN4_nIvO?e=TYZzFc) | 47.3 | - | [tensorboard.dev](https://tensorboard.dev/experiment/0wwQtBNFRp2VBwQeFpZy0Q/#scalars&tagFilter=mAP&_smoothingWeight=0)  | [config](./configs/coco/retinanet.1x.biformer_small.py) |
| BiFormer-B | IN1k | RetinaNet | 1x | 47.1 | - | [log](https://1drv.ms/u/s!AkBbczdRlZvChg-8GDypSY9leBsm?e=FyJQm1) |- | - | - | [config](./configs/coco/retinanet.1x.biformer_base.py) |

<font size=1>* : reproduced right before code release.</font>

**NOTE**: This repository produces significantly better performance than the paper reports, **possibly** due to

1. We fixed a ["bug"](./models_mm/biformer_mm.py) of extra normalization layers.
2. We used a different version of mmcv and mmdetetcion.
3. We used native AMP provided by torch instead of [Nvidia apex](https://github.com/NVIDIA/apex).

We do not know which factors actually work though.

## Acknowledgment 

This code is built using [mmdetection](https://github.com/open-mmlab/mmdetection), [timm](https://github.com/rwightman/pytorch-image-models) libraries, and [UniFormer](https://github.com/Sense-X/UniFormer) repository.
