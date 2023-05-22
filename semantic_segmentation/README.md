# ADE20k Semantic segmentation

## How to use

```bash
$ mkdir data && ln -s /your/path/to/ade20k data/ade20k # prepare data
$ vim slurm_train.sh # change config file, slurm partition, etc.
$ bash slurm_train.sh
```

See [`slurm_train.sh`](./slurm_train.sh) for details.

## Results

| name | Pretrained Model | Method | Crop Size | Lr Schd | mIoU | mIoU (ms + flip) | log | tensorboard log<sup>*</sup> | config |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BiFormer-S | IN1k | S-FPN | 512x512 | 80K | 48.9 | - | [log](https://1drv.ms/u/s!AkBbczdRlZvChSjgc05WLlcybVSL?e=CmnxZd) | [tensorboard.dev](https://tensorboard.dev/experiment/PYUWpJDvTnSLOTVFNEPzMw/#scalars&tagFilter=mIoU&_smoothingWeight=0) | [config](./configs/ade20k/sfpn.biformer_small.py) |
| BiFormer-B | IN1k | S-FPN | 512x512 | 80K | 49.9 | - | [log](https://1drv.ms/u/s!AkBbczdRlZvChSmDV-OKBxckin52?e=zw7PaD) | - | [config](./configs/ade20k/sfpn.biformer_base.py)|
| BiFormer-S | IN1k | UPerNet | 512x512 | 160K | 49.8 | 50.8 | [log](https://1drv.ms/u/s!AkBbczdRlZvChSoPnL0axfXOKP9M?e=xk5Ard) | - | [config](./configs/ade20k/upernet.biformer_small.py) |
| BiFormer-B | IN1k | UPerNet | 512x512 | 160K | 51.0 | 51.7 | [log](https://1drv.ms/u/s!AkBbczdRlZvChSscYzL_UEd8spSi?e=DXjhbY) | - | [config](./configs/ade20k/upernet.biformer_base.py) |

<font size=1>* : reproduced after the acceptance of our paper.</font>

## Acknowledgment 

This code is built using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [timm](https://github.com/rwightman/pytorch-image-models) libraries, and [UniFormer](https://github.com/Sense-X/UniFormer) repository.
