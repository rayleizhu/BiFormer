# [BiFormer: Vision Transformer with Bi-Level Routing Attention](https://arxiv.org/)

Official PyTorch implementation of **BiFormer**, from the following paper:

[BiFormer: Vision Transformer with Bi-Level Routing Attention](https://arxiv.org/). CVPR 2023.\
[Lei Zhu](https://github.com/rayleizhu), [Xinjiang Wang](https://www.linkedin.com/in/wang-xinjiang-784a3462), [Zhanghan Ke](https://zhke.io/), [Wayne Zhang](http://www.statfe.com/), and [Rynson Lau](https://www.cs.cityu.edu.hk/~rynson/)

--- 
<p align="left">
<img src="assets/teaser.png" width=60% height=60% 
class="center">
</p>

<!-- ✅ ⬜️  -->

## Results and Pre-trained Models

### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | model | log | tensorboard log<sup>*</sup> |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:| :---:| 
| BiFormer-T | 224x224 | 81.4 | 13.1 M | 2.2 G | [model](https://swfjkq.bn.files.1drv.com/y4m2NuXZIVOYWymYaQkhlZjv4hvxDlYAa9J_KJMtX4VAP2zH-VfWltt5Gb7EGJ-iPKoACFkpS_yGY8-Fdp7L5ioayU-o3SoBY6QsqBhp46pivkZX60n1TCTcM3T83lUackEsDDA5kz6Lyjh0zVP-ZU3_hbDu6YlMJRIM1P1y0CpHuEE2UKbCWg5MUlTrywHBszp/biformer_tiny_best.pth) | [log](https://public.bn.files.1drv.com/y4mVcUM0bsij64TVBvMNcet2KAfz_nZKJQtPyQCQmTb_3LTELOTrznNq3_uPJDfSO3ATSY7CifzicPtfiNjJv5fQAsQZSsaTZuWCTT07CRE6tZEbvbaXudeOpRoG8k_WxRcCuHmiC-h6LHHixQL-Q2RXf1ZDhGJSjY69QvMqrOoZbryiLajTWDKw6R84A4B4k8pIqLN8M_CsXVqOGNrA2PLhfIV1DLabTvoaWwTIZMCD38?AVOverride=1) | - |
| BiFormer-S | 224x224 | 83.8 | 25.5 M | 4.5 G | [model](https://swguta.bn.files.1drv.com/y4mGZOPzLWa8fPl-9p74HhqkhmuJbSC6_bzH62_XKEOB8bRr5oldpqUbwRpHZ9la5vJo_PU4d4-0RLxjXkpQLHuiiTWlKNSGJ5N3yH7qhvB90GlWCU8cA3lND6Efz4qyx0IEmZ8D5GdIWs746vOlueE-VUVTR4sjOP6FAV4AE67ZQVf4sl5__ixUOQC5WDfRZqP/biformer_small_best.pth) | [log](https://public.bn.files.1drv.com/y4mYemBdj3zMGE2VdJ1Jz0OtDeWh_m_1QxmpJOh-XnUtmzQLbazQDOob1WGIqYIqBeAIFCmzAb0QbG5-w-nWrZ3YemyjEEyqArvcgebjeQ6-xWabtfhNqPef-85ahBzbda5VkuhO2-EHp5XvykZZ0nouDexikB7GImk7_7G6kP5AwJVePA0jyViPNfPVia1JBDHU3tSZzLYMFCV8f3o40DuTJKBy67Q5xtilGHLrX_C8HQ?AVOverride=1) |[tensorboard.dev](https://tensorboard.dev/experiment/VQAZonmIRjasGaVDPloM5Q/#scalars) |
| BiFormer-B | 224x224 | 84.3 | 56.8 M | 9.8 G | [model](https://swf4yw.bn.files.1drv.com/y4mQFcsD6ZnpkWuXPF5Q18mEyVjO9ZclXAcJKp88y2TJ_Yh21SLgajCYL69RXELDdhf2FRz4pwQrkAaPBfkxhD-sC7VJGMJH463cYr7KJ5pOfg8qewBSQB5vFtFCezzEWRWU-1ibV8tHPzmcfrvsqTHnsV3uM4iVStatVgrM1PEHz_Id1Eohjijyt3RxqD6F33Q/biformer_base_best.pth) | [log](https://public.bn.files.1drv.com/y4mOwDRFxYjTPqxXFy0M4fs17M8RA4G5mZiiWm5HUvDeFT8yReGxNtsik7Te868ayKfwzJpnF5Kp7cbDHaDjm6y37mjyLztbk3QOJrSynM7sw_gFIHKlaUFb56fEBWH2V4zUOClTtJyLgnLdtdWGaN1csKZ2WH3ClegR2yDUf1WMjDqjBSuJVqSqNRn6xpQxGygqQ-dYj0H3WCW2gYNcGL3l2VhftDP5fjROP7a834ocgU?AVOverride=1) | - |

<font size=1>* : reproduced after the acceptance of our paper.</font>

## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation

We did evaluation on a slurm cluster environment, using the command below:

```bash
python hydra_main.py \
    data_path=./data/in1k input_size=224  batch_size=128 dist_eval=true \
    +slurm=${CLUSTER_ID} slurm.nodes=1 slurm.ngpus=8 \
    eval=true load_release=true model='biformer_small'
```

To test on a local machine, you may try

```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --data_path ./data/in1k --input_size 224 --batch_size 128 --dist_eval \
  --eval --load_release --model biformer_small
```

This should give 
```
* Acc@1 83.754 Acc@5 96.638 loss 0.869
Accuracy of the network on the 50000 test images: 83.8%
```

**Note**: By setting `load_release=true`, the released checkpoints will be automatically downloaded, so you do not need to download manually in advance.

## Training

To launch training on a slurm cluster, use the command below:

```bash
python hydra_main.py \
    data_path=./data/in1k input_size=224  batch_size=128 dist_eval=true \
    +slurm=${CLUSTER_ID} slurm.nodes=1 slurm.ngpus=8 \
    model='biformer_small'  drop_path=0.15 lr=5e-4
```

**Note**: Our codebase automatically generates output directory for experiment logs and checkpoints, according to the passed arguments. For example, the command above will produce an output directory like

```
$ tree -L 3 outputs/ 
outputs/
└── cls
    └── batch_size.128-drop_path.0.15-input_size.224-lr.5e-4-model.biformer_small-slurm.ngpus.8-slurm.nodes.2
        └── 20230307-21:33:26
```

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, and [ConvNext](https://github.com/facebookresearch/ConvNeXt), [UniFormer](https://github.com/Sense-X/UniFormer) repositories.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@Article{zhu2022biformer,
  author  = {Lei Zhu and Xinjiang Wang and Zhanghan Ke and Wayne Zhang and Rynson Lau},
  title   = {BiFormer: Vision Transformer with Bi-Level Routing Attention},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2023},
}
```

## TODOs
- [ ] Add camera-ready paper link
- [x] IN1k standard training code, log, and pretrained checkpoints
- [ ] IN1k token-labeling code
- [ ] Downstream transfer (detection, segmentation) code
- [x] Swin-Tiny-Layout (STL) models
- [ ] Refactor BRA and BiFormer code
- [ ] Visualization demo 
- [x] ~~More efficient implementation with triton~~. See [triton issue #1279](https://github.com/openai/triton/issues/1279)
- [ ] More efficient implementation with CUDA
