# [BiFormer: Vision Transformer with Bi-Level Routing Attention](https://arxiv.org/abs/2303.08810)

Official PyTorch implementation of **BiFormer**, from the following paper:

[BiFormer: Vision Transformer with Bi-Level Routing Attention](https://arxiv.org/abs/2303.08810). CVPR 2023.\
[Lei Zhu](https://github.com/rayleizhu), [Xinjiang Wang](https://www.linkedin.com/in/wang-xinjiang-784a3462), [Zhanghan Ke](https://zhke.io/), [Wayne Zhang](http://www.statfe.com/), and [Rynson Lau](https://www.cs.cityu.edu.hk/~rynson/)

--- 
<p align="left">
<img src="assets/teaser.png" width=60% height=60% 
class="center">
</p>

<!-- ✅ ⬜️  -->

## News

* 2023-03-18: We are improving the readability and efficiency of [BRA](ops/torch/bra.py), please stay tuned.
  * We treaure reproducibility, hence keep the implementation we used during exploration stage. It is a little bit messy, as many components/arugments are not used, which may distract you.
  * To make it more readable and friendly for optimization with CUDA, we are refactoring it. It is expected to be done in two weeks.
  * After refactoring, we will start optimization with CUDA to make BRA more memory and computationally efficient. 
  * Collaborations and contributions are welcome, especially if you are an expert in CUDA/[cutlass](https://github.com/NVIDIA/cutlass). There is a chance to co-author a paper.



## Results and Pre-trained Models

### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | model | log | tensorboard log<sup>*</sup> |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:| :---:| 
| BiFormer-T | 224x224 | 81.4 | 13.1 M | 2.2 G | [model](https://matix.li/e36fe9fb086c) | [log](https://matix.li/7b7ca227852d) | - |
| BiFormer-S | 224x224 | 83.8 | 25.5 M | 4.5 G | [model](https://matix.li/5bb436318902) | [log](https://matix.li/173324785feb) |[tensorboard.dev](https://tensorboard.dev/experiment/VQAZonmIRjasGaVDPloM5Q/#scalars) |
| BiFormer-B | 224x224 | 84.3 | 56.8 M | 9.8 G | [model](https://matix.li/995db75f585d) | [log](https://matix.li/da2bff937647) | - |
| BiFormer-STL | 224x224 | 82.7 | 28.4 M | 4.6 G | [model](https://matix.li/4e9034a91a23) | [log](https://matix.li/96e971cfb3d5) | - |

<font size=1>* : reproduced after the acceptance of our paper.</font>

Here the `BiFormer-STL`(**S**win-**T**iny-**L**ayout) model is used in our ablation study. We hope it provides a good start proint for developing your own awsome attention mechanisms.

All files can be accessed from [onedrive](https://1drv.ms/u/s!AkBbczdRlZvChGsXFqAA-PVnA-R8?e=IPlOCG).

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
- [x] Add camera-ready paper link
- [x] IN1k standard training code, log, and pretrained checkpoints
- [ ] IN1k token-labeling code
- [x] Semantic segmentation code
- [ ] Object detection code
- [x] Swin-Tiny-Layout (STL) models
- [ ] Refactor BRA and BiFormer code
- [ ] Visualization demo 
- [x] ~~More efficient implementation with triton~~. See [triton issue #1279](https://github.com/openai/triton/issues/1279)
- [ ] More efficient implementation (fusing gather and attention) with CUDA
