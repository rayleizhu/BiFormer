# [BiFormer: Vision Transformer with Bi-Level Routing Attention]()

Official PyTorch implementation of **BiFormer**, from the following paper:

[BiFormer: Vision Transformer with Bi-Level Routing Attention](). CVPR 2023.\
[Zhuang Liu](https://liuzhuang13.github.io), [Hanzi Mao](https://hanzimao.me/), [Chao-Yuan Wu](https://chaoyuan.org/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) and [Saining Xie](https://sainingxie.com)\
[[`arXiv`]()][[`video`]()]

--- 

<p align="center">
<img src="" width=100% height=100% 
class="center">
</p>


## Catalog
- [ ] ImageNet-1K Training Code  
- [ ] Downstream Transfer (Detection, Segmentation) Code




<!-- ✅ ⬜️  -->

## Results and Pre-trained Models

### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| BiFormer-T | 224x224 | - | - M | - G | [model]() |
| BiFormer-S | 224x224 | - | - M | - G | [model]() |
| BiFormer-B | 224x224 | - | - M | - G | [model]() |

## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation
We give an example evaluation command for a ImageNet-22K pre-trained, then ImageNet-1K fine-tuned ConvNeXt-B:

Single-GPU
```
python main.py --model convnext_base --eval true \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```
Multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_base --eval true \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```

This should give 
```
* Acc@1 85.820 Acc@5 97.868 loss 0.563
```

- For evaluating other model variants, change `--model`, `--resume`, `--input_size` accordingly. You can get the url to pre-trained models from the tables above. 
- Setting model-specific `--drop_path` is not strictly required in evaluation, as the `DropPath` module in timm behaves the same during evaluation; but it is required in training. See [TRAINING.md](TRAINING.md) or our paper for the values used for different models.

## Training
See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and [ConvNext](https://github.com/facebookresearch/ConvNeXt) repository.

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
