# Installation

We provide installation instructions for ImageNet classification experiments here.

## Environment Setup

Please make sure you have installed [conda](https://docs.conda.io/projects/conda/en/stable/), then you can create the environment with the command below:

```bash
bash scripts/envtool.sh create
conda activate biformer
```

If you are using slurm clusters, it is recommended to create a slurm config file for each available cluster:

```bash
export CLUSTER_ID=[YOUR_CLUSTER_ALIAS]
cp configs/slurm/sz10.yaml configs/slurm/${CLUSTER_ID}.yaml && vim configs/slurm/${CLUSTER_ID}.yaml
```
hence you can launch experiments in any available cluster consistently with `+slurm=${CLUSTER_ID}`.

## Dataset Preparation

1. Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

2. Create a soft link to the IN1k path
```
mkdir data
ln -s /path/to/imagenet-1k data/in1k
```

