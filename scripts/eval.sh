
python hydra_main.py \
    data_path=./data/in1k input_size=224  batch_size=128 dist_eval=true \
    +slurm=${CLUSTER_ID} slurm.nodes=1 slurm.ngpus=8 \
    eval=true load_release=true model='biformer_small'
