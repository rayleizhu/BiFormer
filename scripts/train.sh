

# python hydra_main.py \
#     data_path=./data/in1k input_size=224 batch_size=128 dist_eval=true \
#     model='biformer_small'  drop_path=0.15 lr=5e-4 \
#     +slurm=${CLUSTER_ID} slurm.nodes=2 slurm.ngpus=8


############### Swin-Tiny-Layout moodels ################

# python hydra_main.py \
#     data_path=./data/in1k input_size=224 batch_size=128 dist_eval=true \
#     model='biformer_stl'  drop_path=0.1 lr=5e-4 \
#     +slurm=${CLUSTER_ID} slurm.nodes=1 slurm.ngpus=8

# python hydra_main.py \
#     data_path=./data/in1k input_size=224 batch_size=128 dist_eval=true \
#     model='maxvit_stl'  drop_path=0.1 lr=5e-4 \
#     +slurm=${CLUSTER_ID} slurm.nodes=1 slurm.ngpus=8

# the refactored version
python hydra_main.py \
    data_path=./data/in1k input_size=224 batch_size=128 dist_eval=true \
    model='biformer_stl_nchw'  drop_path=0.1 lr=5e-4 \
    +slurm=${CLUSTER_ID} slurm.nodes=1 slurm.ngpus=8

