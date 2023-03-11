"""
This script is modified from https://github.com/facebookresearch/ConvNeXt/blob/main/run_with_submitit.py

Modifications:

* Using hydra to manage arguments and generate experiment dir according to argument overrides
* Surpporting both tcp/file mode of DDP communication. Finding free tcp port automatically.
* Change slurm flags to support SenseTime cluster environment.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import uuid
from pathlib import Path

import hydra
import submitit
from omegaconf import DictConfig, OmegaConf


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def get_shared_folder(share_root) -> Path:
    if Path(share_root).parent.is_dir():
        p = Path(f"{share_root}")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError(f"The parent of share_root ({share_root}) must exist!")

def get_init_file(share_root):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(share_root)), exist_ok=True)
    # FIXME: it seems that, processes on different machines may generate different init file
    # This cuase failure when using multiple machines 
    init_file = get_shared_folder(share_root) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

def get_dist_url(ddp_comm_mode='tcp', share_root=None):
    if ddp_comm_mode == 'file':
        assert share_root is not None
        return get_init_file(share_root).as_uri()
    elif ddp_comm_mode == 'tcp':
        return 'env://' # will be set inside trainer.main() later
    else:
        raise ValueError('Unknown DDP communication mode')


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as trainer
        self._setup_gpu_args()
        trainer.main(self.args)

    def checkpoint(self): # for auto resume when on time-constrained slurm cluster
        import os

        import submitit

        self.args.dist_url = get_dist_url(
            ddp_comm_mode=self.args.slurm.ddp_comm_mode,
            share_root=self.args.slurm.share_root)

        self.args.auto_resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path

        import submitit

        job_env = submitit.JobEnvironment()
        # https://shomy.top/2022/01/05/torch-ddp-intro/
        # self.args.dist_url = f'tcp://{job_env.hostname}:{self.args.slurm.port}'
        self.args.output_dir = Path(self.args.slurm.job_dir)
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks

        self.args.slurm_jobid = job_env.job_id

        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")



@hydra.main(version_base=None, config_path="./configs", config_name="train_args.yaml")
def hydra_app(args:DictConfig):
    # NOTE: enable write to unknow field of cfg
    # hence it behaves like argparse.NameSpace
    # this is required as some args (e.g. args.gpu) is determined at runtime
    # https://stackoverflow.com/a/66296809
    OmegaConf.set_struct(args, False)

    if args.slurm.job_dir is None:
        # by default, set slurm job output dir as the one for ddp communication
        # args.slurm.job_dir = get_shared_folder(args.slurm.share_root) / "%j"
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        args.slurm.job_dir = hydra_cfg['runtime']['output_dir']
    
    if args.slurm.master_port is None: # automatically find free port for ddp communication
            args.slurm.master_port = _find_free_port()
    
    if hasattr(args, 'pavi'):
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        if args.pavi.name is None:
            args.pavi.name = hydra_cfg['job']['override_dirname']
        
        cluster_id = hydra_cfg['runtime']['choices']['slurm']
        job_dir = hydra_cfg['runtime']['output_dir']
        desc = f"{cluster_id}, {job_dir}"
        if args.pavi.description is None: # show cluster, job_dir
            args.pavi.description = desc
        else:
            args.pavi.description = ','.join([desc, args.pavi.description])

    executor = submitit.AutoExecutor(folder=args.slurm.job_dir, slurm_max_num_timeout=30)

    ############## NOTE: this part is highly dependent on slurm version ##############
    num_gpus_per_node = args.slurm.ngpus
    nodes = args.slurm.nodes
    timeout_min = args.slurm.timeout * 60 # in minutes
    cpus_per_task = args.slurm.cpus_per_task

    partition = args.slurm.partition
    kwargs = {}
    # if args.use_volta32:
    #     kwargs['slurm_constraint'] = 'volta32gb'
    if args.slurm.comment:
        kwargs['slurm_comment'] = args.slurm.comment

    # NOTE: slurm with different versions may have different flags
    # slurm_additional_parameters is flexible to cope with this scenario
    slurm_additional_parameters={'ntasks': num_gpus_per_node*nodes, 
                                 'gres': f'gpu:{num_gpus_per_node}',
                                 'ntasks-per-node': num_gpus_per_node} # one task per GPU
    if args.slurm.exclude_node:
        slurm_additional_parameters['exclude'] = args.slurm.exclude_node
    
    if args.slurm.quotatype:
        slurm_additional_parameters['quotatype'] = args.slurm.quotatype

    executor.update_parameters(
        ## original
        # mem_gb=40 * num_gpus_per_node,
        # gpus_per_node=num_gpus_per_node, 
        # tasks_per_node=num_gpus_per_node,  # one task per GPU
        # nodes=nodes,
        # timeout_min=timeout_min,  # max is 60 * 72
        ## https://github.com/facebookincubator/submitit/issues/1639
        # mem_per_cpu=4000,
        # gpus_per_node=num_gpus_per_node,
        # cpus_per_task=4,
        cpus_per_task=cpus_per_task,
        nodes=nodes,
        slurm_additional_parameters=slurm_additional_parameters,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )
    ##################################################################################

    executor.update_parameters(name=args.slurm.job_name)

    args.dist_url = get_dist_url(
        ddp_comm_mode=args.slurm.ddp_comm_mode,
        share_root=args.slurm.share_root)

    args.output_dir = args.slurm.job_dir

    # NOTE: upon submission, submitit will
    # 1. serailize the function object to be executed to .pkl file
    # 2. create a submitit.helpers.DelayedSubmission(fn) and 
    #    submitit.helpers.DelayedSubmission(fn) in the queue of submitit (NOT queue of slurm)
    # 3. if the fist job is successfully finished, the second job will be cancelled;
    #    otherwise, the second job will be executed 
    trainer = Trainer(args)
    job = executor.submit(trainer)
    
    print("Submitted job_id:", job.job_id)

if __name__ == "__main__":
    hydra_app()
