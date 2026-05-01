import argparse
import os
from typing import List, Optional

import submitit

from cluster import (
    get_slurm_executor_parameters,
    get_slurm_partition,
    get_shared_folder,
    get_init_file,
)

def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
) -> argparse.ArgumentParser:
    parents = parents or []
    slurm_partition = get_slurm_partition()
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--ngpus",
        "--gpus",
        "--gpus-per-node",
        default=1,
        type=int,
        help="Number of GPUs to request on each node",
    )
    parser.add_argument(
        "--nodes",
        "--nnodes",
        default=1,
        type=int,
        help="Number of nodes to request",
    )
    parser.add_argument(
        "--timeout",
        default=2800,
        type=int,
        help="Duration of the job",
    )
    parser.add_argument(
        "--partition",
        default=slurm_partition,
        type=str,
        help="Partition where to submit",
    )
    parser.add_argument(
        "--use-volta32",
        action="store_true",
        help="Request V100-32GB GPUs",
    )
    parser.add_argument(
        "--a40",
        action="store_true",
        help="Use A40 GPUs (only works on Alvis where there are A40s)",
    )
    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    parser.add_argument(
        "--exclude",
        default="",
        type=str,
        help="Nodes to exclude",
    )
    parser.add_argument('--email', action='store_true', help='Send an email to the user!')
    return parser

def submit_jobs(task_class, args, name: str):
    if not args.output_dir:
        if hasattr(args, 'model') and args.model:
            args.output_dir = get_shared_folder() / str(args.model) / "%j"
        else:
            args.output_dir = get_shared_folder() / "%j"
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    kwargs = {}
    if args.use_volta32:
        kwargs["slurm_constraint"] = "volta32gb"
    if args.comment:
        kwargs["slurm_comment"] = args.comment
    if args.exclude:
        kwargs["slurm_exclude"] = args.exclude

    executor_params = get_slurm_executor_parameters(
        nodes=args.nodes,
        num_gpus_per_node=args.ngpus,
        tgpu='A40' if args.a40 else 'A100', 
        timeout_min=args.timeout,  # max is 60 * 72
        slurm_signal_delay_s=120,
        slurm_partition=args.partition,
        slurm_mail_user="youremail@domain.com" if args.email else None,
        slurm_mail_type="ALL" if args.email else None,
        **kwargs,
    )
    executor.update_parameters(name=name, **executor_params)

    args.dist_url = get_init_file().as_uri()
    args.job_dir = args.output_dir

    task = task_class(args)
    job = executor.submit(task)

    print(f"Submitted job_id: {job.job_id}")
    str_output_dir = os.path.abspath(args.output_dir).replace("%j", str(job.job_id))
    print(f"Logs and checkpoints will be saved at: {str_output_dir}")
