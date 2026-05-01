from enum import Enum
import os
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

class ClusterType(Enum):
    BERZELIUS = "berzelius"
    ALVIS = "alvis"

def _guess_cluster_type() -> ClusterType:
    uname = os.uname()
    if uname.nodename.startswith('alvis'):
        return ClusterType.ALVIS
    elif uname.nodename.startswith('berzelius'):
        return ClusterType.BERZELIUS
    else:
        raise NotImplementedError

def get_cluster_type(cluster_type: Optional[ClusterType] = None) -> Optional[ClusterType]:
    if cluster_type is None:
        return _guess_cluster_type()
    return cluster_type


def get_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None

    CHECKPOINT_DIRNAMES = {
        ClusterType.ALVIS: "/mimer/NOBACKUP/groups/snic2022-6-266",
        ClusterType.BERZELIUS: "/proj/gdl-vision/users",
    }
    return Path("/") / CHECKPOINT_DIRNAMES[cluster_type]


def get_user_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    checkpoint_path = get_checkpoint_path(cluster_type)
    if checkpoint_path is None:
        return None

    username = os.environ.get("USER")
    assert username is not None
    return checkpoint_path / username


def get_slurm_partition(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None

    SLURM_PARTITIONS = {
        ClusterType.ALVIS: "alvis",
        ClusterType.BERZELIUS: "berzelius",
    }
    return SLURM_PARTITIONS[cluster_type]

def get_slurm_account(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None

    SLURM_ACCOUNTS = {
        ClusterType.ALVIS: "NAISS2025-5-255",
        ClusterType.BERZELIUS: "berzelius-2025-148",
    }
    return SLURM_ACCOUNTS[cluster_type]

def get_slurm_executor_parameters(
    nodes: int, num_gpus_per_node: int, cluster_type: Optional[ClusterType] = None, tgpu:str='A100', **kwargs
) -> Dict[str, Any]:
    # create default parameters
    params = {
        "gpus_per_node": num_gpus_per_node,
        "tasks_per_node": num_gpus_per_node,  # one task per GPU
        # "cpus_per_task": 10,
        "slurm_signal_delay_s": 120,
        "nodes": nodes,
        "slurm_partition": get_slurm_partition(cluster_type),
        "slurm_account": get_slurm_account(cluster_type),
    }
    # apply cluster-specific adjustments
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type == ClusterType.ALVIS:
        del params["gpus_per_node"]
        params["slurm_additional_parameters"] = {
            "gpus_per_node" : tgpu+":"+str(num_gpus_per_node)
        } 
   # set additional parameters / apply overrides
    params.update(kwargs)
    return params

def get_shared_folder() -> Path:
    user_checkpoint_path = get_user_checkpoint_path()
    if user_checkpoint_path is None:
        raise RuntimeError("Path to user checkpoint cannot be determined")
    path = user_checkpoint_path / "experiments" / "mv-dino"
    path.mkdir(exist_ok=True)
    return path

def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file