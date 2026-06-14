import datetime
import sys
import time
from pathlib import Path
from typing import Any
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn
from matchbench import benchmark
import random
import numpy as np
import importlib

def set_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def base_setup(output_dir: str | Path = ".", seed: int = 0, *, requeue: bool = True, distributed: bool = True) -> None:
    """Create output dir, set up logging, log git status, setup torch distributed, random seeds and cudnn benchmark."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    set_random_seeds(seed)
    torch._C._set_print_stack_traces_on_fatal_signal(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def import_path(file_path: str | Path):
    file_path = Path(file_path).resolve()
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[file_path.stem] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def main(cfg) -> dict[str, float]:
    start = time.perf_counter()
    output_dir = Path(cfg.output_dir)
    OmegaConf.save(config=cfg, f=output_dir / f"config_{Path(__file__).stem}.yaml")
    print(OmegaConf.to_yaml(cfg))
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)  # pyright: ignore [reportAssignmentType]
    try:
         # load model
        model_loader_kwargs = cfg_dict.pop("model_loader_kwargs", {})
        model: nn.Module = import_path("src/mum/model_loader.py").__model_loader__(**model_loader_kwargs)  # type: ignore
        
        # evaluate
        benchmark(model, **cfg_dict)
    finally:
        torch.distributed.destroy_process_group()
        end = time.perf_counter()
        print(f"Total time: {datetime.timedelta(seconds=round(end - start))}")
    return 


if __name__ == "__main__":
    if Path(sys.argv[1]).is_file():
        cfg = OmegaConf.unsafe_merge(
            OmegaConf.load(sys.argv[1]),
            OmegaConf.from_cli(sys.argv[2:]),
        )
    else:
        cfg = OmegaConf.from_cli(sys.argv[1:])
    base_setup(cfg.output_dir)
    main(cfg)