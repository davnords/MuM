from typing import Literal
from .models import VisionEncoder
from .matcher import LinearMatcher, KNNMatcher
from .datasets import build_megadepth_train, build_scannet_train
from .loss import RegressionByClassificationLoss
import torch
from .utils import CheckPoint, get_amp_dtype
from .benchmarks import MegadepthDenseBenchmark, ScannetDenseBenchmark
import wandb
import matchbench
from .train import train_k_steps
import json
from typing import Any
from collections import defaultdict
import os

resolutions = {"low":(448, 448), "medium":(14*8*5, 14*8*5), "high":(14*8*6, 14*8*6)}

def benchmark(
        backbone: VisionEncoder,
        output_dir: str = "./workspace",
        batch_size: int = 8,
        resolution: Literal["low", "medium", "high"] = "low",
        dataset: Literal["megadepth", "scannet", "hypersim"] = "megadepth",
        matcher: Literal["linear", "knn"] = "linear",
        dataset_path: str = "data/megadepth",
        log_wandb: bool = False,
        wandb_entity: str = "",
        weight_decays: tuple = (1e-2, 1e-3, 5e-2),
        learning_rates: tuple = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2),
        compile: bool = False,
        steps: int = 2500,
        evals: int = 10,
        experiment_name: str = None,
        amp: bool = True,
        num_workers: int = 8,
    ) -> None:

    assert dataset in ["megadepth", "scannet"], f"Unknown dataset {dataset}. Choose from 'megadepth' or 'scannet'"
    assert matcher in ["linear", "knn"], f"Unknown matcher {matcher}. Choose from 'linear' or 'knn'."
    assert resolution in resolutions, f"Unknown resolution {resolution}. Choose from 'low', 'medium', or 'high'."

    os.makedirs(output_dir, exist_ok=True)

    experiment_name = backbone.__class__.__name__ if experiment_name is None else experiment_name
    amp_dtype = get_amp_dtype(amp)

    h, w = resolutions[resolution]
    assert h % backbone.patch_size == 0 and w % backbone.patch_size == 0, "Resolution must be divisible by patch size"
    
    rank = 0

    # Num steps
    global_step = 0
    batch_size = batch_size

    gpus = 1
    step_size = gpus*batch_size
    matchbench.STEP_SIZE = step_size
    
    N = (32 * steps)  # 250k steps of batch size 32
    # checkpoint every
    k = (N // evals) // matchbench.STEP_SIZE

    if dataset == "megadepth":
        dataset_train, ws = build_megadepth_train(dataset_path, h, w)
        bench = MegadepthDenseBenchmark(dataset_path, num_samples = 1000, h=h,w=w)
    elif dataset == "scannet":
        # /mimer/NOBACKUP/groups/3d-dl/scannet
        dataset_train, ws = build_scannet_train(dataset_path, h, w)
        bench = ScannetDenseBenchmark(dataset_path+"/scannet_test_1500", h=h, w=w)
    else:
        raise ValueError(f"Unknown dataset {dataset}. Choose from 'megadepth' or 'scannet'.")
    
    if matcher == "knn":
        model = KNNMatcher(backbone, amp=amp, amp_dtype=amp_dtype).cuda()
        knn_perf = bench.benchmark(model)
        with open(f"{output_dir}/{dataset}_knn.json", "w") as f:
            json.dump(knn_perf, f, indent=2)
        return

    model = LinearMatcher(backbone, amp=amp, amp_dtype=amp_dtype, learning_rates=[step_size * lr / 8 for lr in learning_rates], weight_decays=weight_decays).cuda()
    loss = RegressionByClassificationLoss()

    # Create optimizer for all the linear probes
    params_groups: dict[Any, dict[str, Any]] = defaultdict(lambda: {"params": []})
    all_params = model.all_params
    group_keys = tuple(set(all_params[0].keys()) - {"params"})
    for d in all_params:
        key = tuple(d[k] for k in group_keys)
        params_groups[key]["params"].append(d["params"])
    all_params_groups = [{"params": group["params"], **dict(zip(group_keys, key, strict=False))} for key, group in params_groups.items()]
    
    optimizer = torch.optim.AdamW(all_params_groups)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(9*N/matchbench.STEP_SIZE)//10])

    wandb_mode = "online" if log_wandb and rank == 0 else "disabled"
    wandb.init(project="matchbench", entity=wandb_entity, name=experiment_name, reinit=False, mode = wandb_mode)
    checkpointer = CheckPoint(output_dir+"/checkpoints/", experiment_name)
    model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler, global_step)

    if compile: model = torch.compile(model)

    matchbench.GLOBAL_STEP = global_step

    grad_scaler = torch.amp.GradScaler(growth_interval=1_000_000)
    grad_clip_norm = 0.01

    best_epe, best_metrics = float("inf"), {}

    for n in range(matchbench.GLOBAL_STEP, N, k * matchbench.STEP_SIZE):
        sampler = torch.utils.data.WeightedRandomSampler(
            ws, num_samples = batch_size * k, replacement=False
        )
        dataloader = iter(
            torch.utils.data.DataLoader(
                dataset_train,
                batch_size = batch_size,
                sampler = sampler,
                num_workers = num_workers,
            )
        )
        train_k_steps(n, k, dataloader, model, loss, optimizer, lr_scheduler, grad_scaler, grad_clip_norm = grad_clip_norm)
        # checkpointer.save(model, optimizer, lr_scheduler, matchbench.GLOBAL_STEP)
        eval_dict = bench.benchmark(model)

        model_name_, metrics_ = min(eval_dict.items(), key=lambda x: x[1]["epe"])
        epe = metrics_["epe"]
        print(f"Step {matchbench.GLOBAL_STEP}: EPE: {epe:.4f} by {model_name_}")
        wandb.log(metrics_, step = matchbench.GLOBAL_STEP)

        if epe < best_epe:
            print(f"New best EPE (<{epe:.4f}>) achieved at step {matchbench.GLOBAL_STEP} by {model_name_}")
            best_epe = epe
            best_metrics = metrics_

    with open(f"{output_dir}/{dataset}_linear.json", "w") as f:
        json.dump({
            "name": experiment_name,
            "dataset": dataset,
            "metrics": best_metrics,
            "resolution": resolution,
            "weight_decays": weight_decays,
            "learning_rates": learning_rates,
        }, f, indent=2)
    return
