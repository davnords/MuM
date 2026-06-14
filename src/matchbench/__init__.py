from .benchmark import benchmark
from .models import dinov2, crocov2, dinov3, ijepa, masked_autoencoder, vjepa2, dinov1, capi
import tyro
from typing import Literal, Optional, Dict, Any
import json

# python run.py --model dinov2 --dataset scannet --data_path /mimer/NOBACKUP/groups/snic2022-6-266/data/scannet_test_1500 --matcher knn

ModelType = Literal["dinov2", "dinov3", "crocov2", "ijepa", "mae", "capi", "dinov1", "vjepa2"]

_to_model = {
    "dinov2": dinov2,
    "dinov3": dinov3,
    "crocov2": crocov2,
    "ijepa": ijepa,
    "mae": masked_autoencoder,
    "capi": capi,
    "dinov1": dinov1,
    "vjepa2": vjepa2,
}

GLOBAL_STEP = 0
STEP_SIZE = 1


def main(
    model: ModelType,
    dataset: Literal["megadepth", "scannet"] = "megadepth",
    matcher: Literal["linear", "knn"] = "linear",
    dataset_path: str = "data/megadepth",
    batch_size: int = 8,
    resolution: Literal["low", "medium", "high"] = "low",
    log_wandb: bool = False,
    wandb_entity: str = "",
    model_kwargs: str = "{}", # Maybe this is terribly ugly... Perhaps @parskatt can come up with a better way?
    # ...forgive me father I have forsaken you... typing is not my strong suit :D 
):
    model_inst = _to_model[model](**json.loads(model_kwargs))
    model_inst.eval()
    benchmark(
        model_inst,
        dataset=dataset,
        matcher=matcher,
        dataset_path=dataset_path,
        batch_size=batch_size,
        resolution=resolution,
        log_wandb=log_wandb,
        wandb_entity=wandb_entity,
        output_dir=f"./workspace/{model}"
    )


def cli():
    tyro.cli(main)


if __name__ == "__main__":
    tyro.cli(main)
