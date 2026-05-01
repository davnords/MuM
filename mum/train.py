import time
import random
import math
import sys
import datetime
import os
import logging
import wandb
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import torch
import torch.backends.cudnn as cudnn

from mum.model import build_model, dtype_dict
from mum.data import DataAugmentationMAE, build_dynamic_dataloader, create_imagenet_dataloader
import mum.utils.distributed as dist
from mum.utils.utils import MetricLogger
import mum.utils.lr_sched as lr_sched
import mum.utils.misc as misc
from mum.utils.config import setup, adict_to_dict

os.environ['WANDB_API_KEY'] = "<Fill in your API key here>"

logger = logging.getLogger("mum")

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("MuM training", add_help=add_help)
    parser.add_argument("--base-config", default="ssl_default_config", metavar="FILE", help="path to base config file")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--track-wandb", action="store_true", help="Turn on wandb tracking")

    parser.add_argument(
        "opts",
        help="""Modify config options at the end of the command. For Yacs configs, use space-separated "PATH.KEY VALUE" pairs. For python-based LazyConfig, use "path.key=value".""".strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    parser.add_argument("--name", type=str, required=True, help="Name of the training run. Used for logging and checkpointing.")

    return parser


def launch_evals(
    cfg: DictConfig,
    output_dir: Path,
    iteration: int,
    model_without_ddp: torch.nn.Module,
):
    max_iter = cfg.train.steps
    for eval_name, eval_def in cfg.evals.items():
        if ((iteration + 1) % eval_def.period == 0) or (iteration == max_iter - 1 and eval_def.final):
            print(f"Launching eval: {eval_name}")
            eval_ckpt_dir = output_dir / f"eval/training_{iteration}"
            eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = (eval_ckpt_dir / f"checkpoint-{iteration}.pth").resolve()
            eval_dir = eval_ckpt_dir / eval_name
        
            if dist.is_main_process():
                torch.save({'model': model_without_ddp.state_dict()}, ckpt_path)

                repo_dir = Path(__file__).resolve().parent
                eval_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created dir {eval_dir}")
                # eval cfg
                eval_cfg_path = eval_dir / "eval_cfg.yaml"
                OmegaConf.save(OmegaConf.create(adict_to_dict(eval_def.eval_config)), eval_cfg_path)
                logger.info(f"Saved eval cfg {eval_cfg_path}")
                
                # TODO: 
                # Here we launch some evals during training that are good to track progress...
                # ...implement them yourself here

                logger.info(f"Launched eval {eval_name}")
            torch.cuda.synchronize()

def do_train(cfg, args, model):

    image_aug = DataAugmentationMAE(img_size=cfg.train.img_size)
    batch_size_per_gpu = cfg.data.max_img_per_gpu

    data_loader_train = build_dynamic_dataloader(
        datasets=cfg.data.datasets,
        common_config=cfg.data.common_config,
        image_aug=image_aug,
        num_workers=cfg.data.num_workers,
        shuffle=cfg.data.shuffle,
        pin_memory=cfg.data.pin_memory,
        max_img_per_gpu=batch_size_per_gpu,
    )

    if cfg.data.imagenet.prob > 0.0:
        imagenet_dataloader = create_imagenet_dataloader(cfg, image_aug)
        imagenet_iter = iter(imagenet_dataloader)
        print('Probability of using ImageNet data:', cfg.data.imagenet.prob)

    print("Model = %s" % str(model))

    eff_batch_size = batch_size_per_gpu * dist.get_global_size()
    
    lr = cfg.train.lr
    if lr is None:  # only base_lr is specified
        lr = cfg.train.blr * eff_batch_size / 256

    print("base lr: %.2e" % (lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % lr)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_local_rank()], find_unused_parameters=False)
    model_without_ddp = model.module

    param_groups = misc.get_parameter_groups(model_without_ddp, cfg.train.weight_decay, skip_list=[f"pos_embed", "decoder_pos_embed"] )
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = misc.NativeScalerWithGradNormCount()

    if cfg.checkpoint:
        start_iter = misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    else:
        start_iter = 0

    metrics_file = os.path.join(args.output_dir, "training_metrics.json") if args.output_dir else None
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)

    header = 'Training'
    accum_iter = 1 # No need for this now
    optimizer.zero_grad()
    step = start_iter
    max_steps = cfg.train.steps

    print(f"Start training for {max_steps} steps from {start_iter}")
    start_time = time.time()
    for data in metric_logger.log_every(
        data_loader_train,
        10,
        header,
        max_steps,
        start_iter
    ):
        if cfg.data.imagenet.prob > 0.0 and (random.random() < cfg.data.imagenet.prob):
            data = {"images": next(imagenet_iter)[0]}

        current_batch_size = data['images'].shape[0]

        if step > max_steps:
            return
        
        if step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, lr, step, cfg)

        samples = data['images']
        samples = samples.to("cuda", non_blocking=True)

        with torch.amp.autocast('cuda', dtype=dtype_dict[cfg.dtype]):
            loss, _, _ = model(samples, mask_ratio=cfg.loss.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(step + 1) % accum_iter == 0)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(grad=total_norm)
        metric_logger.update(batch_size=current_batch_size)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        wandb.log({"loss": loss_value, "lr": lr, "grad": total_norm})


        if step % cfg.train.checkpoint_steps == 0 and step != 0:
            misc.save_model(step=step, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, checkpoint_path=args.output_dir + f"/checkpoint-last.pth")

        launch_evals(cfg, 
                     Path(args.output_dir), 
                     step,
                     model_without_ddp)
        step += 1

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def main(args):
    cfg = setup(args)

    wandb_mode = "online" if args.track_wandb and dist.is_main_process() else "disabled"
    wandb.init(name=args.name or cfg.model.name, project="mum", entity="<fill in your wandb entity>", config=cfg, reinit=False, mode = wandb_mode)
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
        
    model = build_model(cfg)

    model.cuda()
    do_train(cfg, args, model)

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)