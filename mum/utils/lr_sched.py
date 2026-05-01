import math

def adjust_learning_rate(optimizer, lr, step, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < cfg.train.warmup_steps:
        lr = lr * step / cfg.train.warmup_steps 
    else:
        lr = cfg.train.min_lr + (lr - cfg.train.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (step - cfg.train.warmup_steps) / (cfg.train.steps - cfg.train.warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
