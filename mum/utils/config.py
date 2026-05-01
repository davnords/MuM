import math
import logging
import os

from omegaconf import OmegaConf

from .distributed import enable, get_global_rank
from .logging import setup_logging
from ..configs import load_config
import pathlib
import torch
import numpy as np
import random

logger = logging.getLogger("dinov2")

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"train.output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(load_config(args.base_config))
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def default_setup(args, enable_dist: bool = True):
    if enable_dist:
        enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    fix_random_seeds(seed + rank)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


class adict(dict):
    def __init__(self, iterable=None, **kwargs):#, _allow_non_exist=True
        if iterable is not None:
            for key, value in iterable.items():
                self.__setattr__(key, value)
        if kwargs:
            for key, value in kwargs.items():
                self.__setattr__(key, value)
        #self._allow_non_exist = _allow_non_exist
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            #if self._allow_non_exist:
            return None
            raise self.__attr_error(name)

    def __setattr__(self, name, value):
        if type(value) is dict:
            value = adict(value)
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise self.__attr_error(name)

    def __attr_error(self, name):
        return AttributeError("type object '{subclass_name}' has no attribute '{attr_name}'".format(subclass_name=type(self).__name__, attr_name=name))

    def copy(self):
        return adict(self)
def setup(args, enable_dist=True):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    default_setup(args, enable_dist)
    write_config(cfg, args.output_dir)
    cfg = adict(OmegaConf.to_object(cfg))
    return cfg

def adict_to_dict(obj):
    if isinstance(obj, adict):
        return {k: adict_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [adict_to_dict(v) for v in obj]
    else:
        return obj