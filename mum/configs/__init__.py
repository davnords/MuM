import pathlib
from omegaconf import OmegaConf

def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)

def load_and_merge_config(default_config:str, config_name: str):
    default_config = OmegaConf.create(load_config("ssl_default_config"))
    loaded_config = load_config(config_name)
    return OmegaConf.merge(default_config, loaded_config)
