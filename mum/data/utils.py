from simdinov2.data import SamplerType, make_data_loader, make_dataset

from timm.data import create_dataset
from torch.utils.data import DataLoader
import numpy as np

class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.dataloader)
        
        try:
            return next(self.iterator)
        except StopIteration:
            # Reset iterator when exhausted
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def create_imagenet_dataloader(cfg, transform):
    if cfg.data.imagenet.dataset_path.startswith("ImageNet22k"):
        dataset = create_dataset(
            "wds/imagenet-22k-wds",
            root="/mimer/NOBACKUP/groups/3d-dl/imagenet-22k",
            split="train",
            batch_size=cfg.data.max_img_per_gpu,
            repeats=999999,
        )
        dataset.transform = transform

        data_loader = DataLoader(
            dataset,
            batch_size=cfg.data.max_img_per_gpu,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )

        data_loader = InfiniteDataLoader(data_loader)

    else:
        # setup data loader
        dataset = make_dataset(
            dataset_str=cfg.data.imagenet.dataset_path,
            transform=transform,
            target_transform=lambda _: (),
        )

        # sampler_type = SamplerType.INFINITE
        sampler_type = SamplerType.SHARDED_INFINITE
        data_loader = make_data_loader(
            dataset=dataset,
            batch_size=cfg.data.max_img_per_gpu,
            num_workers=cfg.data.num_workers,
            shuffle=True,
            seed=0, 
            sampler_type=sampler_type,
            sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
            drop_last=True,
        )

    return data_loader


def get_target_shape(aspect_ratio, img_size, patch_size):
    """
    Calculate the target shape based on the given aspect ratio.
    
    Args:
        aspect_ratio: Target aspect ratio
        
    Returns:
        numpy.ndarray: Target image shape [height, width]
    """
    short_size = int(img_size * aspect_ratio)
    small_size = patch_size

    # ensure the input shape is friendly to vision transformer
    if short_size % small_size != 0:
        short_size = (short_size // small_size) * small_size

    image_shape = np.array([short_size, img_size])
    return image_shape