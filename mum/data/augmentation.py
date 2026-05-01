from torchvision import transforms
import torch

class DataAugmentationMAE(object):
    def __init__(
        self,
        img_size=224,
    ):

        # random resized crop and flip
        self.tranform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )


    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        return torch.stack([self.tranform(image) for image in images])