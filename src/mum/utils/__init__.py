from .layers import (
    cat_keep_shapes,
    count_parameters,
    fix_random_seeds,
    get_conda_env,
    get_sha,
    named_apply,
    named_replace,
    uncat_with_shapes,
)
from .viz import qualitative_evaluation
from torchvision import transforms
from PIL import Image

def transform_image(img_path, size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size), # Choose any resolution evenly divisble by 16, this is our pretraining resolution.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with IN1K mean and std
    ])
    img = transform(Image.open(img_path).convert('RGB'))
    return img
