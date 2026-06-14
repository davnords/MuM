import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams["savefig.bbox"] = 'tight'

def show_grids(grids, titles, vertical=True):
    if vertical: 
        fig, axs = plt.subplots(len(grids), 1, figsize=(40, 10))
    else:
        fig, axs = plt.subplots(1, len(grids), figsize=(5 * len(grids), 5))
        fig.subplots_adjust(wspace=0.05)  # reduce horizontal space between images
    for ax, img, title in zip(axs, grids, titles):
        img = img.detach()
        img = tf.to_pil_image(img)
        ax.imshow(np.asarray(img))
        ax.set_title(title)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def unnormalize(x):
    imagenet_mean_tensor = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1,3,1,1).to(x.device, non_blocking=True)
    imagenet_std_tensor = torch.tensor(IMAGENET_DEFAULT_STD).view(1,3,1,1).to(x.device, non_blocking=True)
    x = torch.clip((x * imagenet_std_tensor + imagenet_mean_tensor) * 255, 0, 255).detach().cpu()/255
    return x 

def patchify(imgs, p):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

    h, w = imgs.shape[2] // p, imgs.shape[3] // p 
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def unpatchify(x, patch_size, H, W, channels=3):
    """
    x: (N, L, patch_size**2 *channels)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = H // p
    w = W // p
    assert h * w == x.shape[1]


    x = x.reshape(shape=(x.shape[0], h, w, p, p, channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], channels, h * p, w * p))
    return imgs


def reconstruct_predicted_image(pred, seq, patchified, patch_size, mask, visible=True):
    B, S, C_in, H, W = seq.shape
    mean, var = patchified.mean(dim=-1, keepdim=True), patchified.var(dim=-1, keepdim=True)
    pred = pred.view(B*S, -1, patch_size**2*3)
    predicted_image = unpatchify(pred * (var + 1.e-6)**0.5 + mean, patch_size, H, W)
    if visible:
        # Replace visible patches with the original image
        image_masks = unpatchify(patchify(torch.ones_like(predicted_image), patch_size) * mask[:, :, None], patch_size, H, W)
        masked_target_image = (1 - image_masks) * seq.view(B*S, C_in, H, W)
        predicted_image = predicted_image * image_masks + masked_target_image
    predicted_image = predicted_image.view(B, S, C_in, H, W)
    return predicted_image

def qualitative_evaluation(model: torch.nn.Module, seq: torch.Tensor, path:str, visible:bool=True):
    patch_size = model.patch_size
    seq = seq[0].unsqueeze(0) # We only visualize the first batch element
    B, S, C_in, H, W = seq.shape
    with torch.inference_mode():
        loss, out, mask = model(seq)
        mask = mask.to(dtype=torch.bool)
        mask = mask.view(B*S, -1)
        patchified = patchify(seq.view(B*S, C_in, H, W), patch_size)        
        predicted_image = reconstruct_predicted_image(out, seq, patchified, patch_size, mask, visible=visible)

        masked_images = patchified.clone()
        masked_images[mask] = masked_images.min()
        masked_images = unpatchify(masked_images, patch_size, H, W)
        show_grids(
            [
                make_grid(unnormalize(seq[0]), normalize=True),
                make_grid(unnormalize(masked_images), normalize=True),
                make_grid(unnormalize(predicted_image[0]), normalize=True),
            ],
            [
                'Original',
                'Masked',
                'Predicted',
            ]
        )
        plt.show()
        plt.savefig(path)
        plt.close()