from functools import partial
import timm.models.vision_transformer
import torch
from torch import Tensor, nn

# Class from https://github.com/facebookresearch/mae/blob/main/models_vit.py
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = self.patch_embed.patch_size[0]

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    # custom fwd
    def forward(  # type: ignore
        self,
        x: Tensor,
    ) -> Tensor:
        bs, _, h, w = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls = x[:, 0]
        patch = x[:, 1:]
        return patch


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        num_classes=0,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        num_classes=0,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        num_classes=0,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


model_dict = {
    "vit_base_patch16": (vit_base_patch16, "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"),
    "vit_large_patch16": (vit_large_patch16, "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth"),
    "vit_huge_patch14": (vit_huge_patch14, "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth"),
}


def masked_autoencoder(model_name: str = "vit_large_patch16") -> nn.Module:
    model_fn, url = model_dict[model_name]
    model = model_fn(dynamic_img_size=True)
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location="cpu")["model"])
    return model