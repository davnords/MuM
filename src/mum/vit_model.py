from functools import partial
from typing import Any, Optional, Dict
import torch
from torch import nn, Tensor
from .layers import Mlp, PatchEmbed, RopePositionEmbedding, SelfAttentionBlock

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

class MuMVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,

        norm_layer: str = "layernormbf16",
        n_storage_tokens: int = 0,
        device: Optional[Any] = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            print(f"WARNING: Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))

        rope_cls = partial(
            RopePositionEmbedding,
            dtype=torch.bfloat16,
            device=device,
        )
        self.rope_embed = rope_cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        block_cls = partial(SelfAttentionBlock, 
                ffn_ratio=4.0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                mask_k_bias=False,
                device=device,
        )
        self.blocks = nn.ModuleList([block_cls(dim=embed_dim, num_heads=num_heads) for i in range(depth)])
        self.norm = norm_layer_cls(embed_dim)

    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.patch_embed(x)
        rope_sincos = self.rope_embed(H=x.shape[1], W=x.shape[2])
        x = x.flatten(1,2)  # [SB, L, C], with L=H*W

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x, rope_sincos)
        x_norm = self.norm(x)
        return {
            "x_norm_cls_token": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1:],
            "x_prenorm": x,
        }

def mum_vitl16(pretrained=True, **kwargs) -> MuMVisionTransformer:
    model = MuMVisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(
            "https://github.com/davnords/mum/releases/download/weights/MuM_ViTLarge.pth"
        )
        msg = model.load_state_dict(weights, strict=True)
    return model