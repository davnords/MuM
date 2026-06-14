import torch
import torch.nn as nn
from enum import Enum
import torch
from mum.vit_model import mum_vitl16
from types import MethodType

class FwdType(str, Enum):
    SINGLE_VIEW = 'single_view'
    MATCHING = 'matching'

def custom_fwd_single_view(
    self: nn.Module,
    x: torch.Tensor,
):
    bs, _, h, w = x.shape
    x = self.forward_encoder(x, 0, return_all_blocks=True)[-1]
    x = self.norm(x)
    outcome = x[:, 0]
    patch_h, patch_w = self.patch_embed.patch_size
    return outcome, None, x[:, 1:].reshape(bs, h // patch_h, w // patch_w, -1)

def custom_fwd_matching(
    self: nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
):
    feat1 = self.forward_features(x1)['x_norm_patchtokens']
    feat2 = self.forward_features(x2)['x_norm_patchtokens']
    return feat1, feat2
    # feat1 = self.forward_encoder(x1, 0, return_all_blocks=True)[-1]
    # feat2 = self.forward_encoder(x2, 0, return_all_blocks=True)[-1]
    # return feat1[:, 1:], feat2[:, 1:]  # Exclude cls token

def __model_loader__(
    fwd: FwdType = FwdType.MATCHING
) -> nn.Module: 

    model = mum_vitl16(pretrained=True)

    if fwd == FwdType.SINGLE_VIEW:
        model.forward = MethodType(custom_fwd_single_view, model)
    elif fwd == FwdType.MATCHING:
        model.forward = MethodType(custom_fwd_matching, model)
    else:
        raise ValueError(f"Unknown forward type: {fwd}")

    model.cuda()
    model.eval()
    return model
    