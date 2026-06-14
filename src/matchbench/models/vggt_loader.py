# Note, this is not using VGGT's built in matching functionality, but rather simply extracting its features...
# ...for linear probing

from types import MethodType
from typing import Literal
from torch import Tensor, nn
from .vggt import VGGT

def custom_fwd(
    self: nn.Module,
    x: Tensor,
) -> Tensor:
    aggregated_tokens_list, patch_start_idx = self.aggregator(x.unsqueeze(dim=1))

    feat = aggregated_tokens_list[-1].squeeze(dim=1)
    cls_tok = feat[:, 0]
    registers = feat[:, 1 : patch_start_idx]
    patches = feat[:, patch_start_idx:]
    return patches


# Note, they have said they will release more VGGT versions, so I am keeping this dynamic
VGGT_MODELS = {
    "VGGT_1B": "facebook/VGGT-1B",
}

def __model_loader__(arch: Literal["VGGT_1B"]="VGGT_1B") -> nn.Module:
    model: nn.Module = VGGT.from_pretrained(VGGT_MODELS[arch])
    model.patch_size = model.aggregator.patch_size
    model.forward = MethodType(custom_fwd, model)
    return model