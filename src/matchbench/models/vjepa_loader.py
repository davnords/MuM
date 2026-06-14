import torch
from typing import Literal
from torch import nn, Tensor
from types import MethodType

def custom_fwd(
        self: nn.Module,
        x1: Tensor, 
        x2: Tensor,
    ) -> Tensor:
        print('X1 shape:', x1.shape)
        return self._old_forward(x1), self._old_forward(x2)


def vjepa2(model_name: Literal["vjepa2_vit_large", "vjepa2_vit_huge", "vjepa2_vit_giant", "vjepa2_vit_giant_384"] 
           = "vjepa2_vit_large") -> nn.Module:
    model, _ = torch.hub.load('facebookresearch/vjepa2', model_name)
    # model._old_forward = model.forward
    # model.forward = MethodType(custom_fwd, model)
    return model


if __name__ == "__main__":
    model = vjepa2().cuda()
    print(model.is_video)
    x = torch.randn(2, 3, 2, 224, 224).cuda()
    with torch.no_grad():
        y = model(x)
        print(y.shape)