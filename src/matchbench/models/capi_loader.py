# torch.hub.load('facebookresearch/capi:main', 'capi_vitl14_lvd')

import torch
from typing import Literal
from torch import nn, Tensor
from types import MethodType

def custom_fwd(
        self: nn.Module,
        x1: Tensor, 
        x2: Tensor,
    ) -> Tensor:
        # print('X1 shape:', x1.shape)
        # print('out shape: ', self.forward_features(x1, None, None, enc_layer=self.out_layer, dec_layer=1)[0].shape)
        return self.forward_features(x1, None, None, enc_layer=self.out_layer, dec_layer=1)[0][:, self.num_prefix_tokens :], self.forward_features(x2, None, None, enc_layer=self.out_layer, dec_layer=1)[0][:, self.num_prefix_tokens :]

def capi(model_name: Literal["capi_vitl14_lvd"] = "capi_vitl14_lvd") -> nn.Module:
    model = torch.hub.load('facebookresearch/capi:main', model_name)
    model.forward = MethodType(custom_fwd, model)
    return model