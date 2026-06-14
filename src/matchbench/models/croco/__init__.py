from .croco import CroCoNet
import torch
from types import MethodType

def custom_fwd(self, x):
    x, _, _ = self._encode_image(x, do_mask=False)
    return x

def crocov2(weight_path: str = "pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth"):
    ckpt = torch.load(weight_path, map_location='cpu')
    model = CroCoNet(**ckpt.get('croco_kwargs', {}))
    model.load_state_dict(ckpt['model'], strict=True)
    model.forward = MethodType(custom_fwd, model)
    model.patch_size = model.patch_embed.patch_size[0]
    return model