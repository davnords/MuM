import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import cls_to_flow_refine
from .models import VisionEncoder
import itertools


class LogSoftmaxKernel(nn.Module):
    def __init__(self, T = 1/10):
        super().__init__()
        self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bdn,bdm->bmn", x, y) / self.T
        logP = torch.log_softmax(c, dim = 1)
        return logP


class AllClassifiers(nn.Module):
    def __init__(self, classifiers: dict[tuple[float, float], nn.Module]):
        super().__init__()
        self.idx_to_key = list(classifiers.keys())
        self.classifiers = nn.ModuleList(classifiers.values())

    def forward(self, x: Tensor) -> dict[tuple[float, float], Tensor]:
        return {
            key: self.classifiers[i](x)
            for i, key in enumerate(self.idx_to_key)
        }

class BaseMatcher(nn.Module):
    def __init__(self, backbone: VisionEncoder, amp = True, amp_dtype = torch.float16):
        super().__init__()
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            backbone = backbone.to(self.amp_dtype)
        self.backbone = [backbone] # ugly hack to not show parameters to DDP

        self.embed_dim = backbone.embed_dim
        self.patch_size = backbone.patch_size
        self.kernel = LogSoftmaxKernel()
        self.keys = ["default"]
        
    def process_features(self, features):
        """Override this method to add feature processing layers. Default is identity"""
        return features
        
    def forward(self, batch):
        with torch.autocast("cuda", enabled=self.amp, dtype = self.amp_dtype):
            im_A, im_B = batch["im_A"], batch["im_B"]
            # x = torch.cat((im_A,im_B))
            B2,C,H,W = im_A.shape    
            B = B2//2
            with torch.no_grad():
                if next(self.backbone[0].parameters()).device != im_A.device:
                    self.backbone[0] = self.backbone[0].to(im_A.device).to(self.amp_dtype)
                feats_A, feats_B = self.backbone[0](im_A.to(self.amp_dtype), im_B.to(self.amp_dtype))
                features = torch.cat((feats_A, feats_B))
                features = features.permute(0,2,1).reshape(B2*2,self.embed_dim,H//self.patch_size, W//self.patch_size)
            
            # Apply subclass-specific feature processing
            processed = self.process_features(features)
            
            # if multiple probes → run downstream for each
            if isinstance(processed, dict):
                return {
                    key: self._compute_outputs(feats_A, feats_B)
                    for key, (feats_A, feats_B) in {
                        k: v.chunk(2) for k, v in processed.items()
                    }.items()
                }
            else:
                feats_A, feats_B = processed.chunk(2)
                return self._compute_outputs(feats_A, feats_B)

    def _compute_outputs(self, feats_A, feats_B):
        feats_A = feats_A / feats_A.norm(dim=1, keepdim=True)
        feats_B = feats_B / feats_B.norm(dim=1, keepdim=True)

        H_A, W_A = feats_A.shape[-2:]
        H_B, W_B = feats_B.shape[-2:]

        corr_map = self.kernel(
            feats_A.reshape(-1, feats_A.shape[1], H_A*W_A),
            feats_B.reshape(-1, feats_B.shape[1], H_B*W_B)
        )
        corr_map = corr_map.reshape(-1, H_B*W_B, H_A, W_A)
        flow = cls_to_flow_refine(corr_map)
        return {"cls": corr_map, "flow": flow}

    @torch.inference_mode()
    def match(self, im_A, im_B):
        self.eval()
        im_A, im_B = im_A.cuda(), im_B.cuda()
        hs,ws = im_A.shape[-2:]
        B = im_A.shape[0]
        corresps = self.forward({"im_A":im_A, "im_B":im_B})
        flow = corresps["flow"]
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device="cuda"),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device="cuda"),
            )
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(B, 2, hs, ws).permute(0, 2, 3, 1)
        flow = F.interpolate(flow.permute(0,3,1,2), size = im_A.shape[-2:], mode = "bilinear").permute(0,2,3,1)
        return {"default": (torch.cat((im_A_coords, flow), dim = -1), torch.ones_like(flow[...,0]))}

class LinearMatcher(BaseMatcher):
    def __init__(
                self, 
                backbone: VisionEncoder, 
                amp = True, 
                amp_dtype = torch.float16,
                weight_decays: tuple = (5e-4, 1e-3, 5e-2),
                learning_rates: tuple = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2),
        ):
        super().__init__(backbone, amp, amp_dtype)
        # self.linear_probe = nn.Conv2d(self.embed_dim, self.embed_dim, 1,1,0)

        classifiers = {
            (lr, wd): nn.Conv2d(self.embed_dim, self.embed_dim, 1, 1, 0)
            for lr, wd in itertools.product(learning_rates, weight_decays)
        }

        classifiers = {}

        self.keys = classifiers.keys()

        all_params = []

        for lr, wd in itertools.product(learning_rates, weight_decays):
            classifiers[(lr, wd)] = nn.Conv2d(self.embed_dim, self.embed_dim, 1, 1, 0)
            for name, param in classifiers[(lr, wd)].named_parameters():
                all_params.append(
                    {
                        "lr": lr,
                        "weight_decay": 0.0 if "bias" in name else wd,
                        "params": param,
                    },
                )
        self.all_params = all_params
        self.linear_probes = AllClassifiers(classifiers)
        
    def process_features(self, features):
        return self.linear_probes(features)
    
    @torch.inference_mode()
    def match(self, im_A, im_B):
        self.eval()
        im_A, im_B = im_A.cuda(), im_B.cuda()
        hs,ws = im_A.shape[-2:]
        B = im_A.shape[0]
        corresps = self.forward({"im_A":im_A, "im_B":im_B})

        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device="cuda"),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device="cuda"),
            )
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(B, 2, hs, ws).permute(0, 2, 3, 1)

        result = {}
        for name, out in corresps.items():
            flow = out["flow"]
            flow = F.interpolate(flow.permute(0,3,1,2), size = im_A.shape[-2:], mode = "bilinear").permute(0,2,3,1)
            result[name] = torch.cat((im_A_coords, flow), dim = -1), torch.ones_like(flow[...,0])
        return result


class KNNMatcher(BaseMatcher):
    def __init__(self, backbone: VisionEncoder, amp = True, amp_dtype = torch.float16):
        super().__init__(backbone, amp, amp_dtype)