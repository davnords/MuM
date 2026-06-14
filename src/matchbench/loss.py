import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_gt_warp
import wandb
import matchbench
import math

class RegressionByClassificationLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def cls_loss(self, x2, prob, cls):
        with torch.no_grad():
            B, C, H, W = next(iter(cls.values())).shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)], indexing='ij')
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2)
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices

        cls = torch.stack(list(cls.values()), dim=-1)  # each [B, C, H, W]
        K = cls.shape[-1]
        cls_loss = F.cross_entropy(cls, 
                                   GT.unsqueeze(-1).expand(-1, -1, -1, K), 
                                   reduction  = 'none')[prob > 0.99]
            
        losses = {
            "cls_loss": cls_loss.mean(),
        }
        wandb.log(losses, step = matchbench.GLOBAL_STEP)
        return losses

    def forward(self, corresps, batch):
        
        cls = {k: v['cls'] for k, v in corresps.items()}
        # cls, flow = (corresps["cls"], corresps.get("flow"))
        _, _, H, W = next(iter(cls.values())).shape

        # B, C, H, W = cls.shape

        gt_warp, gt_prob = get_gt_warp(                
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            H=H,
            W=W,
        )
        x2 = gt_warp.float()
        prob = gt_prob            
        return self.cls_loss(x2, prob, cls)["cls_loss"]