from abc import ABC, abstractmethod
from torch import Tensor
from .croco import crocov2 as crocov2
from .dino_loader import dinov2, dinov3, dinov1
from .ijepa_loader import ijepa
from .mae_loader import masked_autoencoder
from .vjepa_loader import vjepa2
from .capi_loader import capi

# API that a vision encoder must implement to fit in the benchmark
class VisionEncoder(ABC):
    patch_size: int
    embed_dim: int
    @abstractmethod
    def forward(
        self,
        x: Tensor, # 2*B, C, H, W
    ) -> Tensor: # 2*B, N, C
        raise NotImplementedError

