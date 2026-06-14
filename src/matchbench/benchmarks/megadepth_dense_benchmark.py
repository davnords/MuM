from romatch.datasets import MegadepthBuilder
from torch.utils.data import ConcatDataset
from ..utils import dense_benchmark


class MegadepthDenseBenchmark:
    def __init__(self, data_root="data/megadepth", h = 384, w = 512, num_samples = 2000) -> None:
        mega = MegadepthBuilder(data_root=data_root)
        self.dataset = ConcatDataset(
            mega.build_scenes(split="test_loftr", ht=h, wt=w)
        )  # fixed resolution of 384,512
        self.num_samples = num_samples

    def benchmark(self, model, batch_size=8):
        return dense_benchmark(model, self.dataset, batch_size=batch_size, num_samples=self.num_samples)