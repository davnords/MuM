import os
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from loguru import logger
import gc

import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import json
import math
import torch.nn.functional as F
import romatch
import numpy as np
from torch.nn.functional import cosine_similarity

def get_rank_if_available():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

class CheckPoint:
    def __init__(self, dir=None, name="tmp"):
        self.name = name
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)

    def save(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        ):
        if get_rank_if_available() == 0:
            assert model is not None
            if isinstance(model, (DataParallel, DistributedDataParallel)):
                model = model.module
            states = {
                "model": model.state_dict(),
                "n": n,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(states, self.dir + self.name + f"_latest.pth")
            logger.info(f"Saved states {list(states.keys())}, at step {n}")
    
    def load(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        ):
        if os.path.exists(self.dir + self.name + f"_latest.pth") and torch.distributed.get_rank() == 0:
            states = torch.load(self.dir + self.name + f"_latest.pth")
            if "model" in states:
                model.load_state_dict(states["model"])
            if "n" in states:
                n = states["n"] if states["n"] else n
            if "optimizer" in states:
                try:
                    optimizer.load_state_dict(states["optimizer"])
                except Exception as e:
                    print(f"Failed to load states for optimizer, with error {e}")
            if "lr_scheduler" in states:
                lr_scheduler.load_state_dict(states["lr_scheduler"])
            print(f"Loaded states {list(states.keys())}, at step {n}")
            del states
            gc.collect()
            torch.cuda.empty_cache()
        return model, optimizer, lr_scheduler, n



class MetricLogger(object):
    def __init__(self, delimiter="\t", output_file=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not dist.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")
        pass

    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                self.dump_in_output_file(iteration=i, iter_time=iter_time.avg, data_time=data_time.avg)
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} Total time: {} ({:.6f} s / it)".format(header, total_time_str, total_time / n_iterations))


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self):
        """
        Distributed synchronization of the metric
        Warning: does not synchronize the deque!
        """
        if not dist.is_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


@torch.no_grad()
def cls_to_flow_refine(cls):
    B,C,H,W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(
        *[torch.linspace(-1+1/res, 1-1/res, steps = res, device = device) for _ in range(2)],
        indexing = 'ij'
        )
    G = torch.stack([G[1],G[0]],dim=-1).reshape(C,2)
    # FIXME: below softmax line causes mps to bug, don't know why.
    if device.type == 'mps':
        cls = cls.log_softmax(dim=1).exp()
    else:
        cls = cls.softmax(dim=1)
    mode = cls.max(dim=1).indices
    
    index = torch.stack((mode-1, mode, mode+1, mode - res, mode + res), dim = 1).clamp(0,C - 1).long()
    neighbours = torch.gather(cls, dim = 1, index = index)[...,None]
    flow = neighbours[:,0] * G[index[:,0]] + neighbours[:,1] * G[index[:,1]] + neighbours[:,2] * G[index[:,2]] + neighbours[:,3] * G[index[:,3]] + neighbours[:,4] * G[index[:,4]]
    tot_prob = neighbours.sum(dim=1)  
    flow = flow / tot_prob
    return flow



def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch



def get_gt_warp(depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode = 'bilinear', relative_depth_error_threshold = 0.05, H = None, W = None):
    
    if H is None:
        B,H,W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(
                    -1 + 1 / n, 1 - 1 / n, n, device=depth1.device
                )
                for n in (B, H, W)
            ],
            indexing = 'ij'
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            depth_interpolation_mode = depth_interpolation_mode,
            relative_depth_error_threshold = relative_depth_error_threshold,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return x2, prob

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, smooth_mask = False, return_relative_depth_error = False, depth_interpolation_mode = "bilinear", relative_depth_error_threshold = 0.05):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    if depth_interpolation_mode == "combined":
        # Inspired by approach in inloc, try to fill holes from bilinear interpolation by nearest neighbour interpolation
        if smooth_mask:
            raise NotImplementedError("Combined bilinear and NN warp not implemented")
        valid_bilinear, warp_bilinear = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "bilinear",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        valid_nearest, warp_nearest = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "nearest-exact",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        nearest_valid_bilinear_invalid = (~valid_bilinear).logical_and(valid_nearest) 
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp
        
        
    kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode = depth_interpolation_mode, align_corners=False)[
        :, 0, :, 0
    ]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    # Sample depth, get calculable_mask on depth != 0
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    kpts0_cam = kpts0_n

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_kpts0[~covisible_mask, :] = -5 # xd

    w_kpts0_depth = F.grid_sample(
        depth1[:, None], w_kpts0[:, :, None], mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]
    
    relative_depth_error = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs()
    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = (-relative_depth_error/smooth_mask).exp()
    valid_mask = nonzero_mask * covisible_mask * consistent_mask
    if return_relative_depth_error:
        return relative_depth_error, w_kpts0
    else:
        return valid_mask, w_kpts0

def get_amp_dtype(amp: bool) -> torch.dtype:
    if amp:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32


def geometric_dist(depth1, depth2, T_1to2, K1, K2, dense_matches):
    b, h1, w1, d = dense_matches.shape
    with torch.no_grad():
        x1 = dense_matches[..., :2].reshape(b, h1 * w1, 2)
        mask, x2 = warp_kpts(
            x1.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
        )
        x2 = torch.stack(
            (w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1
        )
        prob = mask.float().reshape(b, h1, w1)
    x2_hat = dense_matches[..., 2:]
    x2_hat = torch.stack(
        (w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1
    )
    gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
    gd = gd[prob == 1]
    robustness = (gd < 32.0).float().mean()
    pck_1 = (gd < 1.0).float().mean()
    pck_3 = (gd < 3.0).float().mean()
    pck_5 = (gd < 5.0).float().mean()
    return gd, robustness, pck_1, pck_3, pck_5, prob


def dense_benchmark(model, dataset, batch_size=8, num_samples=2000):
    model.train(False)
    with torch.no_grad():
        metrics = {
            key: {
                "gd_tot": 0.0,
                "robustness_tot": 0.0,
                "pck_1_tot": 0.0,
                "pck_3_tot": 0.0,
                "pck_5_tot": 0.0,
            }
            for key in model.keys
        }
        
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.ones(len(dataset)), replacement=False, num_samples=num_samples
        )
        B = batch_size
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=B, num_workers=batch_size, sampler=sampler
        )
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # for idx, data in tqdm.tqdm(enumerate(dataloader), disable = romatch.RANK > 0):
        for idx, data in enumerate(metric_logger.log_every(dataloader, 10, header)):
            im_A, im_B, depth1, depth2, T_1to2, K1, K2 = (
                data["im_A"].cuda(),
                data["im_B"].cuda(),
                data["im_A_depth"].cuda(),
                data["im_B_depth"].cuda(),
                data["T_1to2"].cuda(),
                data["K1"].cuda(),
                data["K2"].cuda(),
            )
            model_out = model.match(im_A, im_B)

            for key, (matches, certainty) in model_out.items():
                gd, robustness, pck_1, pck_3, pck_5, prob = geometric_dist(
                    depth1, depth2, T_1to2, K1, K2, matches
                )
                metrics[key]["gd_tot"] += gd.mean().item()
                metrics[key]["robustness_tot"] += robustness.item()
                metrics[key]["pck_1_tot"] += pck_1.item()
                metrics[key]["pck_3_tot"] += pck_3.item()
                metrics[key]["pck_5_tot"] += pck_5.item()
            metric_logger.update(epe=gd.mean().item(), robustness=robustness)
            if romatch.DEBUG_MODE:
                from romatch.utils.utils import tensor_to_pil
                import torch.nn.functional as F
                path = "vis"
                H, W = model.get_output_resolution()
                white_im = torch.ones((B,1,H,W),device="cuda")
                im_B_transfer_rgb = F.grid_sample(
                    im_B.cuda(), matches[:,:,:W, 2:], mode="bilinear", align_corners=False
                )
                warp_im = im_B_transfer_rgb
                c_b = certainty[:,None]#(certainty*0.9 + 0.1*torch.ones_like(certainty))[:,None]
                vis_im = c_b * warp_im + (1 - c_b) * white_im
                for b in range(B):
                    import os
                    os.makedirs(f"{path}/{model.name}/{idx}_{b}_{H}_{W}",exist_ok=True)
                    tensor_to_pil(vis_im[b], unnormalize=True).save(
                        f"{path}/{model.name}/{idx}_{b}_{H}_{W}/warp.jpg")
                    tensor_to_pil(im_A[b].cuda(), unnormalize=True).save(
                        f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_A.jpg")
                    tensor_to_pil(im_B[b].cuda(), unnormalize=True).save(
                        f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_B.jpg")

    num_batches = len(dataloader)
    results = {
        str(key): {
            "epe": m["gd_tot"] / num_batches,
            "robustness": m["robustness_tot"] / num_batches,
            "mega_pck_1": m["pck_1_tot"] / num_batches,
            "mega_pck_3": m["pck_3_tot"] / num_batches,
            "mega_pck_5": m["pck_5_tot"] / num_batches,
        }
        for key, m in metrics.items()
    }
    return results


# For ScanNet-1500 taken from https://github.com/mbanani/probe3d
def torch_knn(query, target, k=1, metric="euclidean"):
    if metric == "cosine":
        query = F.normalize(query, dim=-1)
        target = F.normalize(target, dim=-1)

    # Compute pairwise distance/similarity
    if metric == "euclidean":
        dists = torch.cdist(query, target, p=2)  # [N, M]
    else:  # cosine → use dot product (larger = closer)
        dists = 1 - torch.matmul(query, target.T)

    # Get k smallest distances
    dists, indices = torch.topk(dists, k, dim=-1, largest=False)
    return dists, indices

def knn_points(X_f, Y_f, K=1, metric="euclidean"):
    """
    Finds the kNN according to either euclidean distance or cosine distance. This is
    tricky since PyTorch3D's fast kNN kernel does euclidean distance, however, we can
    take advantage of the relation between euclidean distance and cosine distance for
    points sampled on an n-dimension sphere.

    Using the quadratic expansion, we find that finding the kNN between two normalized
    is the same regardless of whether the metric is euclidean distance or cosine
    similiarity.

        -2 * xTy = (x - y)^2 - x^2 - y^2
        -2 * xtY = (x - y)^2 - 1 - 1
        - xTy = 0.5 * (x - y)^2 - 1

    Hence, the metric that would maximize cosine similarity is the same as that which
    would minimize the euclidean distance between the points, with the distances being
    a simple linear transformation.
    """
    assert metric in ["cosine", "euclidean"]
    if metric == "cosine":
        X_f = torch.nn.functional.normalize(X_f, dim=-1)
        Y_f = torch.nn.functional.normalize(Y_f, dim=-1)

    _, X_nn = torch_knn(X_f, Y_f, K)

    # n_points x k x F
    X_f_nn = Y_f[X_nn]

    if metric == "euclidean":
        dists = (X_f_nn - X_f[:, None, :]).norm(p=2, dim=3)
    elif metric == "cosine":
        dists = 1 - cosine_similarity(X_f_nn, X_f[:, None, :], dim=-1)

    return dists, X_nn


def get_correspondences_ratio_test(
    P1_F, P2_F, num_corres, metric="cosine", bidirectional=False, ratio_test=True
):
    # Calculate kNN for k=2; both outputs are (N, P, K)
    # idx_1 returns the indices of the nearest neighbor in P2
    # output is cosine distance (0, 2)
    K = 2

    dists_1, idx_1 = knn_points(P1_F, P2_F, K, metric)
    idx_1 = idx_1[..., 0]
    if ratio_test:
        weights_1 = calculate_ratio_test(dists_1)
    else:
        weights_1 = dists_1[:, 0]

    # Take the nearest neighbor for the indices for k={1, 2}
    if bidirectional:
        dists_2, idx_2 = knn_points(P2_F, P1_F, K, metric)
        idx_2 = idx_2[..., 0]
        if ratio_test:
            weights_2 = calculate_ratio_test(dists_2)
        else:
            weights_2 = dists_2[:, 0]

        # Get topK matches in both directions
        m12_idx1, m12_idx2, m12_dist = get_topk_matches(
            weights_1, idx_1, num_corres // 2
        )
        m21_idx2, m21_idx1, m21_dist = get_topk_matches(
            weights_2, idx_2, num_corres // 2
        )

        # concatenate into correspondences and weights
        all_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1)
        all_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1)
        all_dist = torch.cat((m12_dist, m21_dist), dim=1)
    else:
        all_idx1, all_idx2, all_dist = get_topk_matches(weights_1, idx_1, num_corres)

    return all_idx1, all_idx2, all_dist


@torch.jit.script
def calculate_ratio_test(dists: torch.Tensor):
    """
    Calculate weights for matches based on the ratio between kNN distances.

    Input:
        (N, P, 2) Cosine Distance between point and nearest 2 neighbors
    Output:
        (N, P, 1) Weight based on ratio; higher is more unique match
    """
    # Ratio -- close to 0 is completely unique; 1 is same feature
    # Weight -- Convert so that higher is more unique
    # clamping because some dists will be 0 (when not in the pointcloud
    dists = dists.clamp(min=1e-9)
    ratio = dists[..., 0] / dists[..., 1].clamp(min=1e-9)
    weight = 1 - ratio
    return weight


# @torch.jit.script
def get_topk_matches(dists, idx, num_corres: int):
    num_corres = min(num_corres, dists.shape[-1])
    dist, idx_source = torch.topk(dists, k=num_corres, dim=-1)
    idx_target = idx[idx_source]
    return idx_source, idx_target, dist


def get_grid(H: int, W: int):
    # Generate a grid that's equally spaced based on image & embed size
    grid_x = torch.linspace(0.5, W - 0.5, W)
    grid_y = torch.linspace(0.5, H - 0.5, H)

    xs = grid_x.view(1, W).repeat(H, 1)
    ys = grid_y.view(H, 1).repeat(1, W)
    zs = torch.ones_like(xs)

    # Camera coordinate frame is +xyz (right, down, into-camera)
    # Dims: 3 x H x W
    grid_xyz = torch.stack((xs, ys, zs), dim=0)
    return grid_xyz


def grid_to_pointcloud(K_inv, depth, grid=None):
    _, H, W = depth.shape

    if grid is None:
        grid = get_grid(H, W)

    # Apply inverse projection
    points = depth * grid

    # Invert intriniscs
    points = points.view(3, H * W)
    points = K_inv @ points
    points = points.permute(1, 0)

    return points


def sample_pointcloud_features(feats, K, pc, image_shape):
    H, W = image_shape
    uvd = pc @ K.transpose(-1, -2)
    uv = uvd[:, :2] / uvd[:, 2:3].clamp(min=1e-9)

    uv[:, 0] = (2 * uv[:, 0] / W) - 1
    uv[:, 1] = (2 * uv[:, 1] / H) - 1

    # sample points
    pc_F = F.grid_sample(feats[None], uv[None, None], align_corners=False)
    pc_F = pc_F[:, :, 0].transpose(1, 2)[0]

    return pc_F


def argmax_2d(x, max_value=True):
    h, w = x.shape[-2:]
    x = torch.flatten(x, start_dim=-2)
    if max_value:
        flat_indices = x.argmax(dim=-1)
    else:
        flat_indices = x.argmin(dim=-1)

    min_row = flat_indices // w
    min_col = flat_indices % w
    xy_indices = torch.stack((min_col, min_row), dim=-1)
    return xy_indices


def project_3dto2d(xyz, K_mat):
    uvd = xyz @ K_mat.transpose(-1, -2)
    uv = uvd[:, :2] / uvd[:, 2:3].clamp(min=1e-9)
    return uv


def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return aucs


def estimate_correspondence_depth(feat_0, feat_1, depth_0, depth_1, K, num_corr=500):
    xyz_0 = grid_to_pointcloud(K.inverse(), depth_0)
    xyz_1 = grid_to_pointcloud(K.inverse(), depth_1)
    xyz_0 = xyz_0[xyz_0[:, 2] > 0]
    xyz_1 = xyz_1[xyz_1[:, 2] > 0]

    feat_0 = sample_pointcloud_features(feat_0, K.clone(), xyz_0, depth_0.shape[-2:])
    feat_1 = sample_pointcloud_features(feat_1, K.clone(), xyz_1, depth_1.shape[-2:])

    idx0, idx1, corr_dist = get_correspondences_ratio_test(feat_0, feat_1, num_corr)

    corr_xyz0 = xyz_0[idx0]
    corr_xyz1 = xyz_1[idx1]

    return corr_xyz0, corr_xyz1, corr_dist


def estimate_correspondence_xyz(
    feat_0, feat_1, xyz_grid_0, xyz_grid_1, num_corr=500, ratio_test=True
):
    # upsample feats
    _, h, w = xyz_grid_0.shape
    feat_0 = F.interpolate(feat_0[None], size=(h, w), mode="bicubic")[0]
    feat_1 = F.interpolate(feat_1[None], size=(h, w), mode="bicubic")[0]

    uvd_0 = get_grid(h, w).to(xyz_grid_0)
    uvd_1 = get_grid(h, w).to(xyz_grid_1)

    # only keep values with real points
    feat_0 = feat_0.permute(1, 2, 0)[xyz_grid_0[2] > 0]
    feat_1 = feat_1.permute(1, 2, 0)[xyz_grid_1[2] > 0]
    xyz_0 = xyz_grid_0.permute(1, 2, 0)[xyz_grid_0[2] > 0]
    xyz_1 = xyz_grid_1.permute(1, 2, 0)[xyz_grid_1[2] > 0]
    uvd_0 = uvd_0.permute(1, 2, 0)[xyz_grid_0[2] > 0]
    uvd_1 = uvd_1.permute(1, 2, 0)[xyz_grid_1[2] > 0]

    idx0, idx1, c_dist = get_correspondences_ratio_test(
        feat_0, feat_1, num_corr, ratio_test=ratio_test
    )

    c_xyz0 = xyz_0[idx0]
    c_xyz1 = xyz_1[idx1]
    c_uv0 = uvd_0[idx0][:, :2]
    c_uv1 = uvd_1[idx1][:, :2]

    return c_xyz0, c_xyz1, c_dist, c_uv0, c_uv1


def compute_binned_performance(y, x, x_bins):
    """
    Given two arrays: (x, y), compute the mean y value for specific x_bins
    """
    y_binned = []
    for i in range(len(x_bins) - 1):
        x_min = x_bins[i]
        x_max = x_bins[i + 1]
        x_mask = (x >= x_min) * (x < x_max)
        y_binned.append(y[x_mask].mean())

    return y_binned


def transform_points_Rt(
    points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    R = viewpoint[..., :3, :3]
    t = viewpoint[..., None, :3, 3]
    # N.B. points is (..., n, 3) not (..., 3, n)
    if inverse:
        return (points - t) @ R
    else:
        return points @ R.transpose(-2, -1) + t


def so3_relative_angle(R1: torch.Tensor, R2: torch.Tensor, eps: float = 1e-4):
    """
    based on pytorch3d
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12, eps=eps)


def so3_rotation_angle(R: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Based on pytorch3d version
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5
    return torch.acos(phi_cos.clamp(min=-1, max=1))

def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)

def from_homogeneous(x: torch.Tensor) -> torch.Tensor:
    return x[..., :-1] / x[..., -1:]

def get_normalized_grid(
    B: int, H: int, W: int,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device="cuda") for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n

def get_pixel_grid(
    B: int, H: int, W: int, overload_device: torch.device | None = None,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[torch.arange(n, device=overload_device or "cuda") + 0.5 for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n