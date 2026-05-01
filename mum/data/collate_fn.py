import torch
import random

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None, drop_masks=False):
    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])

    S = len(samples_list[0]["global_crops"][0])  # Number of crops per sample

    collated_global_crops = torch.stack([
        torch.stack([s["global_crops"][i][j] for j in range(S)])  # [S, C, H, W]
        for i in range(n_global_crops)
        for s in samples_list
    ])  # Final shape: [2 * B, S, C, H_glob, W_glob]

    collated_local_crops = torch.stack([
        torch.stack([s["local_crops"][i][j] for j in range(S)])  # [S, C, H, W]
        for i in range(n_local_crops)
        for s in samples_list
    ])  # Final shape: [n_local_crops * B, S, C, H_loc, W_loc]

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    if drop_masks:
        len_keep = int(N * (1 - mask_probability))
        
        noise = torch.rand(B, N)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        #ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], dtype=torch.bool)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        collated_masks = torch.gather(mask, dim=1, index=ids_restore)
        upperbound = N*B*mask_probability
    else:
        for i in range(0, n_samples_masked):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            # masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            masks_list.append(torch.stack([torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))) for _ in range(S)]))
            upperbound += int(N * prob_max)
        for i in range(n_samples_masked, B):
            # masks_list.append(torch.BoolTensor(mask_generator(0)))
            masks_list.append(torch.stack([torch.BoolTensor(mask_generator(0)) for _ in range(S)]))
        random.shuffle(masks_list)
        
        # collated_masks = torch.stack(masks_list).flatten(1)

        collated_masks = torch.stack(masks_list).flatten(2)  # shape: [B, S, n_tokens]
        # NOTE: We need to be carefule here so the structure of the masks is preserved. If things are not working, double check this.
        collated_masks = collated_masks.view(-1, collated_masks.size(-1))  # shape: [B * S, n_tokens]
        
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    return {
        "seq_names": [s["seq_name"] for s in samples_list],
        "ids": torch.stack([s["ids"] for s in samples_list]),

        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
