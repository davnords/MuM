<div align="center">
<h1>MuM: Multi-View Masked Image Modeling for 3D Vision</h1>

<a href="https://arxiv.org/abs/2511.17309"><img src="https://img.shields.io/badge/arXiv-2511.17309-b31b1b" alt="arXiv"></a>

**Chalmers University of Technology**; **Linköping University**; **University of Amsterdam**

[David Nordström](https://scholar.google.com/citations?user=-vJPE04AAAAJ), [Johan Edstedt](https://scholar.google.com/citations?user=Ul-vMR0AAAAJ&hl), [Fredrik Kahl](https://scholar.google.com/citations?user=P_w6UgMAAAAJ), [Georg Bökman](https://scholar.google.com/citations?user=FUE3Wd0AAAAJ)
</div>

## Updates

- [March 4, 2026] Accepted to CVPR26. 
- [November 22, 2025] Initial release of inference code. 

## Overview

MuM is a Transformer (ViT-L) that extracts highly geometric features from an image. It includes an encoder-decoder structure and for most purposes the decoder is discarded. We show in the paper that MuM beats DINOv3 and CroCo v2 on multi-view tasks such as feedforward 3D reconstruction, matching, and relative pose estimation.

## Encoder

The easiest way to use our trained model is to extract patch features solely through the encoder. We use a similar API as DINOv3, e.g., 
```python
from mum import mum_vitl16
from mum.utils import transform_image

model = mum_vitl16(pretrained=True).cuda().eval()
images = transform_image("assets/wma1.jpg").unsqueeze(0).cuda()
feats = model.forward_features(images)['x_norm_patchtokens'] # B, P, 1024
```

## Encoder + Decoder

The full model (including decoder) performs normalized pixel reconstruction in its forward pass. It can also be easily finetuned for any multi-view task. Here is an example use-case:
```python
from mum import mum_vitl16_decoderb
from mum.utils import qualitative_evaluation, transform_image
import torch

IMAGE_LIST = ["assets/wma1.jpg", "assets/wma2.jpg", "assets/wma3.jpg"]
model = mum_vitl16_decoderb(pretrained=True).cuda().eval()
images = torch.stack([transform_image(image_path, size=(256, 256)) for image_path in IMAGE_LIST]).unsqueeze(0).cuda()
# Two example usages:
# (1) Compute reconstruction loss.
loss, pred, mask = model(images, mask_ratio=0.75) # loss should be ~0.23
# (2) Pixel reconstruction.
qualitative_evaluation(model, images, "assets/reconstruction.png")
```
This should give something like:
<p>
  <img src="assets/reconstruction.png" height="300">
</p>

## Pretrain

[COMING]: We aim to release this soon.

## Evaluation

[COMING]: We aim to release all our evaluations in a separate branch.

## Setup/Install

For this release (solely inference code), you really just need a modern `torch` and `torchvision` version. I use `2.6.0+cu118` and `0.21.0+cu118`, respectively, together with Python 3.11.9. Using this, I can replicate the reported evaluations (to be released shortly), but any new version should work fine. For visualizations, you also need `matplotlib`.  

To most easily achieve this, you can e.g. run:
```bash
conda create -n mum python=3.11.9
conda activate mum

# Any modern version should work fine, this is what I use 
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# Optional, needed for running code examples
pip install matplotlib
```

## Training

⚠️ WARNING THIS BRANCH IS PRELIMINARY AND HAS NOT BEEN TESTED. 

To train MuM you need to download the relevant datasets. We provide provide code for training on MegaDepth, however, training on the other datasets follows a similar protocol. We assume you have downloaded MegaDepth and pre-processed it in accordange with e.g. [RoMa](https://github.com/Parskatt/RoMa) and [DKM](https://github.com/parskatt/dkm). You also need to either create your own annotations (that describe the sampling of image sequences) or download ours. We have posted the annotaions for all datasets [here](https://github.com/davnords/MuM/releases/tag/annotations). We provide some unorganized data preprocessing scripts in [experiments/preprocess](experiments/preprocess).

The full MuM model is trained on 32xA100. We ran something like:
```bash
python experiments/train.py --config-file mum/configs/mum.yaml --name mum --ngpus 8 --nodes 4 --track-wandb
```

## Checklist

- [x] Provide basic feature extraction inference code
- [x] Release the pre-training code
- [x] Provide data processing scripts and annotation files
- [ ] Release all evaluations

## License

Code is MIT licensed. Code for layers is directly descending from DINOv3. DINOv3 has a custom license, see [DINOv3](https://github.com/facebookresearch/dinov3/tree/main?tab=License-1-ov-file#readme).

## Acknowledgement

Our code is largely inspired by [VGGT](https://github.com/facebookresearch/vggt), [DINOv3](https://github.com/facebookresearch/dinov3), [CroCo](https://github.com/naver/croco), and [MAE](https://github.com/facebookresearch/mae).

## BibTeX

If you find our work useful, please consider citing our paper!
```bibtex
@inproceedings{nordström2025mum,
  title={MuM: Multi-View Masked Image Modeling for 3D Vision}, 
  author={David Nordström and Johan Edstedt and Fredrik Kahl and Georg Bökman},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
