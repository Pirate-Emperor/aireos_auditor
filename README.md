
# Multi-Modal Auditor

## Overview

The **Multi-Modal Auditor** is an autonomous pipeline designed to evaluate, sanitize, and restore multi-modal datasets. It integrates three primary subsystems:
1. **Optical Flow Variance (RAFT):** For identifying and calculating temporal diffusion artifacts.
2. **3D Topological Heuristics (Mesh Evaluation):** For verifying manifold integrity and computing accuracy/completeness distances between 3D geometries.
3. **Video Restoration Transformer (VRT):** For spatio-temporal benchmarking, super-resolution, and deblurring of sequential frames.

---

## Part I: Recurrent All-Pairs Field Transforms (RAFT)

This module contains the optical flow calculation engine to evaluate temporal variances across sequential frames.

### Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```shell
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch

```

### Demos

Pretrained models can be downloaded by running:

```shell
./download_models.sh

```

You can demo a trained model on a sequence of frames:

```shell
python demo.py --model=models/raft-things.pth --path=demo-frames

```

### Required Data

To evaluate/train the optical flow module, you will need to download the required datasets (FlyingChairs, FlyingThings3D, Sintel, KITTI, HD1K). By default `datasets.py` will search for the datasets in these locations:

```text
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow

```

### Evaluation & Training

You can evaluate a trained model using `evaluate.py`:

```shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision

```

Training logs will be written to the `runs` directory which can be visualized using TensorBoard:

```shell
./train_standard.sh

```

*(Optional) Efficient Implementation:* You can use the alternate (efficient) implementation by compiling the provided CUDA extension:

```shell
cd alt_cuda_corr && python setup.py install && cd ..

```

Note: This implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.

---

## Part II: 3D Mesh Topology Evaluation

This is a parallel C++ implementation for efficiently computing distances (in particular, accuracy and completeness) between meshes or between point clouds and meshes.

The main use case is evaluating 3D (surface) reconstruction or shape completion algorithms that generate a mesh as output which is then evaluated against a mesh or point cloud ground truth. Accuracy is the distance of the reconstruction (i.e. the input) to the ground truth (i.e. the reference); completeness is the distance from ground truth to reconstruction.

### Installation

Requirements for C++ tool:

* CMake, Boost, Eigen, OpenMP, C++11.

Requirements for Python tools:

* Numpy, h5py, skimage (or PyMCubes).

On Ubuntu and related Linux distributions:

```shell
sudo apt-get install build-essential cmake libboost-all-dev libeigen3-dev
pip install numpy

```

To build, first adapt `cmake/FindEigen3.cmake` to include the correct path to Eigen3's include directory, and run:

```shell
mkdir build
cd build
cmake ..
make

```

### Usage

Using the `--help` option will give a detailed summary of available options:

```text
$ ../bin/evaluate --help
Allowed options:
  --help                  produce help message
  --input arg             input, either single OFF file or directory containing OFF files
  --reference arg         reference, either single OFF or TXT file or directory
  --output arg            output file, a TXT file containing accuracy and completeness
  --n_points arg (=10000) number points to sample from meshes

```

---

## Part III: Video Restoration Transformer (VRT)

This module tackles video restoration by exploiting long-range temporal dependency modelling. VRT is composed of multiple scales, each of which consists of two kinds of modules: temporal mutual self attention (TMSA) and parallel warping.

TMSA divides the video into small clips, on which mutual attention is applied for joint motion estimation, feature alignment and feature fusion, while self-attention is used for feature extraction.

### Quick Testing

Following commands will download pretrained models and test datasets automatically. If out-of-memory, try to reduce `--tile` at the expense of slightly decreased performance.

```bash
# video sr trained on REDS (6 frames), tested on REDS4
python main_test_vrt.py --task 001_VRT_videosr_bi_REDS_6frames --folder_lq testsets/REDS4/sharp_bicubic --folder_gt testsets/REDS4/GT --tile 40 128 128 --tile_overlap 2 20 20

# video deblurring trained and tested on DVD
python main_test_vrt.py --task 005_VRT_videodeblurring_DVD --folder_lq testsets/DVD10/test_GT_blurred --folder_gt testsets/DVD10/test_GT --tile 12 256 256 --tile_overlap 2 20 20

# video denoising trained on DAVIS (noise level 0-50), tested on Set8 and DAVIS
python main_test_vrt.py --task 008_VRT_videodenoising_DAVIS --sigma 10 --folder_lq testsets/Set8 --folder_gt testsets/Set8 --tile 12 256 256 --tile_overlap 2 20 20

# video frame interpolation trained on Vimeo
python main_test_vrt.py --task 009_VRT_videofi_Vimeo_4frames --folder_lq testsets/UCF101 --folder_gt testsets/UCF101 --tile 0 0 0 --tile_overlap 0 0 0

```

### Dataset Preparation

For better I/O speed, convert `.png` datasets to `.lmdb` datasets.

| Task | Training Set | Testing Set |
| --- | --- | --- |
| **Video SR** | REDS sharp & sharp_bicubic | REDS4 (4 videos, 400 frames) |
| **Video Deblurring** | DVD (61 videos, 5708 frames) | DVD (10 videos, 1000 frames) |
| **Video Denoising** | DAVIS-2017 (90 videos) | DAVIS-2017-test + Set8 |
| **Frame Interpolation** | Vimeo90K | Vimeo90K-T + UCF101 |

## License

This project is licensed under the Pirate-Emperor License. See the [LICENSE](LICENSE) file for details.

## Author

**Pirate-Emperor**

[![Twitter](https://skillicons.dev/icons?i=twitter)](https://twitter.com/PirateKingRahul)
[![Discord](https://skillicons.dev/icons?i=discord)](https://discord.com/users/1200728704981143634)
[![LinkedIn](https://skillicons.dev/icons?i=linkedin)](https://www.linkedin.com/in/piratekingrahul)

[![Reddit](https://img.shields.io/badge/Reddit-FF5700?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/u/PirateKingRahul)
[![Medium](https://img.shields.io/badge/Medium-42404E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@piratekingrahul)

- GitHub: [Pirate-Emperor](https://github.com/Pirate-Emperor)
- Reddit: [PirateKingRahul](https://www.reddit.com/u/PirateKingRahul/)
- Twitter: [PirateKingRahul](https://twitter.com/PirateKingRahul)
- Discord: [PirateKingRahul](https://discord.com/users/1200728704981143634)
- LinkedIn: [PirateKingRahul](https://www.linkedin.com/in/piratekingrahul)
- Skype: [Join Skype](https://join.skype.com/invite/yfjOJG3wv9Ki)
- Medium: [PirateKingRahul](https://medium.com/@piratekingrahul)

Thank you for visiting this project!

---