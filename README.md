# MLX, Core ML and Torch Performance on Apple Silicon
MLX vs PyTorch Performance on Apple Silicon for Training
Core ML vs PyTorch Performance on Apple Silicon for Inference

## Introduction

Comparing MLX and PyTorch on Apple Silicon

MLX uses Metal Performance Shaders (MPS) and PyTorch uses Metal Performance Shaders (MPS) and Metal Performance Shaders Graph (MPSGraph) on Apple Silicon.

PyTorch only uses CPU on Apple Silicon.

Comparing Core ML vs PyTorch on Apple Silicon

Core ML models uses the Neural Engine as well as the GPU for computation.

## Environment
Python 3.10.15 on conda-forge

Clang 17.0.6 on darwin

### Libraries
- PyTorch: torch 2.4.0
- TorchVision: torchvision 0.19.0
- Transformers: transformers 4.46.2
- MLX: mlx 0.20.0
- MLX-data: mlx-data 0.0.2
- coremltools: 8.1
- NumPy: numpy 2.0.2

## Results Training
| device | processor | ram  | model     | batch_size | epochs | mlx (s) | torch (s) |
|:-------|:----------|:----:|:---------:|:----------:|:------:|:-------:|:---------:|
| MBA    | M1        | 8 GB | resnet-18 | 32         | 2      | 108.29  | 1176.04   |
| MBP    | M1        |16 GB | resnet-18 | 32         | 2      |         |           |
| MBA    | M2        | 8 GB | resnet-18 | 32         | 2      | 59.11   | 1036.72   |

## Results Inference
| device | processor | ram  |  images  |   model   |  coreML    | torch  |
|:-------|:----------|:----:|:--------:|:---------:|:----------:|:------:|
| MBA    | M1        | 8 GB | 2023     | resnet-152| 8.15       | 119.5  |
| MBP    | M1        |16 GB | 2023     | resnet-152| 4.88       | 70.45  |
| MBA    | M2        | 8 GB | 2023     | resnet-152| NIL        | NIL    | 