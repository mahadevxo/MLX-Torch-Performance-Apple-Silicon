# MLX vs Torch Performance Comparison on Apple Silicon
MLX vs PyTorch Performance on Apple Silicon

## Introduction

Comparing MLX and PyTorch on Apple Silicon

MLX uses Metal Performance Shaders (MPS) and PyTorch uses Metal Performance Shaders (MPS) and Metal Performance Shaders Graph (MPSGraph) on Apple Silicon.

PyTorch only uses CPU on Apple Silicon.

## Environment
Python 3.10.15 on conda-forge
Clang 17.0.6 on darwin

### Libraries
- PyTorch: torch 2.4.0
- TorchVision: torchvision 0.19.0
- Transformers: transformers 4.46.2
- MLX: mlx 0.20.0
- MLX-data: mlx-data 0.0.2
- NumPy: numpy 2.0.2

## Results
| device | processor | ram  | model     | batch_size | epochs | mlx (s) | torch (s) |
|:-------|:----------|:----:|:---------:|:----------:|:------:|:-------:|:---------:|
| MBA    | M1        | 8 GB | resnet-18 | 32         | 2      | 108.29  | -         |