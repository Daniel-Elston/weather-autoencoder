from __future__ import annotations

from dataclasses import dataclass

import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


@dataclass
class ModelParams:
    device: str = device
    epochs: int = 10
    input_dim: int = 365
    latent_dims: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    dampening: float = 0.0
