import enum
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NeTIBatch:
    input_ids: torch.Tensor
    placeholder_token_id: int
    timesteps: torch.Tensor
    unet_layers: torch.Tensor
    truncation_idx: Optional[int] = None

@dataclass
class NeTIVPsBatch(NeTIBatch):
    azimuths: torch.Tensor = None
    elevations: torch.Tensor = None


@dataclass
class PESigmas:
    sigma_t: float
    sigma_l: float
    sigma_az: float = 0.1
    sigma_el: float = 0.1
