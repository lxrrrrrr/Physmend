"""Weight-free VAE metadata for training from precomputed posteriors only."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch


class CachedVAE(torch.nn.Module):
    def __init__(self, directory):
        super().__init__()
        self.config = SimpleNamespace(**json.loads((Path(directory)/'config.json').read_text()))
        self.dtype = torch.float16

    def encode(self, *args, **kwargs):
        raise RuntimeError('VAE weights omitted: cached training must supply posterior tensors')

    def decode(self, *args, **kwargs):
        raise RuntimeError('VAE weights omitted: cached training must supply the original first frame')

    def memory_efficient_context(self):
        raise RuntimeError('VAE computation is disabled during cached training')
