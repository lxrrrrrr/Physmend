"""Conservative pixel-to-latent masks, with causal temporal VAE alignment."""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_pixel_mask(path, frames, height, width, box=None, frame_range=None):
    if path is None:
        if box is None:
            raise ValueError("Specify --mask or --box")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("--box uses normalized x0 y0 x1 y1 in [0,1]")
        out = torch.zeros(frames, height, width, dtype=torch.bool)
        out[:, math.floor(y0*height):math.ceil(y1*height),
            math.floor(x0*width):math.ceil(x1*width)] = True
    else:
        path = Path(path)
        if path.is_dir():
            from PIL import Image
            images = sorted(path.glob("*.png"))
            if len(images) != frames:
                raise ValueError(f"Mask directory needs exactly {frames} PNG frames")
            value = np.stack([np.asarray(Image.open(p).convert("L")) for p in images])
        elif path.suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        elif path.suffix in {".pt", ".pth"}:
            value = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(value, torch.Tensor):
                raise ValueError("Mask PT file must contain a tensor")
        else:
            from PIL import Image
            value = np.asarray(Image.open(path).convert("L")).copy()
        out = torch.as_tensor(value)
        if out.ndim == 2:
            out = out[None].expand(frames, -1, -1)
        if tuple(out.shape) != (frames, height, width):
            raise ValueError(f"Mask must match prepared canvas {(frames,height,width)}, got {tuple(out.shape)}")
        out = out > 0
    if frame_range is not None:
        start, end = frame_range
        if not (0 <= start < end <= frames):
            raise ValueError("--frame-range must be a nonempty [start,end) interval")
        out = out.clone()
        out[:start] = False
        out[end:] = False
    if not out.any():
        raise ValueError("Pixel edit mask is empty")
    return out


def pixel_to_latent_mask(pixel_mask, spatial=16, temporal=4, dilation=0):
    frames, height, width = pixel_mask.shape
    if height % spatial or width % spatial or (frames - 1) % temporal:
        raise ValueError("Canvas must align to spatial VAE factor and frames=temporal*n+1")
    if dilation < 0:
        raise ValueError("Mask dilation cannot be negative")
    # Causal VAE: frame 0 is its own latent; frames 1..4 map to latent 1, etc.
    pooled = F.max_pool2d(pixel_mask.float()[:, None], spatial, spatial)[:, 0]
    if frames == 1:
        out = pooled
    else:
        tail = pooled[1:].reshape((frames-1)//temporal, temporal, height//spatial, width//spatial).amax(1)
        out = torch.cat((pooled[:1], tail), 0)
    if dilation:
        out = F.max_pool3d(out[None, None], 2*dilation+1, stride=1, padding=dilation)[0, 0]
    return out.bool()


def clamp_background(latents, source, mask):
    return torch.where(mask[None, None], latents, source)
