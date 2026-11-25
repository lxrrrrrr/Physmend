"""Inference-only local-Q execution for OmniWeaving's double-stream DiT.

No weights are modified. Static context is constructed at t=0 once per run.
The sparse and dense-static paths use exactly the same background memory.
"""
from dataclasses import dataclass

import torch
from .attention import RectangularAttention


def apply_rope(q, k, freqs):
    if freqs is None:
        return q, k
    if not isinstance(freqs, tuple):
        raise ValueError("Local editing expects real (cos, sin) rotary embeddings")
    cos, sin = (f.to(q.device)[None, :, None, :] for f in freqs)
    def rotate(x):
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        half = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
        return (x.float() * cos + half * sin).to(x.dtype)
    return rotate(q), rotate(k)


def modulation(normed, shift, scale):
    return normed * (1 + scale[:, None]) + shift[:, None]


def project_video(block, img, vec, freqs):
    mods = block.img_mod(vec).chunk(6, -1)
    x = modulation(block.img_norm1(img), mods[0], mods[1])
    b, n, d = x.shape
    shape = (b, n, block.heads_num, d // block.heads_num)
    v = block.img_attn_v(x).reshape(shape)
    q = block.img_attn_q_norm(block.img_attn_q(x).reshape(shape)).to(v)
    k = block.img_attn_k_norm(block.img_attn_k(x).reshape(shape)).to(v)
    q, k = apply_rope(q, k, freqs)
    return q, k, v, mods


def project_text(block, txt, vec):
    mods = block.txt_mod(vec).chunk(6, -1)
    x = modulation(block.txt_norm1(txt), mods[0], mods[1])
    b, n, d = x.shape
    shape = (b, n, block.heads_num, d // block.heads_num)
    v = block.txt_attn_v(x).reshape(shape)
    q = block.txt_attn_q_norm(block.txt_attn_q(x).reshape(shape)).to(v)
    k = block.txt_attn_k_norm(block.txt_attn_k(x).reshape(shape)).to(v)
    return q, k, v, mods


@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor

    @property
    def nbytes(self):
        return self.key.numel() * self.key.element_size() + self.value.numel() * self.value.element_size()


def double_block(block, img, txt, vec, freqs, text_mask, attention,
                 edit_indices=None, background_indices=None, cached=None, capture=False):
    iq, ik, iv, im = project_video(block, img, vec, freqs)
    tq, tk, tv, tm = project_text(block, txt, vec)
    new_cache = None
    if capture:
        # index_select owns storage; it must not retain the full source K/V allocation.
        new_cache = LayerKV(
            ik.index_select(1, background_indices).detach(),
            iv.index_select(1, background_indices).detach())
    if cached is not None:
        if edit_indices is not None:  # Dense-static still computes every video query/FFN.
            ik = ik.index_select(1, edit_indices)
            iv = iv.index_select(1, edit_indices)
        ik = torch.cat((cached.key, ik), 1)
        iv = torch.cat((cached.value, iv), 1)
    n_image = img.shape[1]
    out = attention(
        torch.cat((iq, tq), 1), torch.cat((ik, tk), 1), torch.cat((iv, tv), 1),
        text_mask, video_queries=n_image, video_keys=ik.shape[1])
    ia, ta = out[:, :n_image], out[:, n_image:]
    img = img + block.img_attn_proj(ia) * im[2][:, None]
    img = img + block.img_mlp(modulation(block.img_norm2(img), im[3], im[4])) * im[5][:, None]
    txt = txt + block.txt_attn_proj(ta) * tm[2][:, None]
    txt = txt + block.txt_mlp(modulation(block.txt_norm2(txt), tm[3], tm[4])) * tm[5][:, None]
    return img, txt, new_cache


class LocalEditState:
    def __init__(self, edit_mask, mode="sparse", attention="auto"):
        if edit_mask.ndim != 3 or edit_mask.dtype != torch.bool:
            raise ValueError("edit_mask must be a boolean latent-grid [T,H,W] tensor")
        if mode not in {"sparse", "dense-static", "full"}:
            raise ValueError("mode must be sparse, dense-static, or full")
        self.mask = edit_mask
        self.grid = tuple(edit_mask.shape)
        self.edit_indices = edit_mask.flatten().nonzero().flatten()
        self.background_indices = (~edit_mask).flatten().nonzero().flatten()
        if not self.edit_indices.numel():
            raise ValueError("The edit mask is empty")
        self.mode = mode
        self.phase = "denoise"
        self.layers = []
        self._full_freqs = None
        self._edit_freqs = None
        self.attention = RectangularAttention(attention)

    @property
    def cache_bytes(self):
        return sum(layer.nbytes for layer in self.layers)

    @property
    def statistics(self):
        return dict(mode=self.mode, total_video_tokens=self.mask.numel(),
                    edit_video_tokens=self.edit_indices.numel(),
                    background_video_tokens=self.background_indices.numel(),
                    edit_fraction=self.edit_indices.numel() / self.mask.numel(),
                    static_kv_bytes=self.cache_bytes, cached_layers=len(self.layers),
                    attention_backend=self.attention.backend)

    def forward_embedded(self, model, img, txt, vec, freqs, text_mask,
                         grid, stack_states=None):
        if tuple(grid) != self.grid:
            raise ValueError(f"Edit mask grid {self.grid} differs from model grid {grid}")
        if len(model.single_blocks):
            raise ValueError("Local editor currently supports double-stream-only checkpoints")
        if tuple(model.patch_size) != (1, 1, 1):
            raise ValueError("Local editor currently requires patch_size=[1,1,1]")
        if model.training:
            raise RuntimeError("This prototype is inference-only; call transformer.eval()")
        if freqs is None:
            raise ValueError("Absolute video-grid rotary embeddings are required")
        if self.edit_indices.device != img.device:
            self.edit_indices = self.edit_indices.to(img.device)
            self.background_indices = self.background_indices.to(img.device)
        if self._full_freqs is None:
            self._full_freqs = tuple(f.to(img.device) for f in freqs)
        freqs = self._full_freqs
        prefill = self.phase == "prefill"
        sparse = self.mode == "sparse" and not prefill
        use_static = self.mode != "full" and not prefill
        if use_static and len(self.layers) != len(model.double_blocks):
            raise RuntimeError("Static context must be prefilled once before denoising")
        if sparse:
            img = img.index_select(1, self.edit_indices)
            if self._edit_freqs is None:
                self._edit_freqs = tuple(f.index_select(0, self.edit_indices) for f in freqs)
            freqs = self._edit_freqs
        if prefill:
            self.layers.clear()
        for i, block in enumerate(model.double_blocks):
            img, txt, memory = double_block(
                block, img, txt, vec, freqs, text_mask, self.attention,
                edit_indices=self.edit_indices if use_static and not sparse else None,
                background_indices=self.background_indices,
                cached=self.layers[i] if use_static else None, capture=prefill)
            if memory is not None:
                self.layers.append(memory)
            if stack_states is not None and i < len(stack_states):
                n = stack_states[i].shape[-2]
                txt = torch.cat((txt[:, :-n], txt[:, -n:] + stack_states[i]), 1)
        if prefill:
            return (None, None)
        prediction = model.final_layer(img, vec)
        if sparse:
            full = prediction.new_zeros(prediction.shape[0], self.mask.numel(), prediction.shape[-1])
            full.index_copy_(1, self.edit_indices, prediction)
            prediction = full
        return (model.unpatchify(prediction, *grid), None)
