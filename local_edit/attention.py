"""Noncausal rectangular attention; query and KV sequence lengths may differ."""
import torch
import torch.nn.functional as F


class RectangularAttention:
    def __init__(self, backend="auto"):
        if backend not in {"auto", "flash", "sdpa"}:
            raise ValueError("attention backend must be auto, flash, or sdpa")
        self.requested = backend
        self.backend = None
        self._plans = {}
        self._flash = None
        self._flash_varlen = None

    def _select(self, q):
        if self.backend is not None:
            return
        if self.requested != "sdpa" and q.is_cuda:
            try:
                from flash_attn import flash_attn_func, flash_attn_varlen_func
                self._flash, self._flash_varlen = flash_attn_func, flash_attn_varlen_func
                self.backend = "flash"
                return
            except ImportError:
                if self.requested == "flash":
                    raise
        if self.requested == "flash":
            raise RuntimeError("FlashAttention requires CUDA and the existing flash_attn installation")
        self.backend = "sdpa"

    def __call__(self, q, k, v, text_mask, video_queries, video_keys):
        self._select(q)
        batch, nq, heads, dim = q.shape
        nk = k.shape[1]
        key = (batch, nq, nk, str(q.device))
        if key not in self._plans:
            text = text_mask.to(device=q.device, dtype=torch.bool)
            qm = torch.cat((torch.ones(batch, video_queries, device=q.device, dtype=torch.bool), text), 1)
            km = torch.cat((torch.ones(batch, video_keys, device=q.device, dtype=torch.bool), text), 1)
            if qm.shape[1] != nq or km.shape[1] != nk:
                raise ValueError("Text masks do not match rectangular attention lengths")
            # Build packing indices once, including CFG branches with different padding.
            qc, kc = qm.sum(1).to(torch.int32), km.sum(1).to(torch.int32)
            self._plans[key] = dict(
                qm=qm, km=km, qi=qm.flatten().nonzero().flatten(),
                ki=km.flatten().nonzero().flatten(),
                cq=F.pad(qc.cumsum(0, dtype=torch.int32), (1, 0)),
                ck=F.pad(kc.cumsum(0, dtype=torch.int32), (1, 0)),
                maxq=int(qc.max()), maxk=int(kc.max()),
                no_padding=bool(qm.all() and km.all()))
        plan = self._plans[key]
        if self.backend == "sdpa":
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=None if plan["no_padding"] else plan["km"][:, None, None, :],
                dropout_p=0.0, is_causal=False).transpose(1, 2)
            return out.masked_fill(~plan["qm"][:, :, None, None], 0).reshape(batch, nq, heads * dim)
        if q.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("FlashAttention requires fp16/bf16 tensors")
        if plan["no_padding"]:
            return self._flash(q, k, v, dropout_p=0.0, causal=False).reshape(batch, nq, heads * dim)
        packed = self._flash_varlen(
            q.reshape(-1, heads, dim).index_select(0, plan["qi"]),
            k.reshape(-1, heads, dim).index_select(0, plan["ki"]),
            v.reshape(-1, heads, dim).index_select(0, plan["ki"]),
            plan["cq"], plan["ck"], plan["maxq"], plan["maxk"],
            dropout_p=0.0, causal=False)
        out = q.new_zeros(batch * nq, heads, dim)
        out.index_copy_(0, plan["qi"], packed)
        return out.reshape(batch, nq, heads * dim)
