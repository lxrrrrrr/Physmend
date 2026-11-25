"""CPU correctness tests with the actual small OmniWeaving Transformer."""
import copy
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(1)
from local_edit.attention import RectangularAttention
from local_edit.engine import LocalEditState, apply_rope
from local_edit.masking import pixel_to_latent_mask, clamp_background, load_pixel_mask
from local_edit.pipeline import run_editor
import hyvideo.models.transformers.hunyuanvideo_1_5_transformer as native
from hyvideo.models.transformers.modules.posemb_layers import apply_rotary_emb

native.get_parallel_state = lambda: types.SimpleNamespace(sp_enabled=False)


def dense_attention(q, k, v, img_q_len, img_kv_len, text_mask, **kwargs):
    return RectangularAttention("sdpa")(
        torch.cat(q, 1), torch.cat(k, 1), torch.cat(v, 1), text_mask,
        img_q_len, img_kv_len)

native.parallel_attention = dense_attention


def model():
    torch.manual_seed(123)
    net = native.HunyuanVideo_1_5_DiffusionTransformer(
        patch_size=[1,1,1], in_channels=4, out_channels=4, hidden_size=32,
        heads_num=4, mlp_width_ratio=2, mm_double_blocks_depth=3,
        mm_single_blocks_depth=0, rope_dim_list=[2,2,4],
        text_projection="linear", text_states_dim=12, text_states_dim_2=None,
        glyph_byT5_v2=False, vision_projection="linear", vision_states_dim=6,
        use_cond_type_embedding=False, deepstack=None)
    # The real output layer may initialize to zeros. Randomize so comparisons
    # test actual signal propagation rather than a trivially zero prediction.
    for parameter in net.parameters():
        if parameter.ndim > 1:
            torch.nn.init.normal_(parameter, std=0.08)
    return net.eval()


def inputs(batch=2):
    torch.manual_seed(456)
    x = torch.randn(batch, 9, 3, 4, 4)
    kw = dict(text_states=torch.randn(batch, 5, 12), text_states_2=None,
              encoder_attention_mask=torch.tensor([[1,1,1,0,0], [1,1,1,1,0]])[:batch],
              vision_states=torch.randn(batch, 2, 6), mask_type="editing",
              return_dict=False)
    return x, kw


def prefill(net, state, x, kwargs):
    state.phase = "prefill"
    with torch.inference_mode():
        net(x, torch.zeros(x.shape[0]), local_edit_state=state, **kwargs)
    state.phase = "denoise"


class LocalEditTests(unittest.TestCase):
    def setUp(self):
        self.net = model()
        self.x, self.kw = inputs()
        self.mask = torch.zeros(3, 4, 4, dtype=torch.bool)
        self.mask[0, 0, 0] = True
        self.mask[1, 1:3, 2] = True
        self.mask[2, 3, 1] = True

    def test_sparse_matches_dense_static_and_skips_background_mlp(self):
        sparse = LocalEditState(self.mask, "sparse", "sdpa")
        prefill(self.net, sparse, self.x, self.kw)
        dense = LocalEditState(self.mask, "dense-static", "sdpa")
        dense.layers = sparse.layers
        cached_before = [layer.key.clone() for layer in sparse.layers]
        lengths = []
        handle = self.net.double_blocks[0].img_mlp.register_forward_pre_hook(
            lambda module, args: lengths.append(args[0].shape[1]))
        with torch.inference_mode():
            for t in (800., 200.):
                a = self.net(self.x, torch.full((2,), t), local_edit_state=sparse, **self.kw)[0]
                b = self.net(self.x, torch.full((2,), t), local_edit_state=dense, **self.kw)[0]
                torch.testing.assert_close(a[:, :, self.mask], b[:, :, self.mask], atol=2e-6, rtol=2e-5)
                self.assertEqual(float(a[:, :, ~self.mask].abs().max()), 0.)
        handle.remove()
        self.assertEqual(lengths, [4,48,4,48])
        for before, layer in zip(cached_before, sparse.layers):
            torch.testing.assert_close(before, layer.key, atol=0, rtol=0)

    def test_full_custom_path_matches_original_model(self):
        state = LocalEditState(self.mask, "full", "sdpa")
        with torch.inference_mode():
            native_output = self.net(self.x, torch.tensor([250.,250.]), **self.kw)[0]
            actual = self.net(self.x, torch.tensor([250.,250.]), local_edit_state=state, **self.kw)[0]
        torch.testing.assert_close(actual, native_output, atol=2e-6, rtol=2e-5)

    def test_all_edit_matches_full_and_stores_no_background(self):
        state = LocalEditState(torch.ones_like(self.mask), "sparse", "sdpa")
        prefill(self.net, state, self.x, self.kw)
        self.assertEqual(state.cache_bytes, 0)
        with torch.inference_mode():
            native_output = self.net(self.x, torch.tensor([500.,500.]), **self.kw)[0]
            actual = self.net(self.x, torch.tensor([500.,500.]), local_edit_state=state, **self.kw)[0]
        torch.testing.assert_close(actual, native_output, atol=2e-6, rtol=2e-5)

    def test_absolute_rope_is_gathered_without_renumbering(self):
        cos, sin = self.net.get_rotary_pos_embed((3,4,4))
        q, k = torch.randn(2,48,4,8), torch.randn(2,48,4,8)
        a, b = apply_rope(q, k, (cos,sin))
        ra, rb = apply_rotary_emb(q,k,(cos,sin))
        torch.testing.assert_close(a,ra)
        torch.testing.assert_close(b,rb)
        ids = self.mask.flatten().nonzero().flatten()
        local_a, local_b = apply_rope(q[:,ids],k[:,ids],(cos[ids],sin[ids]))
        torch.testing.assert_close(local_a,a[:,ids])
        torch.testing.assert_close(local_b,b[:,ids])

    def test_byt5_deepstack_and_native_text_refiner(self):
        import hyvideo.models.transformers.modules.token_refiner as refiner
        refiner.attention = lambda q,k,v,attn_mask=None,**kw: RectangularAttention("sdpa")(
            q,k,v,attn_mask if attn_mask is not None else torch.ones(q.shape[:2],dtype=torch.bool),0,0)
        torch.manual_seed(78)
        net = native.HunyuanVideo_1_5_DiffusionTransformer(
            patch_size=[1,1,1],in_channels=4,out_channels=4,hidden_size=32,heads_num=4,
            mlp_width_ratio=2,mm_double_blocks_depth=3,mm_single_blocks_depth=0,
            rope_dim_list=[2,2,4],text_projection="single_refiner",text_states_dim=12,
            text_states_dim_2=None,glyph_byT5_v2=True,vision_projection="linear",
            vision_states_dim=6,use_cond_type_embedding=True,deepstack=[1,2]).eval()
        torch.nn.init.normal_(net.final_layer.linear.weight,std=0.08)
        kwargs=dict(self.kw)
        kwargs.update(extra_kwargs=dict(byt5_text_states=torch.randn(2,3,1472),
                     byt5_text_mask=torch.tensor([[1,1,0],[1,0,0]])),
                     all_stack_text_states=torch.randn(2,2,5,12))
        state=LocalEditState(self.mask,"full","sdpa")
        with torch.inference_mode():
            a=net(self.x,torch.tensor([500.,500.]),**kwargs)[0]
            b=net(self.x,torch.tensor([500.,500.]),local_edit_state=state,**kwargs)[0]
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        sparse=LocalEditState(self.mask,"sparse","sdpa")
        prefill(net,sparse,self.x,kwargs)
        dense=LocalEditState(self.mask,"dense-static","sdpa")
        dense.layers=sparse.layers
        with torch.inference_mode():
            a=net(self.x,torch.tensor([800.,800.]),local_edit_state=sparse,**kwargs)[0]
            b=net(self.x,torch.tensor([800.,800.]),local_edit_state=dense,**kwargs)[0]
        torch.testing.assert_close(a[:,:,self.mask],b[:,:,self.mask],atol=2e-6,rtol=2e-5)

    def test_pixel_mask_causal_temporal_mapping_and_dilation(self):
        pixels = torch.zeros(9,32,48,dtype=torch.bool)
        pixels[4,16,32] = True
        mask = pixel_to_latent_mask(pixels)
        self.assertEqual(tuple(mask.shape),(3,2,3))
        self.assertEqual(mask.nonzero().tolist(),[[1,1,2]])
        pixels[5,0,0] = True
        mask = pixel_to_latent_mask(pixels)
        self.assertEqual(mask.nonzero().tolist(),[[1,1,2],[2,0,0]])
        self.assertGreater(pixel_to_latent_mask(pixels,dilation=1).sum(),mask.sum())
        with self.assertRaises(ValueError):
            LocalEditState(torch.zeros(3,2,3,dtype=torch.bool))

    def test_background_clamp_is_exact(self):
        src = torch.randn(1,4,3,4,4)
        z = clamp_background(torch.randn_like(src),src,self.mask)
        torch.testing.assert_close(z[:,:,~self.mask],src[:,:,~self.mask],atol=0,rtol=0)

    def test_flow_loop_dense_static_sparse_equivalence(self):
        from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
        source=self.x[:1,:4].clone()
        cond=self.x[:1,4:].clone()
        kwargs={k:v for k,v in self.kw.items()}
        pipe=types.SimpleNamespace(
            execution_device="cpu",target_dtype=torch.float32,transformer=self.net,
            _create_scheduler=lambda shift:FlowMatchDiscreteScheduler(shift=shift,reverse=True,solver="euler"),use_meanflow=False)
        results=[]
        for mode in ("dense-static","sparse"):
            result,metrics=run_editor(
                pipe,source,self.mask,cond,kwargs,mode=mode,steps=3,cfg=4,
                seed=7,flow_shift=7,attention="sdpa",log=lambda *a,**kw:None)
            self.assertEqual(metrics["background_latent_max_abs"],0.)
            results.append(result)
        torch.testing.assert_close(results[0],results[1],atol=2e-6,rtol=2e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
