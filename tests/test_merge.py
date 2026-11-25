"""Merge arithmetic and standalone loader routing without a large model."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file, load_file
from tools.merge_lora import merge
from model_assets import load_pipeline


class MergeTests(unittest.TestCase):
    def test_fusion_matches_linear_lora_and_preserves_untouched_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / 'adapter'
            adapter.mkdir()
            config = dict(peft_type='LORA', bias='none', r=2, lora_alpha=4)
            (adapter / 'adapter_config.json').write_text(json.dumps(config))
            torch.manual_seed(6)
            base = torch.randn(5, 7)
            bias = torch.randn(5)
            a, b = torch.randn(2, 7).bfloat16(), torch.randn(5, 2).bfloat16()
            save_file({'layer.weight': base, 'layer.bias': bias}, str(root / 'base.safetensors'))
            save_file({'layer.lora_A.weight': a, 'layer.lora_B.weight': b}, str(adapter / 'pytorch_lora_weights.safetensors'))
            output = root / 'final/transformer/diffusion_pytorch_model.safetensors'
            with contextlib.redirect_stdout(io.StringIO()):
                report = merge(root / 'base.safetensors', adapter, output)
            saved = load_file(str(output))
            torch.testing.assert_close(saved['layer.weight'], base + 2 * (b.float() @ a.float()), rtol=0, atol=0)
            torch.testing.assert_close(saved['layer.bias'], bias, rtol=0, atol=0)
            x = torch.randn(3, 7)
            fused = x @ saved['layer.weight'].T + bias
            separate = x @ base.T + 2 * ((x @ a.float().T) @ b.float().T) + bias
            torch.testing.assert_close(fused, separate, rtol=2e-6, atol=2e-6)
            self.assertEqual(report['merged_modules'], 1)
            self.assertTrue(report['every_saved_tensor_readback_sha256_verified'])
            self.assertTrue((root / 'final/MERGE_COMPLETE.json').is_file())

    def test_loader_passes_external_model_path_without_adapter(self):
        pipeline = MagicMock()
        with patch('hyvideo.pipelines.hunyuan_video_pipeline.HunyuanVideo_1_5_Pipeline.create_pipeline', return_value=pipeline) as create:
            result = load_pipeline('/tmp/external-model', torch.device('cpu'))
        self.assertIs(result, pipeline)
        self.assertEqual(create.call_args.args[0], '/tmp/external-model')
        pipeline.transformer.add_adapter.assert_not_called()
        pipeline.transformer.load_lora_adapter.assert_not_called()


if __name__ == '__main__':
    unittest.main()
