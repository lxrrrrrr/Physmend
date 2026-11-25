"""Input contracts and real video/mask I/O, without full model loading."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from inference import parse, generation_kwargs
from local_edit.trained_inference import composite_and_crop, write_video
from local_edit.masking import load_pixel_mask
from video_io import instruction_text, read_video


class InferenceTests(unittest.TestCase):
    def test_model_path_cli_overrides_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict('os.environ', {'PHYSMEND_MODEL_PATH': str(root / 'env')}):
                for task in ('t2v', 'i2v', 'edit'):
                    self.assertEqual(parse(task, []).model_path, root / 'env')
                    self.assertEqual(parse(task, ['--model-path', str(root / 'cli')]).model_path, root / 'cli')

    def test_instruction_extracts_only_instruct(self):
        value = '<reasoning>do not send this</reasoning>\n<instruct>Repair liquid flow.</instruct>'
        self.assertEqual(instruction_text(text=value), 'Repair liquid flow.')
        for invalid in ('', '<instruct>incomplete', '<instruct>a</instruct><instruct>b</instruct>'):
            with self.assertRaises(ValueError):
                instruction_text(text=invalid)

    def test_generation_routes_native_i2v_input(self):
        args = parse('i2v', ['--prompt', 'Red liquid flowing.'])
        kwargs = generation_kwargs(args)
        self.assertEqual(kwargs['task_type'], 'i2v')
        self.assertEqual(kwargs['ref_image_paths'], [str(args.image.resolve())])
        self.assertFalse(kwargs['think'])
        self.assertFalse(kwargs['only_give_text'])
        self.assertEqual(kwargs['aspect_ratio'], '864:480')
        self.assertTrue(generation_kwargs(parse('t2v', ['--prompt', 'Flow.']))['only_give_text'])

    def test_invalid_output_geometry_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (['--frames', '80'], ['--width', '853'], ['--height', '0']):
                with self.assertRaises(SystemExit):
                    parse('t2v', argv)

    def test_video_sampling_mask_composite_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A genuine 80-frame/16fps file with right padding required.
            original = np.full((80, 32, 50, 3), 55, dtype=np.uint8)
            original[:, :, :12] = 190
            video = root / 'input.mp4'
            write_video(video, original, 16)
            source, geometry = read_video(SimpleNamespace(video=video, short_side=32, frames=81, fps=16., start_seconds=0.))
            self.assertEqual(source.shape, (81, 32, 64, 3))
            np.testing.assert_array_equal(source[-1], source[-2])
            self.assertEqual(geometry['held_last_frames'], 1)
            mask = np.zeros((32, 64), dtype=bool)
            mask[:, 14:22] = True
            np.save(root / 'mask.npy', mask)
            loaded = load_pixel_mask(root / 'mask.npy', 81, 32, 64)
            raw = np.zeros_like(source)
            composite, error = composite_and_crop(raw, source, loaded.numpy(), geometry)
            self.assertEqual(composite.shape, (80, 32, 50, 3))
            self.assertEqual(error, 0)
            outside = ~loaded.numpy()[:80, :, :50]
            np.testing.assert_array_equal(composite[outside], source[:80, :, :50][outside])
            write_video(root / 'output.mp4', composite, 16)
            self.assertTrue((root / 'output.mp4').is_file())


if __name__ == '__main__':
    unittest.main()
