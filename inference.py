"""Single-GPU T2V, I2V and editing from supplied mask + instruction."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

PACKAGE = Path(__file__).resolve().parent
EXAMPLE = Path(os.environ.get('PHYSMEND_EXAMPLE_PATH', str(PACKAGE / 'examples/liquid_overflow'))).expanduser().resolve()
sys.dont_write_bytecode = True
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def parse(task, argv=None):
    p = argparse.ArgumentParser(description={'t2v': '文生视频', 'i2v': '图生视频', 'edit': '加载 mask 和 instruct 做区域编辑'}[task])
    p.add_argument('--model-path', type=Path,
                   default=Path(os.environ.get('PHYSMEND_MODEL_PATH', str(PACKAGE.parent / 'physmend_model'))),
                   help='Merged final model directory; alternatively set PHYSMEND_MODEL_PATH')
    p.add_argument('--output', type=Path, default=PACKAGE / 'outputs' / (task + '.mp4'))
    p.add_argument('--steps', type=int, default=30 if task == 'edit' else 50)
    p.add_argument('--cfg', type=float, default=6.)
    p.add_argument('--flow-shift', type=float, default=7.)
    p.add_argument('--seed', type=int, default=2768594355 if task == 'edit' else 42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--check', action='store_true', help='Validate assets and inputs without loading full models or requiring CUDA')
    p.add_argument('--verify-weights', action='store_true', help='Also hash every packaged model asset')
    if task == 'edit':
        p.add_argument('--video', type=Path, default=EXAMPLE / 'source.mp4')
        p.add_argument('--mask', type=Path, default=EXAMPLE / 'mask.npy')
        group = p.add_mutually_exclusive_group()
        group.add_argument('--instruct', type=Path, help='Plain TXT or TXT containing one <instruct> section')
        group.add_argument('--instruction', help='Instruction string supplied directly')
        p.add_argument('--attention', choices=['auto', 'flash', 'sdpa'], default='flash')
    else:
        group = p.add_mutually_exclusive_group()
        group.add_argument('--prompt', help='Generation prompt')
        group.add_argument('--prompt-file', type=Path)
        p.add_argument('--width', type=int, default=854)
        p.add_argument('--height', type=int, default=480)
        p.add_argument('--frames', type=int, default=81, help='Model frames, 4*n+1; first 4*n frames are written')
        p.add_argument('--fps', type=int, default=16)
        if task == 'i2v':
            p.add_argument('--image', type=Path, default=EXAMPLE / 'first_frame.jpg')
    args = p.parse_args(argv)
    args.model_path = args.model_path.expanduser().resolve()
    args.task = task
    if args.steps < 1 or args.cfg < 1 or args.flow_shift <= 0 or not 0 <= args.seed < 2**32:
        p.error('steps>=1, cfg>=1, flow-shift>0 and 0<=seed<2**32 required')
    if task != 'edit' and (args.frames < 5 or (args.frames - 1) % 4 or min(args.width, args.height, args.fps) <= 0 or args.width % 2 or args.height % 2):
        p.error('Use positive even width/height, positive fps and frames=4*n+1>=5')
    if task == 'edit':
        from video_io import instruction_text
        try:
            args.instruction = instruction_text(
                path=args.instruct or (EXAMPLE / 'instruct.txt' if args.instruction is None else None),
                text=args.instruction)
        except ValueError as exc:
            p.error(str(exc))
    else:
        args.prompt = args.prompt_file.read_text().strip() if args.prompt_file else args.prompt or (EXAMPLE / 'prompt.txt').read_text().strip()
        if not args.prompt.strip():
            p.error('Prompt must be nonempty')
    return args


def generation_kwargs(args):
    width = math.ceil(args.width / 32) * 32
    height = math.ceil(args.height / 32) * 32
    kw = dict(prompt=args.prompt, task_type=args.task, aspect_ratio=f'{width}:{height}',
              video_length=args.frames, num_inference_steps=args.steps, guidance_scale=args.cfg,
              flow_shift=args.flow_shift, negative_prompt='', seed=args.seed, think=False,
              only_give_text=args.task == 't2v', deepstack=[8, 16, 24], setclip=True,
              output_type='pt', enable_vae_tile_parallelism=False)
    if args.task == 'i2v':
        # OmniWeaving's I2V branch reads ref_image_paths, not reference_image.
        kw['ref_image_paths'] = [str(args.image.resolve())]
    return kw


def check_inputs(args):
    from model_assets import validate_assets
    result = dict(task=args.task, **validate_assets(args.model_path, args.verify_weights))
    if args.task == 'edit':
        from video_io import read_video
        from local_edit.masking import load_pixel_mask, pixel_to_latent_mask
        pixels, geometry = read_video(SimpleNamespace(video=args.video, short_side=480, frames=81, fps=16., start_seconds=0.))
        if (geometry['source_height'], geometry['source_width']) != (geometry['resized_height'], geometry['resized_width']):
            raise ValueError('Editing expects a source video with shortest side480; resize source and mask together first')
        mask = load_pixel_mask(args.mask, 81, *pixels.shape[1:3])
        latent = pixel_to_latent_mask(mask, dilation=0)
        result.update(source=str(args.video.resolve()), mask=str(args.mask.resolve()),
                      instruction=args.instruction, geometry=geometry, mask_shape=list(mask.shape),
                      latent_mask_shape=list(latent.shape), edit_fraction=float(latent.float().mean()))
    else:
        if args.task == 'i2v':
            from PIL import Image
            with Image.open(args.image) as image:
                image.verify()
            result['image'] = str(args.image.resolve())
        result.update(prompt=args.prompt, model_frames=args.frames, output_frames=args.frames - 1,
                      width=args.width, height=args.height, fps=args.fps)
    return result


def run(task, argv=None):
    args = parse(task, argv)
    result = check_inputs(args)
    print('PHYSMEND_CHECK ' + json.dumps(result, ensure_ascii=False), flush=True)
    if args.check:
        return result
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for inference; use --check for CPU validation')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('These entry points use one process and one GPU')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('--device must be a CUDA device')
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protected = []
    if task == 'edit':
        protected = [args.video, args.mask] + ([args.instruct] if args.instruct else [])
    elif task == 'i2v':
        protected = [args.image]
    if args.output.is_relative_to(args.model_path) or args.output.is_relative_to(EXAMPLE) or args.output in [p.resolve() for p in protected]:
        raise ValueError('Output must not overwrite model or example/input assets')
    rendezvous = args.output.parent / ('.distributed-' + str(os.getpid()))
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group('nccl', init_method='file://' + str(rendezvous), rank=0, world_size=1, timeout=timedelta(hours=2))
    from hyvideo.commons.parallel_states import initialize_parallel_state
    initialize_parallel_state(sp=1)
    try:
        from model_assets import load_pipeline
        from local_edit.trained_inference import edit_one, write_video
        started = time.perf_counter()
        pipe = load_pipeline(args.model_path, device)
        if task == 'edit':
            request = dict(source_video=str(args.video.resolve()), mask=str(args.mask.resolve()),
                           instruction=args.instruction, seed=args.seed, sample_id=args.video.stem)
            policy = dict(steps=args.steps, cfg=args.cfg, flow_shift=args.flow_shift,
                          attention=args.attention, mode='sparse')
            prepared = args.output.with_suffix('.prepared_source.mp4')
            frames, metrics = edit_one(pipe, request, policy, prepared)
            write_video(args.output, frames, 16)
        else:
            import types
            pw, ph = math.ceil(args.width / 32) * 32, math.ceil(args.height / 32) * 32
            def resolution(_self, _origin_size, _target_size):
                return ph, pw
            pipe.get_closest_resolution_given_original_size = types.MethodType(resolution, pipe)
            output = pipe(**generation_kwargs(args))
            frames = (output.videos[0].permute(1, 2, 3, 0).float() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            top, left = (ph - args.height) // 2, (pw - args.width) // 2
            frames = frames[:args.frames - 1, top:top + args.height, left:left + args.width]
            write_video(args.output, frames, args.fps)
            metrics = dict(output_frames=len(frames), width=args.width, height=args.height,
                           merged_final_model=True)
        result.update(output=str(args.output), seconds=time.perf_counter() - started, metrics=metrics,
                      steps=args.steps, cfg=args.cfg, flow_shift=args.flow_shift, seed=args.seed)
        args.output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print('PHYSMEND_DONE ' + str(args.output), flush=True)
        return result
    finally:
        if owns_group:
            dist.destroy_process_group()
            rendezvous.unlink(missing_ok=True)
