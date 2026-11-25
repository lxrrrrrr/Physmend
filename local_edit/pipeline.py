"""Native OmniWeaving conditioning plus timed local flow-matching sampling."""
import time

import torch
from .engine import LocalEditState
from .masking import clamp_background


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def prepare_conditioning(pipe, source, video_path, prompt, negative_prompt, cfg):
    from hyvideo.commons import auto_offload_model
    device = torch.device(pipe.execution_device)
    pipe._guidance_scale = cfg
    pipe._guidance_rescale = 0.0
    pipe._clip_skip = None
    frames = (source.shape[2]-1) * pipe.vae_temporal_compression_ratio + 1
    task = pipe.get_task_specific_input(
        prompt=prompt, task_type="editing", condition_videos=[str(video_path)],
        condition_video_latents=source, video_length=frames,
        aspect_ratio=f"{source.shape[-1]*16}:{source.shape[-2]*16}",
        target_resolution="480p", device=device)
    nframes = 8 if frames >= 96 else 6 if frames >= 72 else 4
    with auto_offload_model(pipe.text_encoder, device, enabled=pipe.enable_offloading):
        positive, negative, pos_mask, neg_mask, stack, neg_stack = pipe.encode_prompt(
            prompt, device, 1, cfg > 1, negative_prompt, clip_skip=None,
            data_type="video", task_type="editing", video_sample_frames=task["video_sample_frames"],
            nframes=nframes, semantic_images=task["semantic_images"],
            only_give_text=False, deepstack=[8,16,24], setclip=True)
    if cfg > 1:
        positive = torch.cat((negative, positive))
        pos_mask = torch.cat((neg_mask, pos_mask))
        if stack is not None:
            stack = torch.cat((neg_stack, stack))
    with auto_offload_model(pipe.byt5_model, device, enabled=pipe.enable_offloading):
        extra = pipe._prepare_byt5_embeddings(prompt, device)
    for key in extra:
        if extra[key].is_floating_point():
            extra[key] = extra[key].to(pipe.target_dtype)
    with auto_offload_model(pipe.vision_encoder, device, enabled=pipe.enable_offloading):
        vision = pipe._prepare_vision_states(
            task["reference_image"], task["target_resolution"], source, device,
            reference_image2=task["reference_image2"])
    cond = pipe._prepare_cond_latents("editing", source, source, task["multitask_mask"])
    kwargs = dict(
        text_states=positive.to(pipe.target_dtype), text_states_2=None,
        encoder_attention_mask=pos_mask, vision_states=vision,
        mask_type="editing", return_dict=False,
        all_stack_text_states=stack.to(pipe.target_dtype) if stack is not None else None,
        extra_kwargs=extra)
    return cond, kwargs


@torch.inference_mode()
def run_editor(pipe, source, edit_mask, cond, conditioning, *,
               mode, steps, cfg, seed, flow_shift, attention,
               log=print, label=None):
    source = source.to(device=pipe.execution_device, dtype=pipe.target_dtype)
    edit_mask = edit_mask.to(source.device)
    state = LocalEditState(edit_mask, mode, attention)
    device = source.device
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = clamp_background(torch.randn(source.shape, device=device, dtype=source.dtype,
                                          generator=generator), source, edit_mask)
    scheduler = pipe._create_scheduler(flow_shift)
    # Match the native retrieve_timesteps path (including optional n_tokens).
    import inspect
    scheduler_kwargs = {}
    if "n_tokens" in inspect.signature(scheduler.set_timesteps).parameters:
        scheduler_kwargs["n_tokens"] = edit_mask.numel()
    scheduler.set_timesteps(steps, device=device, **scheduler_kwargs)
    if getattr(scheduler, "order", 1) != 1:
        raise ValueError("Local editor currently requires a first-order flow scheduler")
    if getattr(pipe, "use_meanflow", False):
        raise ValueError("Meanflow checkpoints are not supported by this local-edit prototype")
    def forward(z, timestep):
        x = torch.cat((z, cond), 1)
        if cfg > 1:
            x = torch.cat((x, x), 0)
        # A static context is constructed at t=0. scale_model_input is identity
        # for the native flow scheduler; still use its public interface.
        x = scheduler.scale_model_input(x, timestep)
        t = torch.as_tensor(timestep, device=device).reshape(1).expand(x.shape[0])
        with torch.autocast(device_type=device.type, dtype=pipe.target_dtype,
                            enabled=device.type == "cuda" and pipe.target_dtype != torch.float32):
            return pipe.transformer(x, t, local_edit_state=state, **conditioning)[0]
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    prefill_s = 0.0
    if mode != "full":
        state.phase = "prefill"
        ts = time.perf_counter()
        forward(source, torch.zeros((), device=device))
        synchronize(device)
        prefill_s = time.perf_counter()-ts
        state.phase = "denoise"
        log(f"LOCAL_EDIT_PREFILL label={label} mode={mode} seconds={prefill_s:.6f} "
            f"static_kv_GiB={state.cache_bytes/2**30:.6f} layers={len(state.layers)} "
            f"backend={state.attention.backend}", flush=True)
    else:
        log(f"LOCAL_EDIT_PREFILL label={label} mode=full seconds=0 static_kv_GiB=0", flush=True)
    step_times = []
    denoise_start = time.perf_counter()
    for index, t in enumerate(scheduler.timesteps):
        ts = time.perf_counter()
        prediction = forward(latents, t)
        if cfg > 1:
            uncond, positive = prediction.chunk(2)
            prediction = uncond + cfg * (positive-uncond)
        latents = scheduler.step(prediction, t, latents, return_dict=False)[0]
        latents = clamp_background(latents, source, edit_mask)
        synchronize(device)
        elapsed = time.perf_counter()-ts
        step_times.append(elapsed)
        log(f"LOCAL_EDIT_STEP label={label} mode={mode} step={index+1}/{len(scheduler.timesteps)} "
            f"t={float(t):.4f} step_s={elapsed:.6f} "
            f"allocated_GiB={torch.cuda.memory_allocated(device)/2**30 if device.type=='cuda' else 0:.4f}",
            flush=True)
    synchronize(device)
    denoise_s = time.perf_counter()-denoise_start
    finite = bool(torch.isfinite(latents).all())
    if not finite:
        raise FloatingPointError(f"Nonfinite latents in {mode}")
    outside = ~edit_mask[None, None].expand_as(latents)
    background_max = float((latents-source).abs()[outside].max()) if outside.any() else 0.0
    metrics = dict(
        **state.statistics, steps=len(step_times), cfg=cfg, seed=seed,
        prefill_s=prefill_s, denoise_s=denoise_s,
        sampling_s=time.perf_counter()-start, step_seconds=step_times,
        mean_step_s=sum(step_times)/len(step_times),
        peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else None,
        peak_reserved_GiB=torch.cuda.max_memory_reserved(device)/2**30 if device.type=="cuda" else None,
        background_latent_max_abs=background_max, finite=finite)
    result = latents.detach().cpu()
    state.layers.clear()
    return result, metrics
