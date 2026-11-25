"""Mask/instruction-only region editing with an already merged final model."""
from pathlib import Path
import time

def composite_and_crop(raw,source,pixel_mask,geometry,frames=80):
    import numpy as np
    raw,source,pixel_mask=np.asarray(raw),np.asarray(source),np.asarray(pixel_mask,dtype=bool)
    if raw.shape!=source.shape or pixel_mask.shape!=source.shape[:3]:raise ValueError('Composite shapes differ')
    composite=np.where(pixel_mask[...,None],raw,source).astype(np.uint8)
    outside=~pixel_mask
    error=int(np.abs(composite.astype(np.int16)-source.astype(np.int16))[outside].max()) if outside.any() else 0
    # read_video adds RIGHT/BOTTOM padding; original generation's center offset
    # was already removed in the saved source MP4 and must not be applied again.
    result=composite[:frames,:geometry['resized_height'],:geometry['resized_width']]
    if result.shape[1:3]!=(geometry['source_height'],geometry['source_width']):
        raise ValueError('This Physics-IQ batch expects source shortest side480 without resizing')
    return result,error


def edit_one(pipe,request,policy,source_condition_path):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from hyvideo.commons import auto_offload_model
    from video_io import read_video
    from local_edit.masking import load_pixel_mask,pixel_to_latent_mask
    from local_edit.pipeline import prepare_conditioning,run_editor,synchronize
    device=pipe.execution_device
    args=SimpleNamespace(video=Path(request['source_video']),short_side=480,frames=81,fps=16.,start_seconds=0.)
    start=time.perf_counter()
    source_frames,geometry=read_video(args)
    source_read_s=time.perf_counter()-start
    mask=load_pixel_mask(Path(request['mask']),81,*source_frames.shape[1:3])
    latent_mask=pixel_to_latent_mask(mask,dilation=0)
    # Keep a prepared-canvas artifact; text/vision conditioning reads the
    # original source file, exactly as the training feature-cache path does.
    write_video(source_condition_path,source_frames,16)
    with torch.inference_mode():
        start=time.perf_counter()
        pixels=torch.from_numpy(source_frames.copy()).permute(3,0,1,2)[None].to(device,dtype=pipe.vae_dtype)/127.5-1
        with auto_offload_model(pipe.vae,device,enabled=True),torch.autocast('cuda',dtype=pipe.vae_dtype),pipe.vae.memory_efficient_context():
            source=pipe.vae.encode(pixels).latent_dist.mode()
            shift=getattr(pipe.vae.config,'shift_factor',None) or 0.
            source=(source-shift)*pipe.vae.config.scaling_factor
        del pixels
        source=source.to(dtype=pipe.target_dtype)
        if tuple(source.shape[2:])!=tuple(latent_mask.shape) or not torch.isfinite(source).all():
            raise ValueError('Source latent and requested mask do not match')
        synchronize(device);source_encode_s=time.perf_counter()-start
        start=time.perf_counter()
        cond,conditioning=prepare_conditioning(pipe,source,Path(request['source_video']),request['instruction'],'',policy['cfg'])
        synchronize(device);conditioning_s=time.perf_counter()-start
        # The training editing condition is [clean source, one-valued channel].
        if not torch.equal(cond[:,:source.shape[1]],source) or not torch.all(cond[:,source.shape[1]:]==1):
            raise ValueError('Native editing condition differs from the trained source+ones condition')
        latent,metrics=run_editor(pipe,source,latent_mask,cond,conditioning,mode=policy.get('mode','sparse'),steps=policy['steps'],
            cfg=policy['cfg'],seed=request['seed'],flow_shift=policy['flow_shift'],attention=policy['attention'],label=request['sample_id'])
        if metrics['background_latent_max_abs']!=0:raise RuntimeError('Background latents changed')
        start=time.perf_counter()
        with auto_offload_model(pipe.vae,device,enabled=True),torch.autocast('cuda',dtype=pipe.vae_dtype),pipe.vae.memory_efficient_context():
            shift=getattr(pipe.vae.config,'shift_factor',None) or 0.
            z=latent.to(device,dtype=pipe.vae_dtype)/pipe.vae.config.scaling_factor+shift
            decoded=pipe.vae.decode(z,return_dict=False)[0]
            raw=((decoded[0]/2+.5).clamp(0,1)*255).to(torch.uint8).permute(1,2,3,0).cpu().numpy()
        synchronize(device);decode_s=time.perf_counter()-start
    result,pixel_error=composite_and_crop(raw,source_frames,mask.cpu().numpy(),geometry)
    if pixel_error!=0:raise RuntimeError('Background pixels changed before encoding')
    metrics.update(source_read_s=source_read_s,source_encode_s=source_encode_s,conditioning_s=conditioning_s,
                   decode_s=decode_s,geometry=geometry,preencode_background_pixel_max_abs=pixel_error,
                   output_frames=len(result),output_wh=list(result.shape[2:0:-1]))
    del source,cond,conditioning,latent,z,decoded,raw
    return result,metrics


def write_video(path,frames,fps=16):
    import av
    import numpy as np
    from fractions import Fraction
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.stem+'.partial.mp4')
    h,w=frames.shape[1:3]
    with av.open(str(temp),'w') as container:
        stream=container.add_stream('libx264',rate=Fraction(fps))
        stream.width,stream.height,stream.pix_fmt=w,h,'yuv420p'
        stream.codec_context.thread_count=2
        stream.options={'crf':'18','preset':'fast'}
        for index,array in enumerate(frames):
            frame=av.VideoFrame.from_ndarray(np.ascontiguousarray(array),format='rgb24')
            frame.pts,frame.time_base=index,Fraction(1,fps)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    temp.replace(path)
