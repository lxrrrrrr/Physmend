"""Video decoding with the original 16 FPS physical-time sampling convention."""
import math
from pathlib import Path


def read_video(args):
    import av
    import numpy as np
    from PIL import Image
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or args.fps)
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.
        arrays, times, canvas = [], [], None
        end = args.start_seconds + (args.frames - 1) / args.fps
        for index, frame in enumerate(container.decode(stream)):
            t = float(frame.pts * frame.time_base) - origin if frame.pts is not None else index / source_fps
            if t < args.start_seconds - 1 / source_fps:
                continue
            if canvas is None:
                scale = args.short_side / min(frame.height, frame.width)
                rh, rw = round(frame.height * scale), round(frame.width * scale)
                ph, pw = math.ceil(rh / 32) * 32, math.ceil(rw / 32) * 32
                canvas = dict(source_height=frame.height, source_width=frame.width,
                              resized_height=rh, resized_width=rw, padded_height=ph, padded_width=pw)
            image = frame.to_image().resize((canvas['resized_width'], canvas['resized_height']), Image.Resampling.LANCZOS)
            array = np.asarray(image)
            array = np.pad(array, ((0, canvas['padded_height'] - array.shape[0]),
                                   (0, canvas['padded_width'] - array.shape[1]), (0, 0)), mode='edge')
            arrays.append(array)
            times.append(t)
            if t >= end + 1 / source_fps:
                break
    if not arrays:
        raise ValueError('No video frames found at the requested start time')
    requested = args.start_seconds + np.arange(args.frames) / args.fps
    pos = np.searchsorted(times, requested).clip(0, len(times) - 1)
    previous = np.maximum(pos - 1, 0)
    choose_prev = np.abs(np.asarray(times)[previous] - requested) < np.abs(np.asarray(times)[pos] - requested)
    pos = np.where(choose_prev, previous, pos)
    canvas.update(source_fps=source_fps, sampled_frames=args.frames, fps=args.fps,
                  start_seconds=args.start_seconds,
                  held_last_frames=int((requested > times[-1] + .5 / source_fps).sum()))
    return np.stack([arrays[i] for i in pos]), canvas


def instruction_text(path=None, text=None):
    """Accept plain instruction text or one existing <instruct> section."""
    import re
    value = Path(path).read_text(encoding='utf-8') if path else text
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Provide a nonempty instruction')
    sections = re.findall(r'<instruct>\s*(.*?)\s*</instruct>', value, flags=re.S)
    if '<instruct' in value or '</instruct>' in value:
        if len(sections) != 1:
            raise ValueError('Expected exactly one <instruct> section')
        value = sections[0]
    if not value.strip() or re.search(r'</?(?:instruct|reasoning)>', value):
        raise ValueError('Invalid instruction; supply plain text or one complete <instruct> block')
    return value.strip()
