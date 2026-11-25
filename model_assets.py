"""Validate and load a standalone final model supplied by directory path."""
import hashlib
import json
from pathlib import Path
import struct


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def header(path):
    path = Path(path)
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if not 0 < n < 32 * 1024**2:
            raise ValueError('Invalid safetensors header: ' + str(path))
        value = json.loads(f.read(n))
    value.pop('__metadata__', None)
    size = path.stat().st_size - 8 - n
    if not value or max(v['data_offsets'][1] for v in value.values()) != size:
        raise ValueError('Safetensors payload length differs: ' + str(path))
    return value


def validate_assets(model_path, verify_weights=False):
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Model directory missing: {root}. Use --model-path /path/to/physmend_model')
    manifest = json.loads((root / 'ASSETS.json').read_text())
    merge = json.loads((root / 'MERGE_COMPLETE.json').read_text())
    if manifest.get('format') != 'physmend-merged-v1' or merge.get('format') != 'physmend-merged-v1':
        raise ValueError('Expected a PhysMend final merged model, with its ASSETS.json and MERGE_COMPLETE.json')
    total = 0
    for entry in manifest['files']:
        p = root / entry['path']
        if not p.resolve().is_relative_to(root) or p.is_symlink() or not p.is_file() or p.stat().st_size != entry['bytes']:
            raise ValueError('Model asset missing or changed: ' + str(p))
        if verify_weights and digest(p) != entry['sha256']:
            raise ValueError('Model checksum mismatch: ' + str(p))
        total += entry['bytes']
    relative = 'transformer/diffusion_pytorch_model.safetensors'
    recorded = next((entry for entry in manifest['files'] if entry['path'] == relative), None)
    if recorded is None or recorded['sha256'] != merge['merged_sha256']:
        raise ValueError('Manifest and final merged transformer identity differ')
    tensors = header(root / relative)
    if len(tensors) != merge['tensors'] or any('lora_' in key for key in tensors):
        raise ValueError('Expected merged transformer tensors without separate LoRA parameters')
    for index in root.glob('**/*.safetensors.index.json'):
        data = json.loads(index.read_text())
        for shard in set(data['weight_map'].values()):
            header(index.parent / shard)
    return dict(model='PhysMend final merged generation8000 + edit10000',
                model_path=str(root), model_files=len(manifest['files']), model_bytes=total,
                transformer_tensors=len(tensors), merged_lora_modules=merge['merged_modules'],
                merged_transformer_sha256=merge['merged_sha256'],
                separate_adapter_required=False, all_model_sha256_checked=verify_weights)


def load_pipeline(model_path, device):
    """Load fused final weights directly. No adapter injection at inference."""
    import torch
    from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
    root = Path(model_path).expanduser().resolve()
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        str(root), device=torch.device('cpu'), transformer_init_device=torch.device('cpu'),
        transformer_dtype=torch.bfloat16, pipeline_config='omniweaving')
    pipe.transformer.requires_grad_(False).eval().to(device)
    for name in ['vae', 'text_encoder', 'text_encoder_2', 'vision_encoder', 'byt5_model']:
        module = getattr(pipe, name, None)
        if module is not None:
            module.requires_grad_(False).eval()
    pipe.enable_offloading = True
    pipe.execution_device = device
    pipe.cache_helper = None
    return pipe
