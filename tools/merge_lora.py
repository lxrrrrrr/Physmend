"""Stream a bias-free ordinary linear LoRA into an FP32 safetensors file.

No complete model is instantiated. Every saved tensor is read back and checked.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import struct
import time


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def merge(base, adapter, output):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    base, adapter, output = Path(base), Path(adapter), Path(output)
    config = json.loads((adapter / 'adapter_config.json').read_text())
    if (config['peft_type'] != 'LORA' or config['bias'] != 'none'
        or any(config.get(k) for k in ('fan_in_fan_out', 'use_dora', 'use_rslora',
                                      'use_qalora', 'rank_pattern', 'alpha_pattern', 'modules_to_save', 'lora_bias'))):
        raise ValueError('Only ordinary bias-free linear LoRA with uniform rank/alpha is supported')
    rank, scale = config['r'], config['lora_alpha'] / config['r']
    weights = adapter / 'pytorch_lora_weights.safetensors'
    with base.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        original_header = json.loads(stream.read(length))
    tensors = {k: v for k, v in original_header.items() if k != '__metadata__'}
    ordered = sorted(tensors, key=lambda k: tensors[k]['data_offsets'][0])
    if any(value['dtype'] != 'F32' for value in tensors.values()):
        raise ValueError('Expected an FP32 base; keep merge arithmetic and storage in FP32')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial.safetensors')
    if temporary.exists():
        raise FileExistsError(temporary)
    started = time.perf_counter()
    metadata = dict(original_header.get('__metadata__', {}))
    metadata.update(format='pt', merged_edit_checkpoint='10000', merge_formula='W + (alpha/r) * B@A, FP32', lora_scale=str(scale))
    result_header = dict(tensors, __metadata__=metadata)
    raw = json.dumps(result_header, separators=(',', ':')).encode()
    raw += b' ' * ((-len(raw)) % 8)
    tensor_digests = {}
    merged_count = 0
    with safe_open(str(base), framework='pt', device='cpu') as source, safe_open(str(weights), framework='pt', device='cpu') as lora:
        keys = set(lora.keys())
        modules = {k.removesuffix('.lora_A.weight') for k in keys if k.endswith('.lora_A.weight')}
        expected = {m + suffix for m in modules for suffix in ('.lora_A.weight', '.lora_B.weight')}
        if keys != expected or not modules:
            raise ValueError('LoRA must contain complete A/B pairs only')
        if any(m + '.weight' not in tensors for m in modules):
            raise ValueError('LoRA target absent from base')
        with temporary.open('xb') as stream:
            stream.write(struct.pack('<Q', len(raw)))
            stream.write(raw)
            for index, key in enumerate(ordered):
                tensor = source.get_tensor(key)
                module = key.removesuffix('.weight') if key.endswith('.weight') else None
                if module in modules:
                    a = lora.get_tensor(module + '.lora_A.weight').float()
                    b = lora.get_tensor(module + '.lora_B.weight').float()
                    if a.shape[0] != rank or b.shape[1] != rank or list(tensor.shape) != [b.shape[0], a.shape[1]]:
                        raise ValueError('LoRA/base shape mismatch: ' + module)
                    tensor = torch.add(tensor, b @ a, alpha=scale)
                    if not torch.isfinite(tensor).all():
                        raise ValueError('Nonfinite merged weight: ' + key)
                    del a, b
                    merged_count += 1
                data = memoryview(tensor.contiguous().numpy()).cast('B')
                tensor_digests[key] = hashlib.sha256(data).hexdigest()
                stream.write(data)
                del data, tensor
                if index % 200 == 0:
                    print(f'MERGE tensor={index}/{len(ordered)} merged_modules={merged_count}', flush=True)
        if merged_count != len(modules):
            raise ValueError('Not all adapter modules were merged')
    gc.collect()
    with safe_open(str(temporary), framework='pt', device='cpu') as merged:
        if set(merged.keys()) != set(tensors):
            raise ValueError('Merged file tensor names differ')
        for index, key in enumerate(ordered):
            tensor = merged.get_tensor(key)
            data = memoryview(tensor.numpy()).cast('B')
            if hashlib.sha256(data).hexdigest() != tensor_digests[key]:
                raise ValueError('Readback mismatch: ' + key)
            del data, tensor
            if index % 400 == 0:
                print(f'VERIFY tensor={index}/{len(ordered)}', flush=True)
    temporary.replace(output)
    report = dict(format='physmend-merged-v1', generation_origin_step=8000, final_edit_step=10000,
                  formula='W_final = W_generation8000 + (alpha/r) * B_edit10000 @ A_edit10000',
                  rank=rank, alpha=config['lora_alpha'], scale=scale, dtype='float32',
                  merged_modules=merged_count, tensors=len(tensors),
                  untouched_tensors=len(tensors)-merged_count,
                  every_saved_tensor_readback_sha256_verified=True,
                  base_sha256=sha256(base), adapter_sha256=sha256(weights),
                  merged_sha256=sha256(output), merged_bytes=output.stat().st_size,
                  seconds=time.perf_counter()-started,
                  gpu_inference_rerun=False,
                  numerical_note='FP32 fused weights; runtime BF16 rounding can differ from separate BF16 base/LoRA evaluation.')
    output.parent.parent.joinpath('MERGE_COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Base transformer safetensors file')
    parser.add_argument('--adapter', type=Path, required=True, help='Adapter directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    merge(args.base, args.adapter, args.output)
