[![Hugging Face Models](https://img.shields.io/badge/HuggingFace-Models-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/lxr2003/Physmend)
[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/lxr2003/PM-Pile/tree/main)

Region editing:

```bash
bash run_edit.sh \
  --video examples/liquid_overflow/source.mp4 \
  --mask examples/liquid_overflow/mask.npy \
  --instruct examples/liquid_overflow/instruct.txt \
  --output outputs/edit.mp4
```

`examples/liquid_overflow/`: `source.mp4` is the input video, `instruct.txt` is the editing instruction, `mask.npy` is the edit mask, and `reference_edited.mp4` is the archived edited video.

Use `--instruction "Your editing instruction"` instead of `--instruct` to pass text directly. The video's shortest side must be 480 pixels. Positive mask values mark editable pixels; 0 preserves them. Masks must match the padded canvas. The example mask has shape `[81,480,864]`.

Text-to-video:

```bash
bash run_t2v.sh --prompt "Red liquid pours into a glass and overflows onto a tray." --output outputs/t2v.mp4
```

Image-to-video:

```bash
bash run_i2v.sh \
  --image examples/liquid_overflow/first_frame.jpg \
  --prompt-file examples/liquid_overflow/prompt.txt \
  --output outputs/i2v.mp4
```

All three entry points accept `--model-path /path/to/model`. Add `--check` to validate model files and inputs without running inference.
