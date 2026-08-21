# tinyrife - RIFE v4.6 Frame Interpolation (tinygrad)

## Project Structure

```
tinyrife/
├── src/
│   └── rife_v46.py      # Core model implementation (Head, ResConv, IFBlock, IFNet, Model, warp, bilinear_grid_sample, load_safetensors_weights)
├── tinyrife.py          # Main CLI entry point (pair mode, dir mode, verbose, timing)
├── tests/
│   └── test_rife.py     # Smoke test for rife_v46.Model
├── models/
│   └── rife-v4.6.safetensors # Safetensors weights
└── demo/                # Demo/test data (input frames, outputs)
```

## Environment & Hardware

Prefer running with OpenCL on Radeon via RustiCL:
```bash
DEV=CL RUSTICL_ENABLE=radeonsi
```
(e.g., `DEV=CL RUSTICL_ENABLE=radeonsi python tinyrife.py ...`)

## Entry Point

**Main CLI:** `tinyrife.py`

```bash
# Pair mode (single interpolation)
DEV=CL RUSTICL_ENABLE=radeonsi python tinyrife.py -0 img0.png -1 img1.png -o output.png -m models/rife-v4.6.safetensors

# Directory mode (sequence interpolation)
DEV=CL RUSTICL_ENABLE=radeonsi python tinyrife.py -i input_frames/ -o output_frames/ -m models/rife-v4.6.safetensors

# With tiling for large images
DEV=CL RUSTICL_ENABLE=radeonsi python tinyrife.py -0 a.png -1 b.png -o out.png --tile 128 --tile_pad 10
```

## Key Files

| File | Purpose |
|------|---------|
| `src/rife_v46.py` | Single source of truth for model + weight loader |
| `tinyrife.py` | CLI only — imports `Model`, `load_safetensors_weights` from `rife_v46` |
| `tests/test_rife.py` | Verifies model loads + forward pass |

## Dependencies

- `tinygrad` (inference engine)
- `safetensors` (weight loading)
- `opencv-python` (image I/O)
- `numpy`

## Weight Loading

`load_safetensors_weights(model, path)` in `src/rife_v46.py` loads `flownet.safetensors` or `rife-v4.6.safetensors` into tinygrad `Model`. Called by `tinyrife.py`.

## Demo Data

Test frames and outputs stored under `demo/`:
- Demo test frames (`demo/i0.png`, `demo/i1.png`, etc.)
- Test outputs and heatmaps (`demo/out_*.png`)

## Commands

```bash
# Run CLI
DEV=CL RUSTICL_ENABLE=radeonsi python tinyrife.py -h

# Run smoke test
DEV=CL RUSTICL_ENABLE=radeonsi python tests/test_rife.py
```

## Notes

- Model: RIFE v4.6 (IFNet with 5 IFBlocks at scales 8,4,2,1 + 1)
- Input: RGB images, normalized to [0,1], shape (B,3,H,W)
- Output: Interpolated frame at timestep (default 0.5)
- Tiling is hybrid: coarse IFBlocks (0,1) run on whole image (globally consistent flow), fine blocks (2,3,4) per tile — this kills tile-boundary seams. Uses TinyJit for constant-shape kernel compilation
- Most efficient tile_pad = tile/8 (56% useful compute); tile ≥ 4×tile_pad. Min quality pad = 32, quality margin = 64 (covers block2 RF ~68px)
- FP16 is pointless here (measured ~20% slower): tinygrad CL fp16 = plain fp16 arithmetic with no pack4/8 vectorization (unlike ncnn), and tiled runs are host-roundtrip-bound anyway. Perf lever is reducing tile host roundtrips, not precision.
- **Avoid non-standard dims (1080p, 720p, etc)**: RIFE needs H/W divisible by 64 (for scale-16 + stride-2 convs). Non-multiples of 128 (e.g. 1080, 720) trigger JIT recompilation per frame and can hang GPU. Use `--tile 128` to force 64-divisible padding, or resize to 128/256/512/1024 multiples.