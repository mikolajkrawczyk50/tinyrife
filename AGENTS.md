# tinyrife - RIFE v4.6 Frame Interpolation (tinygrad)

## Project Structure

```
tinyrife/
├── rife_v46.py          # Core model implementation (Head, ResConv, IFBlock, IFNet, Model, warp, bilinear_grid_sample, load_torch_weights)
├── tinyrife.py          # Main CLI entry point (pair mode, dir mode, verbose, timing)
├── test_rife.py         # Smoke test for rife_v46.Model
├── models/
│   └── rife-v4.6/       # PyTorch weights (flownet.pkl)
└── demo/                # Demo/test data (input frames, outputs)
```

## Entry Point

**Main CLI:** `tinyrife.py`

```bash
# Pair mode (single interpolation)
python tinyrife.py -0 img0.png -1 img1.png -o output.png -m models/rife-v4.6

# Directory mode (sequence interpolation)
python tinyrife.py -i input_frames/ -o output_frames/ -m models/rife-v4.6

# With tiling for large images
python tinyrife.py -0 a.png -1 b.png -o out.png --tile 128 --tile_pad 10
```

## Key Files

| File | Purpose |
|------|---------|
| `rife_v46.py` | Single source of truth for model + weight loader |
| `tinyrife.py` | CLI only — imports `Model`, `load_torch_weights` from `rife_v46` |
| `test_rife.py` | Verifies model loads + forward pass |

## Dependencies

- `tinygrad` (inference engine)
- `torch` (weight loading only)
- `opencv-python` (image I/O)
- `numpy`

## Weight Loading

`load_torch_weights(model, path)` in `rife_v46.py` loads PyTorch `flownet.pkl` into tinygrad `Model`. Called by both `tinyrife.py` and `test_rife.py`.

## Demo Data

Test frames and outputs stored under `demo/`:
- `demo/inputs/` — 302 frame sequence
- `demo/outputs/` — 52 interpolated frames
- `demo/test_input*/` `demo/test_output*/` — small test sets

## Commands

```bash
# Run CLI
python tinyrife.py -h

# Run smoke test
python test_rife.py
```

## Notes

- Model: RIFE v4.6 (IFNet with 5 IFBlocks at scales 8,4,2,1 + 1)
- Input: RGB images, normalized to [0,1], shape (B,3,H,W)
- Output: Interpolated frame at timestep (default 0.5)
- Tiling uses TinyJit for constant-shape kernel compilation