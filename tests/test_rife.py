import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from rife_v46 import Model

from tinygrad import Tensor
import numpy as np

model = Model()
model.eval()

B, C, H, W = 1, 3, 256, 256
img0 = Tensor.rand(B, C, H, W)
img1 = Tensor.rand(B, C, H, W)

print(f"Input shapes: img0={img0.shape}, img1={img1.shape}")

out = model.inference(img0, img1, timestep=0.5)

print(f"Output shape: {out.shape}")

out2 = model.inference(img0, img1, timestep=0.25)
print(f"Output shape (0.25): {out2.shape}")

out3 = model.inference(img0, img1, timestep=0.75)
print(f"Output shape (0.75): {out3.shape}")

# Test TTA modes
print("Testing Temporal TTA...")
out_temporal = model.inference(img0, img1, timestep=0.5, tta_temporal=True)
assert out_temporal.shape == (B, C, H, W)
print(f"Temporal TTA passed, shape: {out_temporal.shape}")

print("Testing Spatial TTA...")
out_spatial = model.inference(img0, img1, timestep=0.5, tta=True)
assert out_spatial.shape == (B, C, H, W)
print(f"Spatial TTA passed, shape: {out_spatial.shape}")

print("Testing Spatial + Temporal TTA...")
out_both = model.inference(img0, img1, timestep=0.5, tta=True, tta_temporal=True)
assert out_both.shape == (B, C, H, W)
print(f"Spatial + Temporal TTA passed, shape: {out_both.shape}")

# Test with non-square / non-64-multiple dims
img0_rect = Tensor.rand(1, 3, 100, 150)
img1_rect = Tensor.rand(1, 3, 100, 150)
out_rect = model.inference(img0_rect, img1_rect, timestep=0.5, tta=True, tta_temporal=True)
assert out_rect.shape == (1, 3, 100, 150)
print(f"Non-standard shape TTA passed: {out_rect.shape}")

print("All tests passed!")