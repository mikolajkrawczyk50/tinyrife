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

print("Test passed!")