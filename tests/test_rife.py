import os
import sys
import tempfile
import unittest

import numpy as np

# Prevent tinygrad from crashing if CPU env var is set to non-numeric string (e.g. CPU=x86_64)
if os.environ.get('CPU') not in (None, '0', '1'):
    os.environ.pop('CPU', None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tinygrad import Tensor

from rife_v46 import IFNet, Model, load_safetensors_weights, warp
from tinyrife import get_image_files, load_image, resolve_model_path, save_image


class TestRife(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model()
        cls.model.eval()
        cls.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        cls.weights_path = os.path.join(cls.models_dir, 'rife-v4.6.safetensors')

    def test_01_ifnet_direct(self):
        """Test direct IFNet invocation with default scale_list and arguments."""
        ifnet = IFNet()
        dummy_pair = Tensor.rand(1, 6, 64, 64)
        flow_list, _mask, merged = ifnet(dummy_pair)
        self.assertEqual(len(flow_list), 5)
        self.assertEqual(merged.shape, (1, 3, 64, 64))
        self.assertFalse(np.isnan(merged.numpy()).any())

    def test_02_weight_loading_if_present(self):
        """Test loading safetensors weights into Model."""
        if os.path.exists(self.weights_path):
            load_safetensors_weights(self.model, self.weights_path)
        else:
            self.skipTest(f"Weights file not found at {self.weights_path}")

    def test_03_inference_timesteps(self):
        """Test forward inference at multiple interpolation timesteps."""
        img0 = Tensor.rand(1, 3, 64, 64)
        img1 = Tensor.rand(1, 3, 64, 64)
        for ts in (0.5, 0.25, 0.75):
            out = self.model.inference(img0, img1, timestep=ts)
            self.assertEqual(out.shape, (1, 3, 64, 64))
            out_np = out.numpy()
            self.assertFalse(np.isnan(out_np).any())
            self.assertFalse(np.isinf(out_np).any())

    def test_04_hybrid_tiling(self):
        """Test hybrid tiled inference and invalid tile size rejection."""
        img0 = Tensor.rand(1, 3, 128, 128)
        img1 = Tensor.rand(1, 3, 128, 128)
        out = self.model.inference(img0, img1, timestep=0.5, tile=64, tile_pad=8)
        self.assertEqual(out.shape, (1, 3, 128, 128))
        out_np = out.numpy()
        self.assertFalse(np.isnan(out_np).any())

        # Verify rejection of non-64-multiple tile sizes (e.g. 100)
        with self.assertRaises(AssertionError):
            self.model.inference(img0, img1, tile=100)

    def test_05_tta_modes(self):
        """Test spatial, temporal, and combined TTA."""
        img0 = Tensor.rand(1, 3, 64, 64)
        img1 = Tensor.rand(1, 3, 64, 64)

        out_temporal = self.model.inference(img0, img1, timestep=0.5, tta_temporal=True)
        self.assertEqual(out_temporal.shape, (1, 3, 64, 64))
        self.assertFalse(np.isnan(out_temporal.numpy()).any())

        out_spatial = self.model.inference(img0, img1, timestep=0.5, tta=True)
        self.assertEqual(out_spatial.shape, (1, 3, 64, 64))
        self.assertFalse(np.isnan(out_spatial.numpy()).any())

        out_both = self.model.inference(img0, img1, timestep=0.5, tta=True, tta_temporal=True)
        self.assertEqual(out_both.shape, (1, 3, 64, 64))
        self.assertFalse(np.isnan(out_both.numpy()).any())

    def test_06_non_standard_dims(self):
        """Test non-square and non-64-multiple dimensions with TTA."""
        img0_rect = Tensor.rand(1, 3, 70, 90)
        img1_rect = Tensor.rand(1, 3, 70, 90)
        out_rect = self.model.inference(img0_rect, img1_rect, timestep=0.5, tta=True, tta_temporal=True)
        self.assertEqual(out_rect.shape, (1, 3, 70, 90))
        self.assertFalse(np.isnan(out_rect.numpy()).any())

    def test_07_cli_helpers(self):
        """Test CLI image I/O and path resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test image save & load
            test_tensor = Tensor.rand(1, 3, 32, 32)
            img_path = os.path.join(tmpdir, "test.png")
            save_image(test_tensor, img_path)
            self.assertTrue(os.path.exists(img_path))

            loaded = load_image(img_path)
            self.assertEqual(loaded.shape, (1, 3, 32, 32))

            # Test natural sorting of image files
            for fname in ["frame_10.png", "frame_2.png", "frame_1.png"]:
                with open(os.path.join(tmpdir, fname), "w") as f:
                    f.write("")
            sorted_files = [os.path.basename(p) for p in get_image_files(tmpdir)]
            self.assertEqual(sorted_files, ["frame_1.png", "frame_2.png", "frame_10.png", "test.png"])

            # Test resolve_model_path valid & invalid
            resolved = resolve_model_path(self.models_dir)
            self.assertTrue(os.path.exists(resolved))
            with self.assertRaises(FileNotFoundError):
                resolve_model_path(os.path.join(tmpdir, "non_existent.safetensors"))
            with self.assertRaises(FileNotFoundError):
                resolve_model_path(tmpdir)

            # Test load_safetensors_weights invalid path
            with self.assertRaises(FileNotFoundError):
                load_safetensors_weights(self.model, os.path.join(tmpdir, "missing.safetensors"))

    def test_08_small_dimensions_tta(self):
        """Test TTA on dimensions smaller than 64 to verify padding guard."""
        img0_small = Tensor.rand(1, 3, 32, 32)
        img1_small = Tensor.rand(1, 3, 32, 32)
        out = self.model.inference(img0_small, img1_small, timestep=0.5, tta=True)
        self.assertEqual(out.shape, (1, 3, 32, 32))
        self.assertFalse(np.isnan(out.numpy()).any())

    def test_09_warp_identity(self):
        """Test that zero optical flow warp preserves the input image identically."""
        img = Tensor.rand(1, 3, 32, 32)
        zero_flow = Tensor.zeros(1, 4, 32, 32)
        warped0 = warp(img, zero_flow[:, :2])
        warped1 = warp(img, zero_flow[:, 2:4])
        np.testing.assert_allclose(warped0.numpy(), img.numpy(), atol=1e-5)
        np.testing.assert_allclose(warped1.numpy(), img.numpy(), atol=1e-5)

    def test_10_directory_sequence_flow(self):
        """Test sequential directory processing workflow."""
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
            # Create 3 test frames
            for i in range(3):
                t = Tensor.rand(1, 3, 64, 64)
                save_image(t, os.path.join(input_dir, f"frame_{i:04d}.png"))

            input_files = get_image_files(input_dir)
            self.assertEqual(len(input_files), 3)

            out_idx = 0
            prev = load_image(input_files[0])
            save_image(prev, os.path.join(output_dir, f"{out_idx:08d}.png"))
            out_idx += 1
            for idx in range(1, len(input_files)):
                cur = load_image(input_files[idx])
                out = self.model.inference(prev, cur, timestep=0.5, tile=64, tile_pad=8)
                save_image(out, os.path.join(output_dir, f"{out_idx:08d}.png"))
                out_idx += 1
                save_image(cur, os.path.join(output_dir, f"{out_idx:08d}.png"))
                out_idx += 1
                prev = cur

            out_files = get_image_files(output_dir)
            # 3 input frames -> 2*3 - 1 = 5 output frames
            self.assertEqual(len(out_files), 5)
            self.assertEqual(os.path.basename(out_files[0]), "00000000.png")
            self.assertEqual(os.path.basename(out_files[4]), "00000004.png")


if __name__ == '__main__':
    unittest.main()