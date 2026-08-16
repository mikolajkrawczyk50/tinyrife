#!/usr/bin/env python3
"""
tinyrife - RIFE v4.6 frame interpolation CLI (tinygrad)
Mirrors basic rife-ncnn-vulkan flags: -0/-1, -i/-o, -m, -v
"""
import argparse
import math
import os
import sys
import time
import glob
import cv2
import numpy as np
from tinygrad import Tensor, TinyJit

from rife_v46 import Model, load_safetensors_weights


def resolve_model_path(path):
    if os.path.isdir(path):
        st = os.path.join(path, 'flownet.safetensors')
        if os.path.exists(st):
            return st
        npz = os.path.join(path, 'flownet.npz')
        if os.path.exists(npz):
            return npz
        pkl = os.path.join(path, 'flownet.pkl')
        if os.path.exists(pkl):
            return pkl
        raise FileNotFoundError(f'flownet.safetensors, flownet.npz or flownet.pkl not found in {path}')
    return path


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = Tensor(img.transpose(2, 0, 1)).unsqueeze(0)
    return img


def save_image(tensor, path):
    img = tensor[0].numpy().transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def get_image_files(dir_path):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(dir_path, ext)))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description='RIFE v4.6 frame interpolation (tinygrad)')
    parser.add_argument('-0', dest='img0', help='Input image 0 path')
    parser.add_argument('-1', dest='img1', help='Input image 1 path')
    parser.add_argument('-i', dest='input_dir', help='Input image directory')
    parser.add_argument('-o', dest='output', required=True, help='Output image path (pair mode) or directory (dir mode)')
    parser.add_argument('-m', dest='model', default='models/rife-v4.6', help='Model path (directory or file, default: models/rife-v4.6)')
    parser.add_argument('--tile', type=int, default=0, help='tile size for processing, 0 disables tiling (default: 0)')
    parser.add_argument('--tile_pad', type=int, default=32, help='pad around each tile (default: 32)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    has_pair = args.img0 is not None and args.img1 is not None
    has_dir = args.input_dir is not None

    if not has_pair and not has_dir:
        parser.error('Either -0/-1 (pair mode) or -i (directory mode) required')
    if has_pair and has_dir:
        parser.error('Cannot use both pair mode (-0/-1) and directory mode (-i)')

    if args.verbose:
        print(f'Loading model from {args.model}...')
        if args.tile > 0:
            print(f'Tiling enabled: tile={args.tile}, tile_pad={args.tile_pad}, base={args.tile - 2 * args.tile_pad}')

    start_time = time.time()
    model = Model()
    load_safetensors_weights(model, resolve_model_path(args.model))
    model.eval()

    if args.verbose:
        print(f'Model loaded in {time.time() - start_time:.2f}s')

    if has_pair:
        if args.verbose:
            print(f'Loading {args.img0} and {args.img1}...')
        img0 = load_image(args.img0)
        img1 = load_image(args.img1)
        if args.verbose:
            print(f'Input shapes: {img0.shape}, {img1.shape}')

        if args.verbose:
            print('Running inference...')
        t0 = time.time()
        out = model.inference(img0, img1, timestep=0.5, tile=args.tile, tile_pad=args.tile_pad)
        if args.verbose:
            print(f'Inference took {time.time() - t0:.2f}s')

        save_image(out, args.output)
        if args.verbose:
            print(f'Saved: {args.output}')

    else:
        input_files = get_image_files(args.input_dir)
        if len(input_files) < 2:
            sys.exit(f'Error: Need at least 2 images in {args.input_dir}, found {len(input_files)}')

        os.makedirs(args.output, exist_ok=True)

        if args.verbose:
            print(f'Found {len(input_files)} frames, will produce {len(input_files) * 2 - 1} output frames')

        out_idx = 0
        prev = load_image(input_files[0])
        save_image(prev, os.path.join(args.output, f'{out_idx:08d}.png'))
        out_idx += 1
        for idx in range(1, len(input_files)):
            cur = load_image(input_files[idx])
            if args.verbose:
                print(f'  [{idx}/{len(input_files)-1}] {os.path.basename(input_files[idx-1])} + {os.path.basename(input_files[idx])}')
            out = model.inference(prev, cur, timestep=0.5, tile=args.tile, tile_pad=args.tile_pad)
            save_image(out, os.path.join(args.output, f'{out_idx:08d}.png'))
            out_idx += 1
            save_image(cur, os.path.join(args.output, f'{out_idx:08d}.png'))
            out_idx += 1
            prev = cur

        if args.verbose:
            print(f'Done. Output in {args.output}')

    if args.verbose:
        print(f'Total time: {time.time() - start_time:.2f}s')


if __name__ == '__main__':
    main()