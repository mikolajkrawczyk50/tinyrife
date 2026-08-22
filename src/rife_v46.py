import math
import os
import time

# Prevent tinygrad from crashing if CPU env var is set to non-numeric string (e.g. CPU=x86_64)
if os.environ.get('CPU') not in (None, '0', '1'):
    os.environ.pop('CPU', None)

import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.nn import Conv2d, ConvTranspose2d


def warp(tenInput: Tensor, tenFlow: Tensor) -> Tensor:
    B, _, H, W = tenInput.shape
    
    tenHorizontal = Tensor.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    tenVertical = Tensor.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    tenGrid = tenHorizontal.cat(tenVertical, dim=1)
    
    tenFlow_scaled = tenFlow[:, 0:1, :, :] / ((W - 1.0) / 2.0)
    tenFlow_scaled = tenFlow_scaled.cat(tenFlow[:, 1:2, :, :] / ((H - 1.0) / 2.0), dim=1)
    
    g = (tenGrid + tenFlow_scaled).permute(0, 2, 3, 1)
    
    return bilinear_grid_sample(tenInput, g)


def bilinear_grid_sample(input: Tensor, grid: Tensor) -> Tensor:
    B, C, H, W = input.shape
    _, Hg, Wg, _ = grid.shape
    
    x = ((grid[..., 0] + 1) * (W - 1) / 2).clip(0, W - 1)
    y = ((grid[..., 1] + 1) * (H - 1) / 2).clip(0, H - 1)
    
    x0 = x.floor().cast('int32')
    x1 = (x0 + 1).clip(0, W - 1)
    y0 = y.floor().cast('int32')
    y1 = (y0 + 1).clip(0, H - 1)
    
    wx1 = x - x0.cast('float32')
    wx0 = 1 - wx1
    wy1 = y - y0.cast('float32')
    wy0 = 1 - wy1
    
    out_list = []
    for b in range(B):
        input_b = input[b].reshape(C, H * W)
        x0_b = x0[b].reshape(-1)
        x1_b = x1[b].reshape(-1)
        y0_b = y0[b].reshape(-1)
        y1_b = y1[b].reshape(-1)

        idx00 = y0_b * W + x0_b
        idx01 = y0_b * W + x1_b
        idx10 = y1_b * W + x0_b
        idx11 = y1_b * W + x1_b
        idx = idx00.cat(idx01).cat(idx10).cat(idx11).reshape(-1)

        gathered = input_b[:, idx].reshape(C, 4, Hg, Wg)
        v00 = gathered[:, 0]
        v01 = gathered[:, 1]
        v10 = gathered[:, 2]
        v11 = gathered[:, 3]

        wx0_b = wx0[b]
        wx1_b = wx1[b]
        wy0_b = wy0[b]
        wy1_b = wy1[b]

        out_b = v00 * wx0_b * wy0_b + v01 * wx1_b * wy0_b + v10 * wx0_b * wy1_b + v11 * wx1_b * wy1_b
        out_list.append(out_b.unsqueeze(0))
    
    return Tensor.cat(*out_list, dim=0)


def _tta_augment_img(x: Tensor, idx: int) -> Tensor:
    """Apply one of 8 TTA augmentations to an image [B, C, H, W] (rife-ncnn-vulkan matching).
    0: original
    1: flip H
    2: flip H + flip V
    3: flip V
    4: transpose (swap H, W)
    5: transpose + flip H
    6: transpose + flip H + flip V
    7: transpose + flip V
    """
    if idx == 0:
        return x
    elif idx == 1:
        return x.flip(3)
    elif idx == 2:
        return x.flip(2).flip(3)
    elif idx == 3:
        return x.flip(2)
    elif idx == 4:
        return x.permute(0, 1, 3, 2)
    elif idx == 5:
        return x.permute(0, 1, 3, 2).flip(3)
    elif idx == 6:
        return x.permute(0, 1, 3, 2).flip(2).flip(3)
    elif idx == 7:
        return x.permute(0, 1, 3, 2).flip(2)
    return x


# Alias for backward compatibility
_tta_augment = _tta_augment_img


def _tta_deaugment_img(img: Tensor, idx: int) -> Tensor:
    """Reverse one of 8 TTA augmentations on image tensor."""
    if idx == 0:
        return img
    elif idx == 1:
        return img.flip(3)
    elif idx == 2:
        return img.flip(2).flip(3)
    elif idx == 3:
        return img.flip(2)
    elif idx == 4:
        return img.permute(0, 1, 3, 2)
    elif idx == 5:
        return img.flip(3).permute(0, 1, 3, 2)
    elif idx == 6:
        return img.flip(2).flip(3).permute(0, 1, 3, 2)
    elif idx == 7:
        return img.flip(2).permute(0, 1, 3, 2)
    return img


def _tta_deaugment_flow_mask(flow_mask: Tensor, idx: int) -> Tensor:
    """De-augment 5-channel flow+mask tensor [flow0_x, flow0_y, flow1_x, flow1_y, mask] to canonical coordinates."""
    if idx == 0:
        return flow_mask
    elif idx == 1:  # flip H: x negated, y unchanged
        f = flow_mask.flip(3)
        return Tensor.cat(-f[:, 0:1], f[:, 1:2], -f[:, 2:3], f[:, 3:4], f[:, 4:5], dim=1)
    elif idx == 2:  # flip H + flip V: x negated, y negated
        f = flow_mask.flip(2).flip(3)
        return Tensor.cat(-f[:, 0:1], -f[:, 1:2], -f[:, 2:3], -f[:, 3:4], f[:, 4:5], dim=1)
    elif idx == 3:  # flip V: x unchanged, y negated
        f = flow_mask.flip(2)
        return Tensor.cat(f[:, 0:1], -f[:, 1:2], f[:, 2:3], -f[:, 3:4], f[:, 4:5], dim=1)
    elif idx == 4:  # transpose: swap x and y channels
        return flow_mask.permute(0, 1, 3, 2)[:, [1, 0, 3, 2, 4], :, :]
    elif idx == 5:  # transpose + flip H
        f = flow_mask.flip(3).permute(0, 1, 3, 2)
        return Tensor.cat(f[:, 1:2], -f[:, 0:1], f[:, 3:4], -f[:, 2:3], f[:, 4:5], dim=1)
    elif idx == 6:  # transpose + flip H + flip V
        f = flow_mask.flip(2).flip(3).permute(0, 1, 3, 2)
        return Tensor.cat(-f[:, 1:2], -f[:, 0:1], -f[:, 3:4], -f[:, 2:3], f[:, 4:5], dim=1)
    elif idx == 7:  # transpose + flip V
        f = flow_mask.flip(2).permute(0, 1, 3, 2)
        return Tensor.cat(-f[:, 1:2], f[:, 0:1], -f[:, 3:4], f[:, 2:3], f[:, 4:5], dim=1)
    return flow_mask


def _tta_augment_flow_mask(flow_mask: Tensor, idx: int) -> Tensor:
    """Augment canonical 5-channel flow+mask tensor to branch coordinates."""
    if idx == 0:
        return flow_mask
    elif idx == 1:
        f = flow_mask.flip(3)
        return Tensor.cat(-f[:, 0:1], f[:, 1:2], -f[:, 2:3], f[:, 3:4], f[:, 4:5], dim=1)
    elif idx == 2:
        f = flow_mask.flip(2).flip(3)
        return Tensor.cat(-f[:, 0:1], -f[:, 1:2], -f[:, 2:3], -f[:, 3:4], f[:, 4:5], dim=1)
    elif idx == 3:
        f = flow_mask.flip(2)
        return Tensor.cat(f[:, 0:1], -f[:, 1:2], f[:, 2:3], -f[:, 3:4], f[:, 4:5], dim=1)
    elif idx == 4:
        return flow_mask.permute(0, 1, 3, 2)[:, [1, 0, 3, 2, 4], :, :]
    elif idx == 5:
        f = flow_mask.permute(0, 1, 3, 2)
        return Tensor.cat(-f[:, 1:2], f[:, 0:1], -f[:, 3:4], f[:, 2:3], f[:, 4:5], dim=1).flip(3)
    elif idx == 6:
        f = flow_mask.permute(0, 1, 3, 2)
        return Tensor.cat(-f[:, 1:2], -f[:, 0:1], -f[:, 3:4], -f[:, 2:3], f[:, 4:5], dim=1).flip(2).flip(3)
    elif idx == 7:
        f = flow_mask.permute(0, 1, 3, 2)
        return Tensor.cat(f[:, 1:2], -f[:, 0:1], f[:, 3:4], -f[:, 2:3], f[:, 4:5], dim=1).flip(2)
    return flow_mask


def _realized_average(tensor_list: list) -> Tensor:
    """Average a list of tensors using pairwise realized reduction to stay well within compute shader storage block limits (e.g. ModernGL max 16)."""
    n = len(tensor_list)
    if n == 1:
        return tensor_list[0]
    cur = [t.realize() for t in tensor_list]
    while len(cur) > 1:
        next_level = []
        for i in range(0, len(cur), 2):
            if i + 1 < len(cur):
                next_level.append((cur[i] + cur[i + 1]).realize())
            else:
                next_level.append(cur[i])
        cur = next_level
    return (cur[0] * (1.0 / n)).realize()


def _temporal_average_flow_mask(fm_fwd: Tensor, fm_rev: Tensor):
    """Temporal average forward and reverse flow+mask.
    Forward flow:  [flow0_x, flow0_y, flow1_x, flow1_y, mask]
    Reverse flow:  [rev0_x, rev0_y, rev1_x, rev1_y, rev_mask]
    """
    avg_f0_x = (fm_fwd[:, 0:1] + fm_rev[:, 2:3]) * 0.5
    avg_f0_y = (fm_fwd[:, 1:2] + fm_rev[:, 3:4]) * 0.5
    avg_f1_x = (fm_fwd[:, 2:3] + fm_rev[:, 0:1]) * 0.5
    avg_f1_y = (fm_fwd[:, 3:4] + fm_rev[:, 1:2]) * 0.5
    avg_mask = (fm_fwd[:, 4:5] - fm_rev[:, 4:5]) * 0.5

    fwd_merged = Tensor.cat(avg_f0_x, avg_f0_y, avg_f1_x, avg_f1_y, avg_mask, dim=1)
    rev_merged = Tensor.cat(avg_f1_x, avg_f1_y, avg_f0_x, avg_f0_y, -avg_mask, dim=1)
    return fwd_merged, rev_merged


class Head:
    def __init__(self):
        self.cnn0 = Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = ConvTranspose2d(16, 4, 4, 2, 1)
        
    def __call__(self, x: Tensor, feat=False):
        x0 = self.cnn0(x).leaky_relu(0.2)
        x1 = self.cnn1(x0).leaky_relu(0.2)
        x2 = self.cnn2(x1).leaky_relu(0.2)
        x3 = self.cnn3(x2)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class ResConv:
    def __init__(self, c, dilation=1):
        self.conv = Conv2d(c, c, 3, 1, dilation, dilation=dilation)
        self.beta = Tensor.ones((1, c, 1, 1))
        
    def __call__(self, x: Tensor) -> Tensor:
        return (self.conv(x) * self.beta + x).leaky_relu(0.2)


class IFBlock:
    def __init__(self, in_planes: int, c: int = 64):
        self.conv0 = [
            Conv2d(in_planes, c//2, 3, 2, 1),
            lambda x: x.leaky_relu(0.2),
            Conv2d(c//2, c, 3, 2, 1),
            lambda x: x.leaky_relu(0.2),
        ]
        self.convblock = [ResConv(c) for _ in range(8)]
        self.lastconv = [
            ConvTranspose2d(c, 4*13, 4, 2, 1),
        ]
        
    def __call__(self, x: Tensor, flow=None, scale=1):
        x = x.interpolate(size=(int(x.shape[2] // scale), int(x.shape[3] // scale)), mode='linear', align_corners=False)
        if flow is not None:
            flow = flow.interpolate(size=(int(flow.shape[2] // scale), int(flow.shape[3] // scale)), mode='linear', align_corners=False)
            flow = flow * (1.0 / scale)
            x = x.cat(flow, dim=1)
        
        feat = x
        for layer in self.conv0:
            feat = layer(feat)
        
        for layer in self.convblock:
            feat = layer(feat)
        
        tmp = self.lastconv[0](feat)
        
        # PixelShuffle(2): [B, C*4, H, W] -> [B, C, 2H, 2W] where C=13
        B, C, H, W = tmp.shape
        r = 2
        tmp = tmp.reshape(B, C // (r*r), r, r, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C // (r*r), H*r, W*r)
        
        if scale != 1:
            tmp = tmp.interpolate(size=(int(tmp.shape[2] * scale), int(tmp.shape[3] * scale)), mode='linear', align_corners=False)
        
        flow_out = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat_out = tmp[:, 5:]
        return flow_out, mask, feat_out


class IFNet:
    def __init__(self):
        self.block0 = IFBlock(15, c=192)   # 3+3+4+4+1
        self.block1 = IFBlock(28, c=128)   # 3+3+4+4+1+1+8 + 4(flow)
        self.block2 = IFBlock(28, c=96)
        self.block3 = IFBlock(28, c=64)
        self.block4 = IFBlock(28, c=32)
        self.encode = Head()
        
    def __call__(self, x: Tensor, timestep=0.5, scale_list=None, start=0, end=5, warm=None, return_state=False):
        if scale_list is None:
            scale_list = [16, 8, 4, 2, 1]
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        
        if not isinstance(timestep, Tensor):
            timestep = Tensor([timestep]).view(1, 1, 1, 1).expand(x.shape[0], 1, img0.shape[2], img0.shape[3])
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        
        flow_list = []
        mask_list = []
        
        if warm is not None:
            flow, mask, feat, warped_img0, warped_img1 = warm
        else:
            warped_img0 = img0
            warped_img1 = img1
            flow = None
            mask = None
            feat = None
        
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        
        for i in range(start, end):
            if flow is None:
                inp = img0[:, :3].cat(img1[:, :3], dim=1).cat(f0, dim=1).cat(f1, dim=1).cat(timestep, dim=1)
                flow, mask, feat = blocks[i](inp, None, scale=scale_list[i])
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                inp = warped_img0[:, :3].cat(warped_img1[:, :3], dim=1).cat(wf0, dim=1).cat(wf1, dim=1).cat(timestep, dim=1).cat(mask, dim=1).cat(feat, dim=1)
                fd, mask, feat = blocks[i](inp, flow, scale=scale_list[i])
                flow = flow + fd
            
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
        
        if return_state:
            return flow_list, mask, (flow, mask, feat, warped_img0, warped_img1)
        
        mask = mask.sigmoid()
        merged = warped_img0 * mask + warped_img1 * (1 - mask)
        
        return flow_list, mask, merged


class Model:
    def __init__(self):
        self.flownet = IFNet()
        self.version = 4.6
        self._jit_cache = {}
        
    def load_model(self, path: str, epoch: int):
        pass
    
    def eval(self):
        pass
    
    def device(self):
        pass

    def _forward(self, img0: Tensor, img1: Tensor, timestep: float = 0.5, scale: float = 1.0) -> Tensor:
        imgs = img0.cat(img1, dim=1)
        scale_list = [int(16/scale), int(8/scale), int(4/scale), int(2/scale), int(1/scale)]
        _, _, merged = self.flownet(imgs, timestep, scale_list)
        return merged

    def _forward_coarse(self, img0: Tensor, img1: Tensor, timestep: float = 0.5, scale: float = 1.0):
        imgs = img0.cat(img1, dim=1)
        scale_list = [int(16/scale), int(8/scale), int(4/scale), int(2/scale), int(1/scale)]
        _, _, state = self.flownet(imgs, timestep, scale_list, start=0, end=2, return_state=True)
        return state

    def _forward_fine(self, img0: Tensor, img1: Tensor, state: Tensor,
                      timestep: float = 0.5, scale: float = 1.0) -> Tensor:
        imgs = img0.cat(img1, dim=1)
        scale_list = [int(16/scale), int(8/scale), int(4/scale), int(2/scale), int(1/scale)]
        flow = state[:, 0:4]
        mask = state[:, 4:5]
        feat = state[:, 5:13]
        warped0 = state[:, 13:16]
        warped1 = state[:, 16:19]
        warm = (flow, mask, feat, warped0, warped1)
        _, _, merged = self.flownet(imgs, timestep, scale_list, start=2, end=5, warm=warm)
        return merged

    def _jit_forward(self, img0: Tensor, img1: Tensor, timestep: float = 0.5, scale: float = 1.0) -> Tensor:
        _, _, H, W = img0.shape

        # Pad to multiple of 64 -> constant shape + exact scale-16 block divisions
        # (RIFE needs H/W divisible by 64: //16 then two stride-2 convs must divide exactly)
        ph = ((H - 1) // 64 + 1) * 64
        pw = ((W - 1) // 64 + 1) * 64
        pad_h = ph - H
        pad_w = pw - W

        if pad_h > 0 or pad_w > 0:
            img0 = img0.pad((0, pad_w, 0, pad_h), mode='constant', value=0)
            img1 = img1.pad((0, pad_w, 0, pad_h), mode='constant', value=0)

        jit_key = (ph, pw, float(scale), float(timestep))
        if jit_key not in self._jit_cache:
            self._jit_cache[jit_key] = TinyJit(
                lambda t0, t1: self._forward(t0, t1, timestep=timestep, scale=scale)
            )
        merged = self._jit_cache[jit_key](img0, img1)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            merged = merged[:, :, :H, :W]

        return merged

    def _jit_coarse(self, img0: Tensor, img1: Tensor, timestep: float = 0.5, scale: float = 1.0):
        _, _, H, W = img0.shape
        ph = ((H - 1) // 64 + 1) * 64
        pw = ((W - 1) // 64 + 1) * 64
        pad_h = ph - H
        pad_w = pw - W

        if pad_h > 0 or pad_w > 0:
            img0 = img0.pad((0, pad_w, 0, pad_h), mode='constant', value=0)
            img1 = img1.pad((0, pad_w, 0, pad_h), mode='constant', value=0)

        jit_key = (ph, pw, float(scale), float(timestep), 'coarse')
        if jit_key not in self._jit_cache:
            self._jit_cache[jit_key] = TinyJit(
                lambda t0, t1: self._forward_coarse(t0, t1, timestep=timestep, scale=scale)
            )
        state = self._jit_cache[jit_key](img0, img1)

        if pad_h > 0 or pad_w > 0:
            state = tuple(s[:, :, :H, :W] for s in state)

        return state

    def _jit_fine(self, img0: Tensor, img1: Tensor, state: Tensor,
                  timestep: float = 0.5, scale: float = 1.0) -> Tensor:
        _, _, H, W = img0.shape
        ph = ((H - 1) // 64 + 1) * 64
        pw = ((W - 1) // 64 + 1) * 64
        pad_h = ph - H
        pad_w = pw - W

        if pad_h > 0 or pad_w > 0:
            img0 = img0.pad((0, pad_w, 0, pad_h), mode='constant', value=0)
            img1 = img1.pad((0, pad_w, 0, pad_h), mode='constant', value=0)
            state = state.pad((0, pad_w, 0, pad_h), mode='constant', value=0)

        jit_key = (ph, pw, float(scale), float(timestep), 'fine')
        if jit_key not in self._jit_cache:
            self._jit_cache[jit_key] = TinyJit(
                lambda t0, t1, st: self._forward_fine(
                    t0, t1, st, timestep=timestep, scale=scale)
            )
        merged = self._jit_cache[jit_key](img0, img1, state)

        if pad_h > 0 or pad_w > 0:
            merged = merged[:, :, :H, :W]

        return merged

    def tile_process(
        self,
        img0: Tensor,
        img1: Tensor,
        timestep: float = 0.5,
        scale: float = 1.0,
        tile: int = 128,
        tile_pad: int = 10,
        verbose: bool = False,
    ) -> Tensor:
        """Hybrid tiled inference.

        Coarse blocks (0, 1) run on the whole image so the coarse flow field is
        globally consistent; only fine blocks (2, 3, 4) run per tile. This removes
        tile-boundary seams: their receptive fields fit inside tile_pad.
        """
        assert tile >= 64 and tile % 64 == 0, f"tile size ({tile}) must be a multiple of 64 (e.g. 64, 128, 256, 512, 1024)"
        if tile_pad <= 0:
            tile_pad = max(1, tile // 8)
        base = tile - 2 * tile_pad
        assert base > 0, f"tile size ({tile}) must be greater than 2 * tile_pad ({2 * tile_pad})"

        B, C, height, width = img0.shape

        tiles_x = math.ceil(width / base)
        tiles_y = math.ceil(height / base)

        pad_left = tile_pad
        pad_top = tile_pad
        pad_right = (tiles_x - 1) * base + tile - (width + pad_left)
        pad_bottom = (tiles_y - 1) * base + tile - (height + pad_top)

        img0_np = img0.numpy()
        img1_np = img1.numpy()

        # reflect-pad whole image on host so every tile window is interior ->
        # zero host<->device roundtrips inside the tile loop
        img0_padded = np.pad(
            img0_np,
            ((0, 0), (0, 0), (pad_top, max(0, pad_bottom)), (pad_left, max(0, pad_right))),
            mode="reflect",
        )
        img1_padded = np.pad(
            img1_np,
            ((0, 0), (0, 0), (pad_top, max(0, pad_bottom)), (pad_left, max(0, pad_right))),
            mode="reflect",
        )

        img0_t = Tensor(img0_padded)
        img1_t = Tensor(img1_padded)

        # global coarse pass on the padded image -> seam-free coarse flow, and the
        # state tensors already carry the reflect context edge tiles need
        state = self._jit_coarse(img0_t, img1_t, timestep=timestep, scale=scale)
        state_all = Tensor.cat(*state, dim=1)

        out = np.zeros((B, C, height, width), dtype=np.float32)

        # NOTE: accumulate in numpy, not Tensor slice .assign() — assign on a
        # sliced view writes a temp buffer, leaving the parent stale and
        # producing seams at tile boundaries.

        n_tiles = tiles_x * tiles_y
        tile_times: np.ndarray | None = np.zeros(n_tiles, dtype=np.float64) if verbose else None
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                t_start = time.perf_counter()
                sx = tx * base
                sy = ty * base

                tile0_t = img0_t[:, :, sy : sy + tile, sx : sx + tile].clone()
                tile1_t = img1_t[:, :, sy : sy + tile, sx : sx + tile].clone()
                state_t = state_all[:, :, sy : sy + tile, sx : sx + tile].clone()

                y_tile = self._jit_fine(tile0_t, tile1_t, state_t,
                                        timestep=timestep, scale=scale).numpy()

                if verbose and tile_times is not None:
                    tile_times[ty * tiles_x + tx] = time.perf_counter() - t_start
                    print(f"    tile {ty * tiles_x + tx + 1}/{n_tiles} (x={tx}, y={ty}): {tile_times[ty * tiles_x + tx]:.3f}s")

                in_x = tx * base
                in_y = ty * base
                w_valid = min(base, width - in_x)
                h_valid = min(base, height - in_y)

                oy_t = tile_pad
                ox_t = tile_pad

                out[:, :, in_y : in_y + h_valid, in_x : in_x + w_valid] = (
                    y_tile[:, :, oy_t : oy_t + h_valid, ox_t : ox_t + w_valid]
                )

        if verbose and tile_times is not None:
            print(f"  Tiles: {n_tiles} total, avg {tile_times.mean()*1000:.1f}ms, "
                  f"min {tile_times.min()*1000:.1f}ms, max {tile_times.max()*1000:.1f}ms, "
                  f"total {tile_times.sum():.3f}s")

        return Tensor(out)

    def _forward_tta(
        self,
        img0: Tensor,
        img1: Tensor,
        timestep: float = 0.5,
        scale: float = 1.0,
        spatial: bool = False,
        temporal: bool = False,
        verbose: bool = False,
    ) -> Tensor:
        """TTA implementation matching rife-ncnn-vulkan:
        - Spatial TTA (8 augmentations) with intermediate flow/mask averaging across blocks
        - Temporal TTA (forward + reverse) with intermediate flow/mask merging across blocks
        """
        B, _, H, W = img0.shape

        # Pre-pad to 64-multiple so all augmentations produce valid dims
        # and model never sees zero-padded borders
        ph = ((H - 1) // 64 + 1) * 64
        pw = ((W - 1) // 64 + 1) * 64
        pad_h = ph - H
        pad_w = pw - W

        if pad_h > 0 or pad_w > 0:
            pad_mode = 'reflect' if (pad_h < H and pad_w < W) else 'constant'
            img0 = img0.pad((0, pad_w, 0, pad_h), mode=pad_mode)
            img1 = img1.pad((0, pad_w, 0, pad_h), mode=pad_mode)

        num_spatial = 8 if spatial else 1
        scale_list = [int(16 / scale), int(8 / scale), int(4 / scale), int(2 / scale), int(1 / scale)]
        blocks = [self.flownet.block0, self.flownet.block1, self.flownet.block2, self.flownet.block3, self.flownet.block4]

        # Prepare augmentations
        aug_img0 = [_tta_augment_img(img0, ti).realize() for ti in range(num_spatial)]
        aug_img1 = [_tta_augment_img(img1, ti).realize() for ti in range(num_spatial)]

        # Prepare timesteps
        aug_timestep = [
            Tensor([timestep]).view(1, 1, 1, 1).expand(B, 1, aug_img0[ti].shape[2], aug_img0[ti].shape[3]).realize()
            for ti in range(num_spatial)
        ]
        if temporal:
            aug_timestep_rev = [
                Tensor([1.0 - timestep]).view(1, 1, 1, 1).expand(B, 1, aug_img0[ti].shape[2], aug_img0[ti].shape[3]).realize()
                for ti in range(num_spatial)
            ]

        # Pre-encode features for heads
        aug_f0 = [self.flownet.encode(aug_img0[ti][:, :3]).realize() for ti in range(num_spatial)]
        aug_f1 = [self.flownet.encode(aug_img1[ti][:, :3]).realize() for ti in range(num_spatial)]

        flow: list[Tensor] = [Tensor()] * num_spatial
        mask: list[Tensor] = [Tensor()] * num_spatial
        feat: list[Tensor] = [Tensor()] * num_spatial

        if temporal:
            rev_flow: list[Tensor] = [Tensor()] * num_spatial
            rev_mask: list[Tensor] = [Tensor()] * num_spatial
            rev_feat: list[Tensor] = [Tensor()] * num_spatial

        # Run 5 blocks
        for i in range(5):
            if verbose:
                print(f"  TTA block {i + 1}/5")

            # Execute block i for each spatial augmentation
            for ti in range(num_spatial):
                if i == 0:
                    inp = aug_img0[ti][:, :3].cat(aug_img1[ti][:, :3], dim=1).cat(aug_f0[ti], dim=1).cat(aug_f1[ti], dim=1).cat(aug_timestep[ti], dim=1)
                    f, m, ft = blocks[0](inp, None, scale=scale_list[0])
                    flow[ti], mask[ti], feat[ti] = f.realize(), m.realize(), ft.realize()

                    if temporal:
                        rev_inp = aug_img1[ti][:, :3].cat(aug_img0[ti][:, :3], dim=1).cat(aug_f1[ti], dim=1).cat(aug_f0[ti], dim=1).cat(aug_timestep_rev[ti], dim=1)
                        rf, rm, rft = blocks[0](rev_inp, None, scale=scale_list[0])
                        rev_flow[ti], rev_mask[ti], rev_feat[ti] = rf.realize(), rm.realize(), rft.realize()
                else:
                    wf0 = warp(aug_f0[ti], flow[ti][:, :2])
                    wf1 = warp(aug_f1[ti], flow[ti][:, 2:4])
                    warped_img0 = warp(aug_img0[ti], flow[ti][:, :2])
                    warped_img1 = warp(aug_img1[ti], flow[ti][:, 2:4])
                    inp = warped_img0[:, :3].cat(warped_img1[:, :3], dim=1).cat(wf0, dim=1).cat(wf1, dim=1).cat(aug_timestep[ti], dim=1).cat(mask[ti], dim=1).cat(feat[ti], dim=1)
                    fd, m, ft = blocks[i](inp, flow[ti], scale=scale_list[i])
                    flow[ti] = (flow[ti] + fd).realize()
                    mask[ti] = m.realize()
                    feat[ti] = ft.realize()

                    if temporal:
                        rev_wf0 = warp(aug_f1[ti], rev_flow[ti][:, :2])
                        rev_wf1 = warp(aug_f0[ti], rev_flow[ti][:, 2:4])
                        rev_warped_img0 = warp(aug_img1[ti], rev_flow[ti][:, :2])
                        rev_warped_img1 = warp(aug_img0[ti], rev_flow[ti][:, 2:4])
                        rev_inp = rev_warped_img0[:, :3].cat(rev_warped_img1[:, :3], dim=1).cat(rev_wf0, dim=1).cat(rev_wf1, dim=1).cat(aug_timestep_rev[ti], dim=1).cat(rev_mask[ti], dim=1).cat(rev_feat[ti], dim=1)
                        rfd, rm, rft = blocks[i](rev_inp, rev_flow[ti], scale=scale_list[i])
                        rev_flow[ti] = (rev_flow[ti] + rfd).realize()
                        rev_mask[ti] = rm.realize()
                        rev_feat[ti] = rft.realize()

            # For intermediate blocks (0, 1, 2, 3): average flow and mask
            if i < 4:
                # 1. Temporal averaging
                if temporal:
                    for ti in range(num_spatial):
                        fm_fwd = flow[ti].cat(mask[ti], dim=1)
                        fm_rev = rev_flow[ti].cat(rev_mask[ti], dim=1)
                        fwd_merged, rev_merged = _temporal_average_flow_mask(fm_fwd, fm_rev)
                        flow[ti] = fwd_merged[:, :4].realize()
                        mask[ti] = fwd_merged[:, 4:5].realize()
                        rev_flow[ti] = rev_merged[:, :4].realize()
                        rev_mask[ti] = rev_merged[:, 4:5].realize()

                # 2. Spatial averaging across 8 augmentations
                if spatial:
                    fm_fwd_list = [_tta_deaugment_flow_mask(flow[ti].cat(mask[ti], dim=1).realize(), ti).realize() for ti in range(8)]
                    fm_fwd_avg = _realized_average(fm_fwd_list)
                    for ti in range(8):
                        fm_aug = _tta_augment_flow_mask(fm_fwd_avg, ti).realize()
                        flow[ti] = fm_aug[:, :4].realize()
                        mask[ti] = fm_aug[:, 4:5].realize()

                    if temporal:
                        fm_rev_list = [_tta_deaugment_flow_mask(rev_flow[ti].cat(rev_mask[ti], dim=1).realize(), ti).realize() for ti in range(8)]
                        fm_rev_avg = _realized_average(fm_rev_list)
                        for ti in range(8):
                            fm_rev_aug = _tta_augment_flow_mask(fm_rev_avg, ti).realize()
                            rev_flow[ti] = fm_rev_aug[:, :4].realize()
                            rev_mask[ti] = fm_rev_aug[:, 4:5].realize()

        # After block 4 (final flow & merge):
        out_imgs = []
        for ti in range(num_spatial):
            warped_img0 = warp(aug_img0[ti], flow[ti][:, :2])
            warped_img1 = warp(aug_img1[ti], flow[ti][:, 2:4])
            m_sig = mask[ti].sigmoid()
            out_ti = (warped_img0 * m_sig + warped_img1 * (1.0 - m_sig)).realize()

            if temporal:
                rev_warped_img0 = warp(aug_img1[ti], rev_flow[ti][:, :2])
                rev_warped_img1 = warp(aug_img0[ti], rev_flow[ti][:, 2:4])
                rev_m_sig = rev_mask[ti].sigmoid()
                out_rev_ti = rev_warped_img0 * rev_m_sig + rev_warped_img1 * (1.0 - rev_m_sig)
                out_ti = ((out_ti + out_rev_ti) * 0.5).realize()

            out_imgs.append(out_ti)

        if spatial:
            deaug_outs = [_tta_deaugment_img(out_imgs[ti], ti).realize() for ti in range(8)]
            merged_avg = _realized_average(deaug_outs)
        else:
            merged_avg = out_imgs[0]

        # Crop back to original size
        return merged_avg[:, :, :H, :W].realize()

    def inference(
        self,
        img0: Tensor,
        img1: Tensor,
        timestep: float = 0.5,
        scale: float = 1.0,
        tile: int = 0,
        tile_pad: int = 10,
        verbose: bool = False,
        tta: bool = False,
        tta_temporal: bool = False,
    ) -> Tensor:
        if tta or tta_temporal:
            return self._forward_tta(img0, img1, timestep=timestep, scale=scale, spatial=tta, temporal=tta_temporal, verbose=verbose)
        if tile > 0:
            return self.tile_process(img0, img1, timestep=timestep, scale=scale, tile=tile, tile_pad=tile_pad, verbose=verbose)
        return self._jit_forward(img0, img1, timestep=timestep, scale=scale)


def load_safetensors_weights(tinygrad_model, safetensors_path):
    from safetensors import safe_open

    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Weights file not found: {safetensors_path}")

    weights = {}
    with safe_open(safetensors_path, framework="numpy") as f:
        for k in f.keys():  # noqa: SIM118
            weights[k] = f.get_tensor(k)

    def assign_conv2d(tg_conv, np_weight, np_bias=None):
        tg_conv.weight.assign(Tensor(np_weight))
        if np_bias is not None:
            tg_conv.bias.assign(Tensor(np_bias))

    def assign_convtranspose2d(tg_conv, np_weight, np_bias=None):
        tg_conv.weight.assign(Tensor(np_weight))
        if np_bias is not None:
            tg_conv.bias.assign(Tensor(np_bias))

    def load_block(tg_block, prefix, c):
        assign_conv2d(tg_block.conv0[0], weights[f'{prefix}.conv0.0.0.weight'], weights[f'{prefix}.conv0.0.0.bias'])
        assign_conv2d(tg_block.conv0[2], weights[f'{prefix}.conv0.1.0.weight'], weights[f'{prefix}.conv0.1.0.bias'])

        for i in range(8):
            beta = weights[f'{prefix}.convblock.{i}.beta'].squeeze()
            tg_block.convblock[i].beta.assign(Tensor(beta.reshape(1, c, 1, 1)))
            assign_conv2d(tg_block.convblock[i].conv, weights[f'{prefix}.convblock.{i}.conv.weight'], weights[f'{prefix}.convblock.{i}.conv.bias'])

        assign_convtranspose2d(tg_block.lastconv[0], weights[f'{prefix}.lastconv.0.weight'], weights[f'{prefix}.lastconv.0.bias'])

    load_block(tinygrad_model.flownet.block0, 'module.block0', 192)
    load_block(tinygrad_model.flownet.block1, 'module.block1', 128)
    load_block(tinygrad_model.flownet.block2, 'module.block2', 96)
    load_block(tinygrad_model.flownet.block3, 'module.block3', 64)
    load_block(tinygrad_model.flownet.block4, 'module.block4', 32)

    tg_encode = tinygrad_model.flownet.encode
    assign_conv2d(tg_encode.cnn0, weights['module.encode.cnn0.weight'], weights['module.encode.cnn0.bias'])
    assign_conv2d(tg_encode.cnn1, weights['module.encode.cnn1.weight'], weights['module.encode.cnn1.bias'])
    assign_conv2d(tg_encode.cnn2, weights['module.encode.cnn2.weight'], weights['module.encode.cnn2.bias'])
    assign_convtranspose2d(tg_encode.cnn3, weights['module.encode.cnn3.weight'], weights['module.encode.cnn3.bias'])

    print("Weights loaded successfully!")


