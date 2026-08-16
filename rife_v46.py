import math
from tinygrad import Tensor, TinyJit
from tinygrad.nn import Conv2d, ConvTranspose2d
import numpy as np


def warp(tenInput: Tensor, tenFlow: Tensor) -> Tensor:
    B, C, H, W = tenInput.shape
    
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
    
    x = (grid[..., 0] + 1) * (W - 1) / 2
    y = (grid[..., 1] + 1) * (H - 1) / 2
    
    x0 = x.floor().cast('int32')
    x1 = (x0 + 1).clip(0, W - 1)
    y0 = y.floor().cast('int32')
    y1 = (y0 + 1).clip(0, H - 1)
    x0 = x0.clip(0, W - 1)
    y0 = y0.clip(0, H - 1)
    
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
        
    def __call__(self, x: Tensor, timestep=0.5, scale_list=[8, 4, 2, 1]):
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
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        feat = None
        
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        
        for i in range(5):
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

    def _jit_forward(self, img0: Tensor, img1: Tensor, timestep: float = 0.5, scale: float = 1.0) -> Tensor:
        B, C, H, W = img0.shape

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
    
    def tile_process(
        self,
        img0: Tensor,
        img1: Tensor,
        timestep: float = 0.5,
        scale: float = 1.0,
        tile: int = 128,
        tile_pad: int = 10,
    ) -> Tensor:
        """Tiled inference with constant model input shape (1, 3, tile, tile) for TinyJit."""
        base = tile - 2 * tile_pad
        assert base > 0, f"tile size ({tile}) must be greater than 2 * tile_pad ({2 * tile_pad})"

        B, C, height, width = img0.shape
        out = np.zeros((B, C, height, width), dtype=np.float32)

        tiles_x = math.ceil(width / base)
        tiles_y = math.ceil(height / base)

        pad_left = tile_pad
        pad_top = tile_pad
        pad_right = (tiles_x - 1) * base + tile - (width + pad_left)
        pad_bottom = (tiles_y - 1) * base + tile - (height + pad_top)

        img0_np = img0.numpy()
        img1_np = img1.numpy()

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

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                sx = tx * base
                sy = ty * base
                tile0_np = img0_padded[:, :, sy : sy + tile, sx : sx + tile]
                tile1_np = img1_padded[:, :, sy : sy + tile, sx : sx + tile]

                tile0_t = Tensor(tile0_np)
                tile1_t = Tensor(tile1_np)

                y_tile = self._jit_forward(tile0_t, tile1_t, timestep=timestep, scale=scale).numpy()

                in_x = tx * base
                in_y = ty * base
                w_valid = min(base, width - in_x)
                h_valid = min(base, height - in_y)

                oy_t = tile_pad
                ox_t = tile_pad

                out[:, :, in_y : in_y + h_valid, in_x : in_x + w_valid] = (
                    y_tile[:, :, oy_t : oy_t + h_valid, ox_t : ox_t + w_valid]
                )

        return Tensor(out)

    def inference(
        self,
        img0: Tensor,
        img1: Tensor,
        timestep: float = 0.5,
        scale: float = 1.0,
        tile: int = 0,
        tile_pad: int = 10,
    ) -> Tensor:
        if tile > 0:
            return self.tile_process(img0, img1, timestep=timestep, scale=scale, tile=tile, tile_pad=tile_pad)
        return self._jit_forward(img0, img1, timestep=timestep, scale=scale)


def load_safetensors_weights(tinygrad_model, safetensors_path):
    from safetensors import safe_open

    weights = {}
    with safe_open(safetensors_path, framework="numpy") as f:
        for k in f.keys():
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