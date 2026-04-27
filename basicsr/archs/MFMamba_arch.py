# ------------------------------------------------------------------------
# MFMamba Architecture for BasicSR
# Adapted from original net.py / net_common.py
# Place this file in:  basicsr/archs/mfmamba_arch.py
# Register entry:      basicsr/archs/__init__.py  (auto-scanned by BasicSR)
# ------------------------------------------------------------------------

import os
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.mambaIR import MambaIR1 as MambaNet

warnings.filterwarnings('ignore')

# =========================================================================
# Primitives  (previously net_common.py)
# =========================================================================

class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def default_conv(ch_in, ch_out, kernel_size, bias=True):
    return nn.Conv2d(
        ch_in, ch_out, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class Default_Conv(nn.Module):
    def __init__(self, ch_in, ch_out, k_size=(3, 3), stride=1,
                 padding=(1, 1), bias=False, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, k_size, stride=stride,
                              padding=padding, bias=bias, groups=groups)

    def forward(self, x):
        return self.conv(x)


class ConvUpsampler(nn.Module):
    """2× upsampler via PixelShuffle."""
    def __init__(self, ch_in, ch_out, bias=False, activation=None):
        super().__init__()
        self.conv1 = Default_Conv(ch_in=ch_in, ch_out=ch_out * 4,
                                  k_size=3, bias=bias)
        self.ps   = nn.PixelShuffle(2)
        self.act  = activation if activation is not None else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.ps(self.conv1(x)))


class involution(nn.Module):
    """Involution operator."""
    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size   = kernel_size
        self.stride        = stride
        self.channels      = channels
        reduction_ratio    = 4
        self.group_channels = 16
        self.groups        = self.channels // self.group_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU())
        self.conv2 = nn.Conv2d(channels // reduction_ratio,
                               kernel_size ** 2 * self.groups, 1)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape  # 以 weight 输出的 h,w 为基准

        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)

        out = self.unfold(x)  # [b, group_channels * groups * k^2, L]
        # L 在边界尺寸时可能 != h*w，强制对齐到 h*w
        out = out[..., :h * w]
        out = out.view(b, self.groups, self.group_channels,
                       self.kernel_size ** 2, h, w)

        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out


# =========================================================================
# Building blocks  (previously net.py)
# =========================================================================

class MDCB(nn.Module):
    """Multi-scale Dense Convolution Block with 3×3 and 5×5 branches."""
    def __init__(self, ch_in, ch_out, bias=True,
                 activation=None):
        super().__init__()
        act = activation if activation is not None else nn.ReLU(inplace=True)

        self.conv_3_1 = default_conv(ch_in, ch_in, 3, bias=bias)
        self.conv_3_2 = default_conv(ch_out, ch_out, 3, bias=bias)
        self.conv_5_1 = default_conv(ch_in, ch_in, 5, bias=bias)
        self.conv_5_2 = default_conv(ch_out, ch_out, 5, bias=bias)

        self.confusion_3      = nn.Conv2d(ch_in * 3, ch_out, 1, bias=True)
        self.confusion_5      = nn.Conv2d(ch_in * 3, ch_out, 1, bias=True)
        self.confusion_bottle = nn.Conv2d(ch_in * 3 + ch_out * 2, ch_out, 1, bias=True)
        self.activation = act

    def forward(self, x):
        o3 = self.activation(self.conv_3_1(x)) + x
        o5 = self.activation(self.conv_5_1(x)) + x
        cat1 = torch.cat([x, o3, o5], dim=1)

        i2_3 = self.confusion_3(cat1)
        i2_5 = self.confusion_5(cat1)
        o3_2 = self.activation(self.conv_3_2(i2_3))
        o5_2 = self.activation(self.conv_5_2(i2_5))

        cat2 = torch.cat([x, o3, o5, o3_2, o5_2], dim=1)
        return self.confusion_bottle(cat2)


class IMUB_Head(nn.Module):
    """Multi-scale image feature extractor head."""
    def __init__(self, ch_in, ch_out, activation=None,
                 bias=False, down_times=2, fe_num=16):
        super().__init__()
        act = activation if activation is not None else nn.ReLU(inplace=True)
        self.down_times = down_times

        def _block1():
            return nn.Sequential(
                nn.Conv2d(ch_in, fe_num, 1, bias=bias), act)

        def _block2():
            return nn.Sequential(
                nn.Conv2d(ch_in, fe_num, 1, bias=bias), act)

        def _block3():
            return nn.Sequential(
                nn.Conv2d(ch_in, fe_num, 1, bias=bias), act)

        # branch-1 (depth 1)
        self.conv1_1 = nn.Sequential(nn.Conv2d(ch_in, fe_num, 1, bias=bias), act)
        self.conv1_2 = nn.Sequential(nn.Conv2d(fe_num, fe_num, 3, padding=1, bias=bias), act)
        # branch-2 (depth 2)
        self.conv2_1 = nn.Sequential(nn.Conv2d(ch_in, fe_num, 1, bias=bias), act)
        self.conv2_2 = nn.Sequential(nn.Conv2d(fe_num, fe_num, 3, padding=1, bias=bias), act)
        self.conv2_3 = nn.Sequential(nn.Conv2d(fe_num, fe_num, 3, padding=1, bias=bias), act)
        # branch-3 (depth 3)
        self.conv3_1 = nn.Sequential(nn.Conv2d(ch_in, fe_num, 1, bias=bias), act)
        self.conv3_2 = nn.Sequential(nn.Conv2d(fe_num, fe_num, 3, padding=1, bias=bias), act)
        self.conv3_3 = nn.Sequential(nn.Conv2d(fe_num, fe_num, 3, padding=1, bias=bias), act)
        self.conv3_4 = nn.Sequential(nn.Conv2d(fe_num, fe_num, 3, padding=1, bias=bias), act)

        self.conv_end = nn.Sequential(
            nn.Conv2d(fe_num * 3, ch_out, 1, bias=bias), act)
        self.conv_sum = nn.Sequential(
            nn.Conv2d(ch_out * (down_times + 1), ch_out, 1, bias=bias), act)

    def forward(self, x):
        b, _, h, w = x.size()
        inputs = [x] + [
            F.interpolate(x, size=(x.shape[2] // (2 * i + 2),
                                   x.shape[3] // (2 * i + 2)),
                          mode='bilinear', align_corners=False)
            for i in range(self.down_times)
        ]

        feats = []
        for item in inputs:
            x1 = self.conv1_2(self.conv1_1(item)) + self.conv1_1(item)
            x2 = self.conv2_1(item)
            x2 = self.conv2_2(x2) + x2
            x2 = self.conv2_3(x2) + x2
            x3 = self.conv3_1(item)
            x3 = self.conv3_2(x3) + x3
            x3 = self.conv3_3(x3) + x3
            x3 = self.conv3_4(x3) + x3
            feats.append(self.conv_end(torch.cat([x1, x2, x3], dim=1)))

        result = [
            f if (f.shape[2] == h and f.shape[3] == w)
            else F.interpolate(f, size=(h, w), mode='bilinear', align_corners=False)
            for f in feats
        ]

        out = result[0]
        for r in result[1:]:
            out = torch.cat([out, r], dim=1)
        return self.conv_sum(out)


class IDB(nn.Module):
    """Involution Down Block."""
    def __init__(self, in_planes, planes, bias, activation):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_planes, planes, 3, padding=1, bias=bias),
            activation,
            involution(channels=planes, kernel_size=3, stride=2),
            activation,
            nn.Conv2d(planes, planes, 1, bias=bias),
            activation,
        )

    def forward(self, x):
        return self.cnn(x)


class IMUB(nn.Module):
    """Involution Multi-scale Up Block."""
    def __init__(self, in_planes, planes, bias, activation):
        super().__init__()
        self.cnn = nn.Sequential(
            IMUB_Head(ch_in=in_planes, ch_out=planes,
                      bias=bias, activation=activation),
            ConvUpsampler(ch_in=planes, ch_out=planes,
                          activation=activation, bias=bias),
            activation,
        )

    def forward(self, x):
        return self.cnn(x)


class No_Multi_SEAttention(nn.Module):
    """Lightweight SE placeholder (1×1 conv)."""
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)

    def forward(self, x):
        return self.conv1(x)


# =========================================================================
# Core network: UNet_Plus_Residual  →  MFMamba
# =========================================================================

@ARCH_REGISTRY.register()
class MFMamba(nn.Module):
    """
    MFMamba: Multi-scale Feature Mamba network for image restoration/SR.

    Args:
        scale      (int):  Upscale factor. 1 = restoration, 2/4/8 = SR.
        depth      (int):  Number of U-Net+ stages.
        grow_rate  (int):  Channel growth rate per stage.
        fe_num     (int):  Base feature channels.
        multi_out  (bool): Return intermediate outputs (training supervision).

    Typical YAML entry::

        network_g:
          type: MFMamba
          scale: 2
          depth: 4
          grow_rate: 32
          fe_num: 128
          multi_out: false
    """

    def __init__(self,
                 scale: int = 2,
                 depth: int = 4,
                 grow_rate: int = 32,
                 fe_num: int = 128,
                 multi_out: bool = False):
        super().__init__()

        self.depth     = depth
        self.fe_num    = fe_num
        self.grow_rate = grow_rate
        self.multi_out = multi_out
        self.scale     = scale

        self.bias       = True
        self.activation = Mish()

        # ---- head --------------------------------------------------------
        self.root = self._make_head()

        # ---- U-Net+ body -------------------------------------------------
        self.layers = self._make_layers(
            depth=depth, in_planes=fe_num,
            down_block=IDB, up_block=IMUB,
            se_block=No_Multi_SEAttention)

        # ---- output projections ------------------------------------------
        n_convs = depth if multi_out else 1
        # always build `depth` projections so indexing is consistent
        # self.end_convs = self._make_end_convs(depth, Default_Conv)
        if self.multi_out:
            self.end_convs = self._make_end_convs(self.depth,Default_Conv)
        else:
            self.end_convs = self._make_end_convs(1,Default_Conv)

        # ---- weight init -------------------------------------------------
        self._init_weights()

    # ------------------------------------------------------------------
    # Head
    # ------------------------------------------------------------------

    def _make_head(self):
        bias = self.bias
        act  = self.activation
        fe   = self.fe_num

        if self.scale == 1:
            head = nn.Sequential(
                MDCB(3, 32, bias=bias, activation=act),
                MDCB(32, fe, bias=bias, activation=act),
            )
        elif self.scale == 2:
            # Import MambaIR lazily so the file works even without it installed
            # try:
            #     from basicsr.archs.mambair_arch import MambaIR as MambaNet
            # except ImportError:
            #     raise ImportError(
            #         "MambaIR not found. For scale=2, install / register "
            #         "MambaIR as basicsr/archs/mambaIR_arch.py first.")
            head = nn.Sequential(
                MDCB(3, 32, bias=bias, activation=act),
                MDCB(32, fe, bias=bias, activation=act),
                ConvUpsampler(fe, fe, activation=act, bias=bias),
                # MambaNet(
                #     img_size=64, patch_size=1, in_chans=fe,
                #     embed_dim=180, depths=(6, 6, 6, 6, 6, 6),
                #     # num_heads=(6, 6, 6, 6, 6, 6),
                #     mlp_ratio=2., drop_rate=0.,
                #     norm_layer=nn.LayerNorm, patch_norm=True,
                #     use_checkpoint=False, upscale=2, img_range=1.,
                #     upsampler='pixelshuffle', resi_connection='1conv'),
            )
        elif self.scale == 4:
            head = nn.Sequential(
                MDCB(3, 32, bias=bias, activation=act),
                MDCB(32, fe, bias=bias, activation=act),
                ConvUpsampler(fe, fe, activation=act, bias=bias),
                ConvUpsampler(fe, fe, activation=act, bias=bias),
            )
        elif self.scale == 8:
            head = nn.Sequential(
                MDCB(3, 32, bias=bias, activation=act),
                MDCB(32, fe, bias=bias, activation=act),
                ConvUpsampler(fe, fe, activation=act, bias=bias),
                ConvUpsampler(fe, fe, activation=act, bias=bias),
                ConvUpsampler(fe, fe, activation=act, bias=bias),
            )
        else:
            raise ValueError(f"Unsupported scale={self.scale}. Choose 1/2/4/8.")
        return head

    # ------------------------------------------------------------------
    # U-Net+ body
    # ------------------------------------------------------------------

    def _make_layers(self, depth, in_planes, down_block, up_block, se_block):
        layers = []
        planes = in_planes + self.grow_rate

        for i in range(depth):
            layer_list = []
            if i == 0:
                lp = planes
                layer_list.append(down_block(lp - self.grow_rate, lp,
                                             self.bias, self.activation))
                layer_list.append(up_block(lp, lp - self.grow_rate,
                                           self.bias, self.activation))
                if se_block is not None:
                    layer_list.append(se_block(lp - self.grow_rate))
                layers.append(nn.Sequential(*layer_list))
                planes += self.grow_rate
            else:
                lp = planes
                layer_list_i = []
                for j in range(i + 1):
                    if j == 0:
                        layer_list_i.append(
                            down_block(lp - self.grow_rate, lp,
                                       self.bias, self.activation))
                    layer_list_i.append(
                        up_block(lp, lp - self.grow_rate,
                                 self.bias, self.activation))
                    if se_block is not None:
                        layer_list_i.append(se_block(lp - self.grow_rate))
                    lp -= self.grow_rate
                layers.append(nn.Sequential(*layer_list_i))
                planes += self.grow_rate

        return nn.Sequential(*layers)

    def _make_end_convs(self, depth, block):
        conv_list = [block(ch_in=self.fe_num, ch_out=3) for _ in range(depth)]
        return nn.Sequential(*conv_list)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inp):
        features = []
        x = self.root(inp)
        features.append([x])

        for i in range(self.depth):
            if i == 0:
                y_11 = self.layers[0][0](x)
                y_12_main = self.layers[0][1](y_11)
                y_12_skip = self.layers[0][2](x)
                # 对齐尺寸
                if y_12_main.shape != x.shape:
                    y_12_main = F.interpolate(y_12_main, size=x.shape[2:], mode='bilinear', align_corners=False)
                if y_12_skip.shape != x.shape:
                    y_12_skip = F.interpolate(y_12_skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                y_12 = y_12_main + y_12_skip + x
                features.append([y_11, y_12])
            else:
                features_layer = []
                layer_step = 0
                feature_step = 0
                for j in range(i + 1):
                    if j == 0:
                        y_n1 = self.layers[i][0](features[i][0])
                        y_n2_main = self.layers[i][1](y_n1)
                        y_n2_skip = self.layers[i][2](features[i][0])
                        ref = features[i][0]
                        # 对齐尺寸
                        if y_n2_main.shape != ref.shape:
                            y_n2_main = F.interpolate(y_n2_main, size=ref.shape[2:], mode='bilinear',
                                                      align_corners=False)
                        if y_n2_skip.shape != ref.shape:
                            y_n2_skip = F.interpolate(y_n2_skip, size=ref.shape[2:], mode='bilinear',
                                                      align_corners=False)
                        y_n2 = y_n2_main + y_n2_skip + ref
                        features_layer.append(y_n1)
                        features_layer.append(y_n2)
                        layer_step = 3
                        feature_step = 1
                    else:
                        ref = features[i][feature_step]
                        y_nm_main = self.layers[i][layer_step](features_layer[-1])
                        y_nm_skip = self.layers[i][layer_step + 1](ref)
                        # 对齐尺寸
                        if y_nm_main.shape != ref.shape:
                            y_nm_main = F.interpolate(y_nm_main, size=ref.shape[2:], mode='bilinear',
                                                      align_corners=False)
                        if y_nm_skip.shape != ref.shape:
                            y_nm_skip = F.interpolate(y_nm_skip, size=ref.shape[2:], mode='bilinear',
                                                      align_corners=False)
                        y_nm = y_nm_main + y_nm_skip + ref
                        features_layer.append(y_nm)
                        layer_step += 2
                        feature_step += 1
                features.append(features_layer)

        result = [stage[-1] for stage in features]

        out = []
        if self.multi_out:
            for i in range(1, len(result)):
                out.append(self.activation(self.end_convs[i - 1](result[i])))
        else:
            out.append(self.activation(self.end_convs[0](result[-1])))

        if not self.multi_out:
            return out[0]
        return out

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    m.bias.data.zero_()