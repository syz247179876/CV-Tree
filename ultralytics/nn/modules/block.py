# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""
import math
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t

from math import log
from timm.layers import DropPath
from timm.models.senet import SEModule

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, ConvOD, PartialConv, PConv, CondConv, RepConv1x1, \
    LightConv2
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C2fOD', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'BottleneckOD', 'Proto', 'RepC3', 'CABlock', 'C2fFaster',
           'SEBlock', 'SKBlock', 'C2fBoT', 'AFPNC2f', 'AFPNPConv', 'FasterBlocks', 'C2fCondConv', 'CPCA', 'ECA',
           'SimAM', 'EMA', 'SPPFAvgAttn', 'RepStem', 'RepStemDWC', 'BiCrossFPN', 'C2fGhostV2', 'GhostConvV2',
           'LightBlocks', 'SPPCA')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class SPPFAvgAttn(nn.Module):

    def __init__(self, c1, c2, k=5, mlp_ratio: int = 2):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        针对小目标检测，减少卷积核大小，更侧重于局部信息的提取, 因此k初始设定为3, k = (3, 5, 7)
        避免产生过大的感受野, 如果感受野过大, 导致特征图中某一块区域内非目标特征占比过大, 会影响检测结果
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.max = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.avg = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.rep_dwc = RepConv(c_ * 4, c_ * 4, k=3, g=c_ * 4, act=True)
        self.attn = ECA(c_ * 4)
        self.cv2 = Conv(c_ * 4, c2, k=1, act=True)
        self.ffn = nn.Sequential(
            nn.Conv2d(c2, c2 * mlp_ratio, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.Conv2d(c2 * mlp_ratio, c2, 1, 1, 0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        max_y1 = self.max(x)
        avg_y1 = self.avg(x)
        max_y2 = self.max(max_y1)
        avg_y2 = self.avg(avg_y1)
        out = self.rep_dwc(torch.cat((x, max_y1 + avg_y1, max_y2 + avg_y2, self.max(max_y2) + self.avg(avg_y2)), dim=1))
        out = self.attn(out)
        out = self.cv2(out)
        # shortcut
        out = out + self.ffn(out)
        return out



class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """
    Faster Implementation of CSP Bottleneck with 2 convolutions.

    集合CSPNet + VoVNet思想, 在ELAN基础上进一步改进，将ELAN中Conv块替换成Bottleneck
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        # 分割梯度流
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        # 沿着通道维度进行拆分
        y = list(self.cv1(x).chunk(2, 1))
        # 右半部分经过一系列的Bottleneck, 将Yolo中的Conv块替换为Bottleneck残差结构
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class GhostConvV2(nn.Module):
    """
    Ghost Block V2 with DFC attention
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 1,
            stride: int = 1,
            group: int = 1,
            ratio: int = 2,
            dw_size: int = 3,
            act_layer: t.Optional[t.Union[nn.Module, t.Callable]] = None
    ):
        super(GhostConvV2, self).__init__()
        if act_layer is None:
            act_layer = nn.ReLU
        # self.avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)
        self.gate_fn = nn.Sigmoid()
        init_chans = math.ceil(out_chans / ratio)
        new_chans = init_chans * (ratio - 1)
        self.primary_conv = Conv(in_chans, init_chans, k=kernel_size, s=stride, g=group, act=act_layer(inplace=True))
        self.cheap_ops = Conv(init_chans, new_chans, k=dw_size, s=1, g=init_chans, act=nn.ReLU(inplace=True))
        self.short_conv = nn.Sequential(
            Conv(in_chans, out_chans, k=kernel_size, s=1, act=False),
            #ECA(out_chans),
            Conv(out_chans, out_chans, k=(1, 5), s=1, p=(0, 2), g=out_chans, act=False),
            Conv(out_chans, out_chans, k=(5, 1), s=1, p=(2, 0), g=out_chans, act=False),
            # ECA(out_chans),
        )
        # self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # attn = self.avgpool2d(x)
        attn = self.short_conv(x)
        attn = self.gate_fn(attn)
        # attn = self.up_sample(attn)

        x1 = self.primary_conv(x)
        x2 = self.cheap_ops(x1)
        out = torch.cat((x1, x2), dim=1)
        out = out * attn
        return out

class C2fGhostV2(C2f):
    """
    Replace ordinary conv with GhostConv v2 on Bottleneck of C2f module
    """
    def __init__(
            self,
            c1: int,
            c2: int,
            n: int = 1,
            shortcut: bool = False,
            g: int = 1,
            e: float = 0.5,
            ratio: int = 2,
            dw_size: int = 3,
    ):
        super(C2fGhostV2, self).__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            BottleneckGhostV2(self.c, self.c, shortcut, g, k=(1, 1), e=1.0, ratio=ratio, dw_size=dw_size)
            for _ in range(n)
        )

class C2fOD(C2f):
    """
    Replace ordinary conv with ODConv on Bottleneck of C2f module
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            n: int = 1,
            shortcut: bool = False,
            expert_num: int = 4,
            reduction: float = 0.0625,
            hidden_chans: int = 16,
            g: int = 1,
            e: float = 0.5
    ):
        super(C2fOD, self).__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(BottleneckOD(self.c, self.c, shortcut, g, k=(3, 3), e=1.0,
                                            expert_num=expert_num, reduction=reduction, hidden_chans=hidden_chans
                                            ) for _ in range(n))


class C2fFaster(C2f):
    """
    Replace ordinary conv with PConv on Bottleneck of C2f module
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            n: int = 1,
            shortcut: bool = False,
            n_div: int = 4,
            pconv_fw_type: str = 'split_cat',
            g: int = 1,
            e: float = 0.5
    ):
        super(C2fFaster, self).__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(BottleneckFaster(self.c, self.c, shortcut, n_div, pconv_fw_type,
                                                g, e=1.0) for _ in range(n))


# class C2fPConv(nn.Module):
#     """
#     Replace all Convs in the C2f block with PConv
#     """
#
#     def __init__(
#             self,
#             c1: int,
#             c2: int,
#             n: int = 1,
#             shortcut: bool = False,
#             n_div: int = 4,
#             g: int = 1,
#             e: float = 0.5,
#     ):
#         super(C2fPConv, self).__init__()
#         self.c = int (c1 * e)
#         self.cv1 = PConv(c1, 2 * self.c, 1, 1, n_div=2)
#         self.cv2 = PConv((2 + n) * self.c, c2, 1, 1, n_div=2)
#         # self.m = nn.ModuleList(BottleneckPConv)


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class BottleneckGhostV2(nn.Module):
    """
    Bottleneck with GhostConv V2
    """
    
    def __init__(
            self,
            c1: int,
            c2: int,
            shortcut: bool = True,
            g: int = 1,
            k: t.Tuple = (1, 1),
            e: float = 0.5,
            ratio: int = 2,
            dw_size: int = 3,
        ):
        super(BottleneckGhostV2, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = GhostConvV2(c1, c_, k[0], 1, ratio=ratio, dw_size=dw_size)
        self.cv2 = GhostConv(c_, c_, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckOD(nn.Module):
    """
    Bottleneck with ODConv
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            shortcut: bool = True,
            g: int = 1,
            k: t.Tuple = (3, 3),
            e: float = 0.5,
            reduction: float = 0.0625,
            expert_num: int = 4,
            hidden_chans: int = 16,
    ):
        super(BottleneckOD, self).__init__()
        c_ = int(c1 * e)
        self.cv1 = ConvOD(c1, c_, k[0], 1, g=g, reduction=reduction, expert_num=expert_num, hidden_chans=hidden_chans)
        self.cv2 = ConvOD(c_, c2, k[1], 1, g=g, reduction=reduction, expert_num=expert_num, hidden_chans=hidden_chans)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckFaster(nn.Module):
    """
    Bottleneck with PConv
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            shortcut: bool = True,
            n_div: int = 4,
            pconv_fw_type: str = 'split_cat',
            g: int = 1,
            e: float = 0.5,
    ):
        super(BottleneckFaster, self).__init__()
        c_ = int(c1 * e)
        self.cv1 = PartialConv(c1, n_div, forward=pconv_fw_type)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(c_, c2, 1, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class CABlock(nn.Module):
    """
    Coordinate Attention Block, which embeds positional information into channel attention.
    1.It considers spatial dimension attention and channel dimension attention, it helps model locate, identify and
    enhance more interesting objects.

    2.CA utilizes two 2-D GAP operation to respectively aggregate the input features along the vertical and horizontal
    directions into two separate direction aware feature maps. Then, encode these two feature maps separately into
    an attention tensor.

    3. Among these two feature maps(Cx1xW and CxHx1), one uses GAP to model the long-distance dependencies of
    the feature maps on a spatial dimension, while retaining position information int the other spatial dimension.
    In that case, the two feature maps, spatial information, and long-range dependencies complement each other.

    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            reduction: int = 32,
            act_layer: t.Optional[nn.Module] = None,
    ):
        super(CABlock, self).__init__()
        if act_layer is None:
            act_layer = nn.Hardswish
        self.in_chans = in_chans
        self.out_chans = out_chans
        # # (B, C, H, 1)
        # self.gap_h = nn.AdaptiveAvgPool2d((None, 1))
        # # (B, C, 1, W)
        # self.gap_w = nn.AdaptiveAvgPool2d((1, None))

        hidden_chans = max(8, in_chans // reduction)
        self.cv = Conv(in_chans, hidden_chans, act=act_layer(inplace=True))

        self.attn_h = nn.Conv2d(hidden_chans, out_chans, 1)
        self.attn_w = nn.Conv2d(hidden_chans, out_chans, 1)
        self.sigmoid = nn.Sigmoid()

        # self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # Obey uniform distribution during attention initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        identity = x
        b, c, h, w = x.size()
        # (b, c, h, 1)
        # x_h = self.gap_h(x)
        x_h = x.mean(3, keepdim=True)
        # (b, c, 1, w) -> (b, c, w, 1)
        # x_w = self.gap_w(x).permute(0, 1, 3, 2)
        x_w = x.mean(2, keepdim=True).permute(0, 1, 3, 2)
        # (b, c, h + w, 1)
        y = torch.cat((x_h, x_w), dim=2)
        y = self.cv(y)

        # split
        # x_h: (b, c, h, 1),  x_w: (b, c, w, 1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # (b, c, 1, w)
        x_w = x_w.permute(0, 1, 3, 2)

        # compute attention
        a_h = self.sigmoid(self.attn_h(x_h))
        a_w = self.sigmoid(self.attn_w(x_w))

        return identity * a_w * a_h


class SEBlock(nn.Module):
    """
    SE Block: A Channel Based Attention Mechanism.

        Traditional convolution in computation, it blends the feature relationships of the channel
    with the spatial relationships learned from the convolutional kernel, because a conv sum the
    operation results of each channel, so, using SE Block to pay attention to more important channels,
    suppress useless channels regard to current task.

    SE Block Contains three parts:
    1.Squeeze: Global Information Embedding.
        Aggregate (H, W, C) dim to (1, 1, C) dim, use GAP to generate aggregation channel,
    encode the entire spatial feature on a channel to a global feature.

    2.Excitation: Adaptive Recalibration.
        It aims to fully capture channel-wise dependencies and improve the representation of image,
    by using two liner layer, one activation inside and sigmoid or softmax to normalize,
    to produce channel-wise weights.
        Maybe like using liner layer to extract feature map to classify, but this applies at channel
    level and pay attention to channels with a large number of information.

    3.Scale: feature recalibration.
        Multiply the learned weight with the original features to obtain new features.
        SE Block can be added to Residual Block.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            reduction: int = 16,
            attention_mode: str = 'conv',
    ):
        super(SEBlock, self).__init__()
        # part 1:(H, W, C) -> (1, 1, C)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_mode = attention_mode
        # part 2, compute weight of each channel
        if attention_mode == 'conv':
            self.attn = nn.Sequential(
                Conv(in_chans, out_chans // reduction),
                nn.Conv2d(out_chans // reduction, out_chans, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.attn = nn.Sequential(
                nn.Linear(in_chans, out_chans // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(out_chans // reduction, in_chans, bias=False),
                nn.Sigmoid(),  # nn.Softmax is OK here
            )

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.size()
        y = x.mean(dim=(2, 3), keepdim=True)
        if self.attention_mode != 'conv':
            y = y.view(b, c)
            y = self.attn(y)
            y = y.view(b, c, 1, 1)
        else:
            y = self.attn(y)
        return x * y


class SKBlock(nn.Module):
    """
    SK Module combines the Inception and SE ideas, considering different channels and kernel block.
    It can split into three parts:

        1.Split: For any feature map, using different size kernel convolutions(3x3, 5x5)
        to extract new feature map. use dilation convolution (3x3, dilation=2) can
        increase regularization to avoid over-fitting caused by large convolutional kernels
        , add multi-scale information and increase receptive field.

        2.Fuse: Fusing different output(last layer feature map) of different branch and
        compute attention on channel-wise

        3.Select: 'chunk, scale, fuse'.  Focusing on different convolutional kernels for different target sizes.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            reduction: int = 16,
            attention_mode: str = 'conv',
            num: int = 2,
            act_layer: t.Optional[nn.Module] = None,
            stride: int = 1,
            groups: int = 1,
    ):
        """
            num: the number of different kernel, by the way, it means the number of different branch, using
                Inception ideas.
            reduction: Multiple of dimensionality reduction, used to reduce params quantities and improve nonlinear
                ability.
            attention_mode: use linear layer(linear) or 1x1 convolution(conv)
        """
        super(SKBlock, self).__init__()
        if act_layer is None:
            act_layer = nn.ReLU
        self.num = num
        self.out_chans = out_chans
        self.conv = nn.ModuleList()
        for i in range(num):
            self.conv.append(Conv(in_chans, out_chans, 3, s=stride, g=groups, d=1 + i,
                                  act=True))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # fc can be implemented by 1x1 conv or linear
        assert attention_mode in ['conv', 'linear'], 'fc layer should be implemented by conv or linear'
        self.attention_mode = 'conv'
        if attention_mode == 'conv':
            self.attn = nn.Sequential(
                # use relu act to improve nonlinear expression ability
                Conv(in_chans, out_chans // reduction),
                nn.Conv2d(out_chans // reduction, out_chans * self.num, 1, bias=False),
            )
        else:
            self.attn = nn.Sequential(
                nn.Linear(in_chans, out_chans // reduction, bias=False),
                act_layer(inplace=True),
                nn.Linear(out_chans // reduction, out_chans * self.num, bias=False)
            )
        # compute channels weight
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        batch, c, _, _ = x.size()
        # use different convolutional kernel to conv
        temp_feature = [conv(x) for conv in self.conv]
        # fuse different output
        u = reduce(lambda a, b: a + b, temp_feature)
        # squeeze
        u = u.mean((2, 3), keepdims=True)
        # excitation
        if self.attention_mode != 'conv':
            u = u.view(batch, c)
            z = self.attn(u)
            z = z.reshape(batch, self.num, self.out_chans, -1)
        else:
            z = self.attn(u)
        z = self.softmax(z)
        # select
        a_b_weight: t.List[..., torch.Tensor] = torch.chunk(z, self.num, dim=1)
        a_b_weight = [c.reshape(batch, self.out_chans, 1, 1) for c in a_b_weight]
        v = map(lambda weight, feature: weight * feature, a_b_weight, temp_feature)
        v = reduce(lambda a, b: a + b, v)
        return v


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.act(y)


class BoTAttention(nn.Module):
    """
    Basic self-attention layer with relative position embedding, but there are some difference from original
    transformer, difference as following:

    1.use relative position embedding
    2.use 1x1 convolution kernel to generate q, k, v instead of Linear layer.

    note:
    The representation quality encoded by relative position encoding is better than that encoded by absolute
    position encoding
    """

    def __init__(
            self,
            dim: int,
            width: int,
            height: int,
            head_num: int = 4,
            qkv_bias: bool = False,
    ):
        super(BoTAttention, self).__init__()
        self.head_num = head_num
        self.head_dim = dim // head_num

        self.scale = qkv_bias or self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.flag = 0
        # relative position embedding Rh and Rw, each head is isolated
        # self.rw = None
        # self.rh = None
        self.rw = nn.Parameter(torch.randn((1, head_num, self.head_dim, 1, width)), requires_grad=True)
        self.rh = nn.Parameter(torch.randn((1, head_num, self.head_dim, height, 1)), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        batch, c, h, w = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, 3, self.head_num, c // self.head_num, -1).permute(1, 0, 2, 3, 4)
        # q, k, v dim is (b, head_num, head_dim, length)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # define position encoding based on the height and width of x, and preventing conflicts with checking
        # the model for forward before training
        # if self.flag == 0 and batch != 1:
        #     self.flag = 1
        #     self.rw = None
        #     self.rh = None
        # if getattr(self, 'rw') is None:
        #     setattr(self, 'rw', nn.Parameter(torch.rand((1, self.head_num, self.head_dim, 1, w)).type_as(x), requires_grad=True))
        # if getattr(self, 'rh') is None:
        #     setattr(self, 'rh', nn.Parameter(torch.rand((1, self.head_num, self.head_dim, h, 1)).type_as(x), requires_grad=True))

        # compute attention for content
        attn_content = (q.transpose(-2, -1) @ k) * self.scale

        # compute attention for position
        r = (self.rw + self.rh).view(1, self.head_num, self.head_dim, -1)
        attn_position = (q.transpose(-2, -1) @ r) * self.scale
        attn = self.softmax(attn_content + attn_position)

        attn = (attn @ v.transpose(-2, -1)).permute(0, 1, 3, 2).reshape(batch, c, h, w)
        return attn


class BoTBottleneck(nn.Module):
    def __init__(
            self,
            c1: int,
            c2: int,
            f_size: int = 20,
            shortcut: bool = True,
            g: int = 1,
            k: t.Tuple = (3, 3),
            e: float = 0.5,
            head_num: int = 4
    ):
        """
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.attention = BoTAttention(c_, f_size, f_size, head_num=head_num)
        self.cv2 = nn.Conv2d(c_, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.act(self.bn(self.cv2(self.attention(self.cv1(x))))) if self.add else \
            self.act(self.bn(self.cv2(self.attention(self.cv1(x)))))


class C2fBoT(C2f):

    def __init__(
            self,
            c1: int,
            c2: int,
            n: int = 1,
            f_size: int = 20,
            shortcut: bool = False,
            head_num: int = 4,
            g: int = 1,
            e: float = 0.5
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(BoTBottleneck(self.c, self.c, f_size, shortcut, g, k=((3, 3), (3, 3)), e=1.0,
                                             head_num=head_num) for _ in range(n))


class BasicBlock(nn.Module):
    """
    BasicBlock with residual
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            stride: int = 1,
            down_sample: t.Optional[nn.Module] = None,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        # maybe use conv to down sample
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_chans)
        self.act = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(out_chans)

        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x: torch.Tensor):

        identity = x
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.down_sample:
            identity = self.down_sample(identity)
        y += identity
        return self.act(y)


class ASFFTwo(nn.Module):
    """
    Adaptive attention mechanism based on spatial dimension

    Fusion two different layers into one new layers
    """

    def __init__(
            self,
            in_chans: int,
            act_layer: t.Optional[t.Callable] = None
    ):
        super(ASFFTwo, self).__init__()

        if act_layer is None:
            act_layer = nn.ReLU
        compress_c = 8
        self.in_chas = in_chans
        self.weight_level_1 = Conv(in_chans, compress_c, 1, 1, act=act_layer(inplace=True))
        self.weight_level_2 = Conv(in_chans, compress_c, 1, 1, act=act_layer(inplace=True))

        # spatial attention
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.conv = Conv(in_chans, in_chans, 1, 1, act=act_layer(inplace=True))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        level_1 = self.weight_level_1(x1)
        level_2 = self.weight_level_2(x2)
        levels = torch.cat([level_1, level_2], dim=1)
        levels_weight = self.weight_levels(levels)
        levels_weight = self.softmax(levels_weight)
        # share for all channels
        fused_out = x1 * levels_weight[:, 0:1, :, :] + x2 * levels_weight[:, 1:2, :, :]
        fused_out = self.conv(fused_out)
        return fused_out


class ASFFThree(nn.Module):
    """
    Adaptive attention mechanism based on spatial dimension

    Fusion three different layers into one new layer
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            act_layer: t.Optional[t.Callable] = None
    ):
        super(ASFFThree, self).__init__()
        if act_layer is None:
            act_layer = nn.ReLU
        self.in_chans = in_chans
        compress_c = 8
        self.weight_level_1 = Conv(in_chans, compress_c, 1, 1, act=act_layer(inplace=True))
        self.weight_level_2 = Conv(in_chans, compress_c, 1, 1, act=act_layer(inplace=True))
        self.weight_level_3 = Conv(in_chans, compress_c, 1, 1, act=act_layer(inplace=True))

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.conv = Conv(in_chans, out_chans, 1, 1, act=act_layer(inplace=True))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        level_1 = self.weight_level_1(x1)
        level_2 = self.weight_level_2(x2)
        level_3 = self.weight_level_3(x3)
        levels = torch.cat([level_1, level_2, level_3], dim=1)
        levels_weight = self.weight_levels(levels)
        levels_weight = self.softmax(levels_weight)
        # share for all channels
        fused_out = x1 * levels_weight[:, 0:1, :, :] + x2 * levels_weight[:, 1:2, :, :] + x3 * levels_weight[:, 2:3, :,
                                                                                               :]
        fused_out = self.conv(fused_out)
        return fused_out


class AFPNUpsample(nn.Module):
    """
    the upsample in AFPN
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            act_layer: t.Callable,
            scale_factor: int = 2,
            mode: str = 'nearest',
    ):
        super(AFPNUpsample, self).__init__()
        self.upsample = nn.Sequential(
            Conv(in_chans, out_chans, act=act_layer(inplace=True)),
            nn.Upsample(scale_factor=scale_factor, mode=mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return x


class AFPNC2f(nn.Module):
    """
    Asymptotic Feature Pyramid Network (AFPN) + C2f
    AFPN Block, it adopts cross layer progressive fusion and adaptive space-wise attention mechanism (ASFF)
    """

    def __init__(
            self,
            channels: t.Union[t.List, t.Tuple],
            width: float = 1.0,
            act_layer: t.Optional[t.Callable] = None,
            c2f_nums: t.List[int] = None,
    ):
        """
        Args:
            channels: the number of channels for different layers used for fusion
            width: width param used to implement models at different scales in the channel dimension
            act_layer: activate layers
        """
        super(AFPNC2f, self).__init__()
        if act_layer is None:
            act_layer = nn.ReLU
        if c2f_nums is None:
            c2f_nums = [1, 2, 2]

        # The semantic degree, from low to high, is 0, 1, and 2, corresponding to top, mid and bottom

        # dimensional alignment, fuse low and mid semantics layers, 0_0_1 means first fuse, from 0 layer to 1 layer
        self.down_sample_1_0_1 = Conv(channels[0], channels[1], 2, 2, p=0, act=act_layer(inplace=True))
        self.up_sample_1_1_0 = AFPNUpsample(channels[1], channels[0], act_layer=act_layer)
        # 1_0 means first fuse, fused layer is 0 layer
        self.asff_top1_0 = ASFFTwo(in_chans=channels[0])
        self.asff_mid1_1 = ASFFTwo(in_chans=channels[1])

        self.c2f1_0 = C2f(channels[0], channels[0], c2f_nums[0])
        self.c2f1_1 = C2f(channels[1], channels[1], c2f_nums[1])

        # dimensional alignment, fuse low, mid, high semantics layers, 2_0_1 means second fuse, from 0 layer to 1 layer
        self.down_sample_2_0_1 = Conv(channels[0], channels[1], 2, 2, p=0, act=act_layer(inplace=True))
        self.down_sample_2_0_2 = Conv(channels[0], channels[2], 4, 4, p=0, act=act_layer(inplace=True))
        self.down_sample_2_1_2 = Conv(channels[1], channels[2], 2, 2, p=0, act=act_layer(inplace=True))
        self.up_sample_2_1_0 = AFPNUpsample(channels[1], channels[0], act_layer=act_layer, scale_factor=2,
                                            mode='bilinear')
        self.up_sample_2_2_0 = AFPNUpsample(channels[2], channels[0], act_layer=act_layer, scale_factor=4,
                                            mode='bilinear')
        self.up_sample_2_2_1 = AFPNUpsample(channels[2], channels[1], act_layer=act_layer, scale_factor=2,
                                            mode='bilinear')

        # 2_0 means second fuse, fused layer is 1 layer
        self.asff_top2_0 = ASFFThree(in_chans=channels[0], out_chans=int(channels[0] * width))
        self.asff_mid2_1 = ASFFThree(in_chans=channels[1], out_chans=int(channels[1] * width))
        self.asff_bottom2_2 = ASFFThree(in_chans=channels[2], out_chans=int(channels[2] * width))
        self.c2f2_0 = C2f(channels[0], channels[0], c2f_nums[0])
        self.c2f2_1 = C2f(channels[1], channels[1], c2f_nums[1])
        self.c2f2_2 = C2f(channels[2], channels[2], c2f_nums[2])

    def forward(self, x: t.Union[t.List[torch.Tensor], t.Tuple[torch.Tensor]]) -> t.Tuple[torch.Tensor, ...]:
        x0, x1, x2 = x

        top = self.asff_top1_0(x0, self.up_sample_1_1_0(x1))
        mid = self.asff_mid1_1(self.down_sample_1_0_1(x0), x1)

        x0 = self.c2f1_0(top)
        x1 = self.c2f1_1(mid)

        top = self.asff_top2_0(x0, self.up_sample_2_1_0(x1), self.up_sample_2_2_0(x2))
        mid = self.asff_mid2_1(self.down_sample_2_0_1(x0), x1, self.up_sample_2_2_1(x2))
        bot = self.asff_bottom2_2(self.down_sample_2_0_2(x0), self.down_sample_2_1_2(x1), x2)

        top = self.c2f2_0(top)
        mid = self.c2f2_1(mid)
        bot = self.c2f2_2(bot)

        return top, mid, bot

class BiCrossFPN(nn.Module):
    """
    BiFPN + Cross FPN, progressive feature fusion is only performed between 4 layers in sequence in this module
    """

    def __init__(
            self,
            channels: t.Union[t.List, t.Tuple],
            width: float = 1.0,
            act_layer: t.Optional[t.Callable] = None,
            c2f_nums: t.List[int] = None,
    ):
        super(BiCrossFPN, self).__init__()
        if act_layer is None:
            act_layer = nn.ReLU
        if c2f_nums is None:
            c2f_nums = [1, 1, 1, 1]

        self.down_sample_0_2 = LightConv2(channels[0], channels[2], 4, 4, p=0, act=act_layer(inplace=True))
        self.down_sample_0_3 = LightConv2(channels[0], channels[3], 8, 8, p=0, act=act_layer(inplace=True))
        self.down_sample_1_3 = LightConv2(channels[1], channels[3], 4, 4, p=0, act=act_layer(inplace=True))

        self.up_sample_2_0 = AFPNUpsample(channels[2], channels[0], act_layer=act_layer, scale_factor=4,
                                            mode='nearest')
        self.up_sample_3_0 = AFPNUpsample(channels[3], channels[0], act_layer=act_layer, scale_factor=8,
                                            mode='nearest')
        self.up_sample_3_1 = AFPNUpsample(channels[3], channels[1], act_layer=act_layer, scale_factor=4,
                                            mode='nearest')

        self.asff_0 = ASFFThree(in_chans=channels[0], out_chans=channels[0])
        self.asff_1 = ASFFTwo(in_chans=channels[1])
        self.asff_2 = ASFFTwo(in_chans=channels[2])
        self.asff_3 = ASFFThree(in_chans=channels[3], out_chans=channels[3])
        self.c2f0 = C2fFaster(channels[0], channels[0], c2f_nums[0], False)
        self.c2f1 = C2fFaster(channels[1], channels[1], c2f_nums[1], False)
        self.c2f2 = C2fFaster(channels[2], channels[2], c2f_nums[2], False)
        self.c2f3 = C2fFaster(channels[3], channels[3], c2f_nums[3], False)

    def forward(self, x: t.Union[t.List[torch.Tensor], t.Tuple[torch.Tensor]]) -> t.Tuple[torch.Tensor, ...]:
        x0, x1, x2, x3 = x
        p0 = self.asff_0(x0, self.up_sample_2_0(x2), self.up_sample_3_0(x3))
        p1 = self.asff_1(x1, self.up_sample_3_1(x3))
        p2 = self.asff_2(self.down_sample_0_2(x0), x2)
        p3 = self.asff_3(self.down_sample_0_3(x0), self.down_sample_1_3(x1), x3)

        p0 = self.c2f0(p0)
        p1 = self.c2f1(p1)
        p2 = self.c2f2(p2)
        p3 = self.c2f3(p3)

        return p0, p1, p2, p3


class FasterBlocks(nn.Module):
    """
    stacking multiple  FasterBlock
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            n: int,
            drop_paths: t.List[float],
            n_div: int = 4,
            mlp_ratio: float = 2.,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
            pconv_fw_type: str = 'split_cat',
    ):
        super(FasterBlocks, self).__init__()
        blocks = []
        for i in range(n):
            blocks.append(FasterBlock(
                dim=dim,
                out_dim=out_dim,
                drop_path=drop_paths[i],
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                pconv_fw_type=pconv_fw_type
            ))
            dim = out_dim
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class FasterBlock(nn.Module):
    """
    based on PConv + PWC
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            drop_path: float,
            n_div: int = 4,
            mlp_ratio: float = 2.,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
            pconv_fw_type: str = 'split_cat',
    ):
        super(FasterBlock, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div
        hidden_chans = int(dim * mlp_ratio)

        self.token_mixer = PartialConv(dim, n_div, pconv_fw_type)
        self.mlp = nn.Sequential(
            Conv(dim, hidden_chans, k=1, s=1),
            Conv(hidden_chans, dim, k=1, s=1, act=False),
        )
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim,), requires_grad=True)
            )
        # adapt to neck, since dim and dim_out may be different
        self.dim = dim
        self.out_dim = out_dim
        if dim != out_dim:
            self.trans = Conv(dim, out_dim, k=1, s=1, act=False)

    def forward(self, x: torch.Tensor):
        """
        X dim is (B, C, H, W)
        """
        identity = x
        x = self.token_mixer(x)
        if self.use_layer_scale:
            x = identity + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        else:
            x = identity + self.drop_path(self.mlp(x))
        if self.dim != self.out_dim:
            x = self.trans(x)
        return x


class AFPNPConv(nn.Module):
    """
    Asymptotic Feature Pyramid Network (AFPN) + Pconv
    AFPN Block, it adopts cross layer progressive fusion and adaptive space-wise attention mechanism (ASFF)
    """

    def __init__(
            self,
            channels: t.Union[t.List, t.Tuple],
            width: float = 1.0,
            act_layer: t.Optional[t.Callable] = None,
            c2f_nums: t.List[int] = None,
            drop_path_ratio: float = 0.,
            n_div: int = 4,
            mlp_ratio: float = 2.,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5
    ):
        """
        Args
            channels: the number of channels for different layers used for fusion
            width: width param used to implement models at different scales in the channel dimension
            act_layer: activate layers
        """
        super(AFPNPConv, self).__init__()
        if act_layer is None:
            act_layer = nn.ReLU
        if c2f_nums is None:
            c2f_nums = [1, 2, 2]

        # The semantic degree, from low to high, is 0, 1, and 2, corresponding to top, mid and bottom

        # dimensional alignment, fuse low and mid semantics layers, 0_0_1 means first fuse, from 0 layer to 1 layer
        self.down_sample_1_0_1 = Conv(channels[0], channels[1], 2, 2, p=0, act=act_layer(inplace=True))
        self.up_sample_1_1_0 = AFPNUpsample(channels[1], channels[0], act_layer=act_layer)
        # 1_0 means first fuse, fused layer is 0 layer
        self.asff_top1_0 = ASFFTwo(in_chans=channels[0])
        self.asff_mid1_1 = ASFFTwo(in_chans=channels[1])

        self.c2f1_0 = nn.Sequential(*(
            FasterBlock(
                dim=channels[0],
                drop_path=drop_path_ratio,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value
            )
            for _ in range(c2f_nums[0])))

        self.c2f1_1 = nn.Sequential(*(
            FasterBlock(
                dim=channels[1],
                drop_path=drop_path_ratio,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value
            ) for _ in range(c2f_nums[1])
        ))

        # dimensional alignment, fuse low, mid, high semantics layers, 2_0_1 means second fuse, from 0 layer to 1 layer
        self.down_sample_2_0_1 = Conv(channels[0], channels[1], 2, 2, p=0, act=act_layer(inplace=True))
        self.down_sample_2_0_2 = Conv(channels[0], channels[2], 4, 4, p=0, act=act_layer(inplace=True))
        self.down_sample_2_1_2 = Conv(channels[1], channels[2], 2, 2, p=0, act=act_layer(inplace=True))
        self.up_sample_2_1_0 = AFPNUpsample(channels[1], channels[0], act_layer=act_layer, scale_factor=2,
                                            mode='bilinear')
        self.up_sample_2_2_0 = AFPNUpsample(channels[2], channels[0], act_layer=act_layer, scale_factor=4,
                                            mode='bilinear')
        self.up_sample_2_2_1 = AFPNUpsample(channels[2], channels[1], act_layer=act_layer, scale_factor=2,
                                            mode='bilinear')

        # 2_0 means second fuse, fused layer is 1 layer
        self.asff_top2_0 = ASFFThree(in_chans=channels[0], out_chans=int(channels[0] * width))
        self.asff_mid2_1 = ASFFThree(in_chans=channels[1], out_chans=int(channels[1] * width))
        self.asff_bottom2_2 = ASFFThree(in_chans=channels[2], out_chans=int(channels[2] * width))

        self.c2f2_0 = nn.Sequential(*(
            FasterBlock(
                dim=channels[0],
                drop_path=drop_path_ratio,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value
            )
            for _ in range(c2f_nums[0])))
        self.c2f2_1 = nn.Sequential(*(
            FasterBlock(
                dim=channels[1],
                drop_path=drop_path_ratio,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value
            )
            for _ in range(c2f_nums[1])))

        self.c2f2_2 = nn.Sequential(*(
            FasterBlock(
                dim=channels[2],
                drop_path=drop_path_ratio,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value
            )
            for _ in range(c2f_nums[2])))

    def forward(self, x: t.Union[t.List[torch.Tensor], t.Tuple[torch.Tensor]]) -> t.Tuple[torch.Tensor, ...]:
        x0, x1, x2 = x

        top = self.asff_top1_0(x0, self.up_sample_1_1_0(x1))
        mid = self.asff_mid1_1(self.down_sample_1_0_1(x0), x1)

        x0 = self.c2f1_0(top)
        x1 = self.c2f1_1(mid)

        top = self.asff_top2_0(x0, self.up_sample_2_1_0(x1), self.up_sample_2_2_0(x2))
        mid = self.asff_mid2_1(self.down_sample_2_0_1(x0), x1, self.up_sample_2_2_1(x2))
        bot = self.asff_bottom2_2(self.down_sample_2_0_2(x0), self.down_sample_2_1_2(x1), x2)

        top = self.c2f2_0(top)
        mid = self.c2f2_1(mid)
        bot = self.c2f2_2(bot)

        return top, mid, bot


class BottleneckCondConv(nn.Module):
    """
    Bottleneck with CondConv
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            shortcut: bool = True,
            g: int = 1,
            k: t.Tuple = (3, 3),
            e: float = 0.5,
            experts_num: int = 8,
            drop_ratio: float = 0.2,
    ):
        super(BottleneckCondConv, self).__init__()
        c_ = int(c1 * e)
        self.cv1 = CondConv(c1, c_, kernel_size=k[0], experts_num=experts_num, groups=g, drop_ratio=drop_ratio)
        self.cv2 = CondConv(c_, c2, kernel_size=k[1], experts_num=experts_num, groups=g, drop_ratio=drop_ratio)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2fCondConv(nn.Module):
    """
    C2f with CondConv
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            n: int = 1,
            shortcut: bool = False,
            experts_num: int = 8,
            drop_ratio: float = 0.2,
            g: int = 1,
            e: float = 0.5,
    ):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        # 分割梯度流
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = CondConv(c1, 2 * self.c, 1, 1, experts_num=experts_num, drop_ratio=drop_ratio)
        self.cv2 = CondConv((2 + n) * self.c, c2, 1, experts_num=experts_num, drop_ratio=drop_ratio)
        self.m = nn.ModuleList(BottleneckCondConv(self.c, self.c, shortcut, g, k=(3, 3),
                                                  e=1.0, experts_num=experts_num, drop_ratio=drop_ratio)
                               for _ in range(n))

    def forward(self, x) -> torch.Tensor:
        """Forward pass through C2f layer."""
        # 沿着通道维度进行拆分
        y = list(self.cv1(x).chunk(2, 1))
        # 右半部分经过一系列的Bottleneck, 将Yolo中的Conv块替换为Bottleneck残差结构
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class ChannelAttention(nn.Module):
    """
    Channel attention module based on CPCA
    use hidden_chans to reduce parameters instead of conventional convolution
    """

    def __init__(self, in_chans: int, hidden_chans: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_chans, hidden_chans, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(hidden_chans, in_chans, kernel_size=1, stride=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.in_chans = in_chans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (B, C, H, W)
        """
        # (B, C, 1, 1)
        # x1 = x.mean(dim=(2, 3), keepdim=True)
        x1 = self.avg_pool(x)
        x1 = self.fc2(self.act(self.fc1(x1)))
        x1 = torch.sigmoid(x1)

        # (B, C, 1, 1)
        # x2 = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x2 = self.max_pool(x)
        x2 = self.fc2(self.act(self.fc1(x2)))
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.in_chans, 1, 1)
        return x


class CPCA(nn.Module):
    """
    Channel Attention and Spatial Attention based on CPCA
    """

    def __init__(self, in_chans: int, reduction_ratio: int = 4):
        super(CPCA, self).__init__()
        self.in_chans = in_chans

        hidden_chans = in_chans // reduction_ratio
        # Channel Attention
        self.ca = ChannelAttention(in_chans, hidden_chans)

        # Spatial Attention
        self.dwc5_5 = nn.Conv2d(in_chans, in_chans, kernel_size=5, padding=2, groups=in_chans)
        self.dwc1_7 = nn.Conv2d(in_chans, in_chans, kernel_size=(1, 7), padding=(0, 3), groups=in_chans)
        self.dwc7_1 = nn.Conv2d(in_chans, in_chans, kernel_size=(7, 1), padding=(3, 0), groups=in_chans)
        self.dwc1_11 = nn.Conv2d(in_chans, in_chans, kernel_size=(1, 11), padding=(0, 5), groups=in_chans)
        self.dwc11_1 = nn.Conv2d(in_chans, in_chans, kernel_size=(11, 1), padding=(5, 0), groups=in_chans)
        self.dwc1_21 = nn.Conv2d(in_chans, in_chans, kernel_size=(1, 21), padding=(0, 10), groups=in_chans)
        self.dwc21_1 = nn.Conv2d(in_chans, in_chans, kernel_size=(21, 1), padding=(10, 0), groups=in_chans)

        # used to model feature connections between different receptive fields
        self.conv = nn.Conv2d(in_chans, in_chans, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        channel_attn = self.ca(x)
        x = channel_attn * x

        x_init = self.dwc5_5(x)
        x1 = self.dwc1_7(x_init)
        x1 = self.dwc7_1(x1)

        x2 = self.dwc1_11(x_init)
        x2 = self.dwc11_1(x2)

        x3 = self.dwc1_21(x_init)
        x3 = self.dwc21_1(x3)

        spatial_atn = x1 + x2 + x3 + x_init
        spatial_atn = self.conv(spatial_atn)
        y = x * spatial_atn
        y = self.conv(y)
        return y


class ECA(nn.Module):
    """
    Efficient Channel Attention:
    1.Efficient and lightweight channel attention mechanism with low model complexity
    2.The Core innovation points are as follows:
        - dimensionality reduction may have a side effect on channel interaction, so discard it.
        - use GWConv, which can be regarded as a depth-wise separable convolution, and generateds channel weights
        by performing a fast 1D convolution of size K, where k is adaptively determined  via a non-linearity mapping
        of channel dimension C.

    note: transpose channel dimension and spatial dimension to use fast 1D convolution with kernel size K. K is based
    on the channel dimension.
    """

    def __init__(
            self,
            in_chans: int,
            kernel_size: t.Optional[int] = None,
            gamma: int = 2,
            b: int = 1,
    ):
        super(ECA, self).__init__()
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        t = int(abs((log(in_chans, 2) + b) / gamma))
        kernel_size = kernel_size or t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (B, C, H, W)
        """
        # (B, C, 1, 1)
        y = x.mean((2, 3), keepdim=True)
        # (B, 1, C)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        # (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class EMA(nn.Module):
    """
    EMA
    """
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class RepStem(nn.Module):
    """
    RepConv with PatchEmbedding
    """

    def __init__(self, in_chans: int, out_chans: int):
        super(RepStem, self).__init__()
        self.stem = nn.Sequential(
            RepConv(in_chans, out_chans // 2,  k=3, s=2, act=nn.ReLU(inplace=False)),
            RepConv(out_chans // 2, out_chans, k=3, s=2, act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class RepStemDWC(nn.Module):
    """
    RepConv + DWConv with PatchEmbedding
    """

    def __init__(self, in_chans: int, out_chans: int):
        super(RepStemDWC, self).__init__()
        self.stem = nn.Sequential(
            RepConv(in_chans, out_chans // 2, k=3, s=2, act=nn.ReLU(inplace=False)),
            RepConv(out_chans // 2, out_chans, k=3, s=2, g=out_chans // 2, act=nn.ReLU(inplace=False)),
            RepConv1x1(out_chans, out_chans, k=1, s=1, act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class LightBlocks(nn.Module):
    """
    stacking multiple blocks of  DWC and PWC
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            depth: int,
            k: int = 1,
            s: int = 1,
            p: t.Optional[int] = None,
            g: t.Optional[int] = None,
            d: int = 1,
            act: t.Union[bool, nn.Module] = True
    ):
        super(LightBlocks, self).__init__()
        blocks = []
        for i in range(depth):
            blocks.append(
                LightConv2(
                    c1=c1,
                    c2=c2,
                    k=k,
                    s=s,
                    p=p,
                    g=g,
                    d=d,
                    act=act
                )
            )
            c1 = c2
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SPPCA(nn.Module):
    """
    Spatial prior partial convolution attention
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            n_div: int = 4,
            forward: str = 'split_cat',
    ):
        super(SPPCA, self).__init__()
        self.pconv = PartialConv(in_chans, n_div, forward)
        self.conv = Conv(in_chans, out_chans, k=1, s=1, act=nn.Hardswish(inplace=True))

        self.conv2 = Conv(in_chans, out_chans, k=1, s=1, act=False)
        self.dw1_5 = Conv(out_chans, out_chans, k=(1, 5), s=1, p=(0, 2), g=out_chans, act=False)
        self.dw5_1 = Conv(out_chans, out_chans, k=(5, 1), s=1, p=(2, 0), g=out_chans, act=False)

        self.dw1_9 = Conv(out_chans, out_chans, k=(1, 9), s=1, p=(0, 4), g=out_chans, act=False)
        self.dw9_1 = Conv(out_chans, out_chans, k=(9, 1), s=1, p=(4, 0), g=out_chans, act=False)

        self.ca = ECA(out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv2(x)
        sa1 = self.dw5_1(self.dw1_5(attn))
        sa2 = self.dw9_1(self.dw1_9(attn))
        attn = attn + sa1 + sa2
        attn = self.ca(attn)

        x = self.conv(self.pconv(x)) * attn
        return x


