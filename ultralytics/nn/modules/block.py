# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""
import math
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, ConvOD, PartialConv
from .transformer import TransformerBlock
__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C2fOD', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'BottleneckOD', 'Proto', 'RepC3', 'CABlock', 'C2fFaster',
           'SEBlock', 'SKBlock')


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
        # self.cv3 = Conv(c_, c2, 1, 1, g=g)
        self.cv3 = nn.Conv2d(c_, c2, 1, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))



class SpatialACT(nn.Module):
    """
    which is smoother than Relu
    """

    def __init__(self, inplace=True):
        super(SpatialACT, self).__init__()
        self.act = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor):
        return x * self.act(x + 3) / 6


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
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        super(CABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU6
        self.in_chans = in_chans
        self.out_chans = out_chans
        # (C, H, 1)
        self.gap_h = nn.AdaptiveAvgPool2d((None, 1))
        # (C, 1, W)
        self.gap_w = nn.AdaptiveAvgPool2d((1, None))

        hidden_chans = max(8, in_chans // reduction)
        self.conv1 = nn.Conv2d(in_chans, hidden_chans, 1)
        self.bn1 = norm_layer(hidden_chans)
        self.act = SpatialACT(inplace=True)

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
        x_h = self.gap_h(x)
        x_w = self.gap_w(x).permute(0, 1, 3, 2)
        y = torch.cat((x_h, x_w), dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # split
        x_h, x_w = torch.split(y, [h, w], dim=2)
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
            act_layer: t.Optional[nn.Module] = None,
    ):
        super(SEBlock, self).__init__()
        if act_layer is None:
            act_layer = nn.SiLU
        # part 1:(H, W, C) -> (1, 1, C)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_mode = attention_mode
        # part 2, compute weight of each channel
        if attention_mode == 'conv':
            self.fc = nn.Sequential(
                nn.Conv2d(in_chans, out_chans // reduction, 1, bias=False),
                act_layer(inplace=True),
                nn.Conv2d(out_chans // reduction, out_chans, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_chans, out_chans // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(out_chans // reduction, in_chans, bias=False),
                nn.Sigmoid(),  # nn.Softmax is OK here
            )

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        if self.attention_mode != 'conv':
            y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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
            act_layer: t.Optional[nn.Module] = None,
            num: int = 2,
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
            act_layer = nn.SiLU
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
            self.fc = nn.Sequential(
                # use relu act to improve nonlinear expression ability
                nn.Conv2d(in_chans, out_chans // reduction, 1, bias=False),
                act_layer(inplace=True),
                nn.Conv2d(out_chans // reduction, out_chans * self.num, 1, bias=False),
            )
        else:
            self.fc = nn.Sequential(
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
        u = self.gap(u)
        # excitation
        if self.attention_mode != 'conv':
            u = u.view(batch, c)
        z = self.fc(u)
        z = z.reshape(batch, self.num, self.out_chans, -1)
        z = self.softmax(z)
        # select
        a_b_weight: t.List[..., torch.Tensor] = torch.chunk(z, self.num, dim=1)
        a_b_weight = [c.reshape(batch, self.out_chans, 1, 1) for c in a_b_weight]
        v = map(lambda weight, feature: weight * feature, a_b_weight, temp_feature)
        v = reduce(lambda a, b: a + b, v)
        return v