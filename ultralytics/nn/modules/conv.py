# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t

__all__ = ('autopad', 'Conv', 'Conv2', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv', 'ODConv', 'ConvOD', 'PartialConv',
           'PConv', 'CondConv', 'RepConv1x1', 'ConcatBiFPN', 'LightConv2', 'ConcatBiDirectFPN', 'MAConv',
           'EnhancedConcat', 'RepMAConv')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class OmniAttention(nn.Module):
    """
    Attention, used to compute include four types attention of different dimensions in ODConv
    1.the spatial kernel size kxk, Cs
    2.the input channel number Cin
    3.the output channel number Cf
    4.the convolution kernel number Cw

    note:
    1.using multi-head attention to compute four dimension in a parallel manner.
    2.the weight of input channel, output channel, spatial kernel is shared to all convolution kernels.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            groups: int = 1,
            reduction: float = 0.0625,
            expert_num: int = 4,
            hidden_chans: int = 16,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        """
        reduction means the scaling factor of the first linear layer, default is 1 / 16 = 0.0625
        expert_num means the number of convolution kernels, default is 4
        hidden_chans means the scaling size after first liner layer, default is 16
        """
        super(OmniAttention, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        attention_chans = max(int(in_chans * reduction), hidden_chans)
        self.kernel_size = kernel_size
        self.expert_num = expert_num
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(in_chans, attention_chans, 1, bias=False)
        self.bn = norm_layer(attention_chans)
        self.act = act_layer(inplace=True)

        # in channel attention, like SE Attention Module
        self.in_chans_fc = nn.Conv2d(attention_chans, in_chans, 1, bias=True)
        self.channel_attention = self.get_channel_attention

        # out channel attention, like SE Attention Module
        # if current conv is DWC, ignore filter attention
        if in_chans == groups and in_chans == out_chans:
            self.filter_attention = self.ignore
        else:
            self.filter_fc = nn.Conv2d(attention_chans, out_chans, 1, bias=True)
            self.filter_attention = self.get_filter_attention

        # spatial channel attention
        # if current conv is PWD, ignore spatial channel attention
        if kernel_size == 1:
            self.spatial_attention = self.ignore
        else:
            self.spatial_fc = nn.Conv2d(attention_chans, kernel_size ** 2, 1, bias=True)
            self.spatial_attention = self.get_spatial_attention

        # kernel num attention, like SE Attention Module
        # if kernel num is one, ignore kernel num attention
        if expert_num == 1:
            self.expert_attention = self.ignore
        else:
            self.expert_fc = nn.Conv2d(attention_chans, expert_num, 1, bias=True)
            self.expert_attention = self.get_expert_attention

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def ignore(x: torch.Tensor) -> float:
        return 1.0

    def get_channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (b, in_chans, 1, 1)
        return dim is (b, in_chans, 1, 1)
        """
        channel_attention = x.view(x.size(0), -1, 1, 1)
        channel_attention = self.in_chans_fc(channel_attention)
        channel_attention = torch.sigmoid(channel_attention)
        return channel_attention

    def get_filter_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (b, in_chans, 1, 1)
        return dim is (b, out_chans, 1, 1)
        """
        filter_attention = x.view(x.size(0), -1, 1, 1)
        filter_attention = self.filter_fc(filter_attention)
        filter_attention = torch.sigmoid(filter_attention)
        return filter_attention

    def get_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (b, in_chans, 1, 1)
        return dim is (b, 1, 1, 1, kernel_size, kernel_size)
        batch size, expert num, height, weight, kernel size, kernel size
        """
        # trans k(kernel size) x k x 1 to 1(channel) x 1(height) x 1(width) x k (kernel size) x k
        spatial_attention = self.spatial_fc(x)
        spatial_attention = spatial_attention.view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention)
        return spatial_attention

    def get_expert_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (b, in_chans, 1, 1)
        return dim is (b, expert_num, 1, 1, 1, 1)
        batch size, expert num, height, weight, kernel size, kernel size
        """
        expert_attention = self.expert_fc(x)
        # -1 means the number of convolution kernel
        expert_attention = expert_attention.view(x.size(0), -1, 1, 1, 1, 1)
        expert_attention = torch.softmax(expert_attention, dim=1)
        return expert_attention

    def forward(self, x: torch.Tensor):
        x = self.gap(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)

        return self.channel_attention(x), self.filter_attention(x), self.spatial_attention(x), self.expert_attention(x)


class ODConv(nn.Module):
    """
    ODConv --- new dynamic convolution called Omni-Dimension Convolution, better than CondConv, DyConv...
    ODConv leverages a multi-dimensional attention mechanism to learn four types of attentions for
    convolutional kernels along four dimensions of the kernel space in a parallel manner.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            reduction: float = 0.0625,
            hidden_chans: int = 16,
            expert_num: int = 4,
            bias: bool = False,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        super(ODConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.reduction = reduction
        self.hidden_chans = hidden_chans
        self.expert_num = expert_num
        self.bias = bias

        self.attention = OmniAttention(in_chans, out_chans, kernel_size, groups, reduction, expert_num,
                                       hidden_chans, norm_layer=norm_layer, act_layer=act_layer)
        # expert weight and bias
        self.weight = nn.Parameter(torch.randn(expert_num, out_chans, in_chans // groups, kernel_size, kernel_size),
                                   requires_grad=True)

        if self.bias:
            self.bias = nn.Parameter(torch.randn(expert_num, out_chans), requires_grad=True)
        else:
            self.bias = None

        # ODConv1x + PWC, not need spatial attention and expert attention
        if self.kernel_size == 1 and self.expert_num == 1:
            self._forward = self._forward_pw1x
        else:
            self._forward = self._forward_omni

        self._init_weights()

    def _init_weights(self):
        for i in range(self.expert_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')
            if self.bias is not None:
                nn.init.zeros_(self.bias[i])

    def _forward_pw1x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Only learn two types of attentions for convolutional along two dimensions of input channel and output channel

        Broadcast point multiplication or matrix multiplication can be used, matrix multiplication reference
        CondConv implementation
        """
        channel_attention, filter_attention, _, _ = self.attention(x)
        x = x * channel_attention
        combined_bias = None

        # note: because expert_attention use softmax to normalize, when only have one expert and 1x1 size,
        # its attentions is 1. so, dynamic convolution in spatial size dimension and expert dimension will be
        # transformed into static convolution!
        if self.bias is not None:
            combined_bias = self.bias.squeeze(dim=0)
        y = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=combined_bias, stride=self.stride,
                     padding=autopad(self.kernel_size, d=self.dilation), groups=self.groups)
        y = y * filter_attention
        return y

    def _forward_omni(self, x: torch.Tensor) -> torch.Tensor:
        """
        Learn four types of attentions for convolutional kernels along four dimensions of the kernel space

        Broadcast point multiplication or matrix multiplication can be used, matrix multiplication reference
        CondConv implementation in other_util/conv/CondConv.py
        """

        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        b, c, h, w = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, h, w)
        combined_weights = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        combined_weights = torch.sum(combined_weights, dim=1).view(-1, self.in_chans // self.groups,
                                                                   self.kernel_size, self.kernel_size)
        combined_bias = None
        if self.bias is not None:
            combined_bias = torch.mm(kernel_attention.squeeze(), self.bias).view(-1)
        y = F.conv2d(x, weight=combined_weights, stride=self.stride, bias=combined_bias,
                     padding=autopad(self.kernel_size, d=self.dilation),
                     dilation=self.dilation, groups=self.groups * b)
        y = y.view(b, self.out_chans, y.size(-2), y.size(-1))
        y = y * filter_attention
        return y

    def forward(self, x: torch) -> torch.Tensor:
        return self._forward(x)

class ConvOD(nn.Module):
    """
    ODConv(Relu) + BN + SiLu
    """
    default_act = nn.SiLU()

    def __init__(
            self,
            c1: int,
            c2: int,
            k: int = 1,
            s: int = 1,
            p: t.Optional[int] = None,
            g: int = 1,
            d: int = 1,
            act: bool = True,
            reduction: float = 0.0625,
            expert_num: int = 4,
            hidden_chans: int = 16,
    ):
        super(ConvOD, self).__init__()
        self.conv = ODConv(c1, c2, k, s, groups=g, dilation=d, reduction=reduction, expert_num=expert_num,
                           hidden_chans=hidden_chans, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class MAConv(nn.Module):
    """
    MaxPool + AvgPool + Conv for down-sampling
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            k: int = 1,
            s: int = 1,
            p: t.Optional[int] = None,
            g: int = 1,
            d: int = 1,
            act: t.Union[bool, nn.Module] = True
    ):


        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, d, act)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c1 = c1
        self.c2 = c2
        if c1 != c2:
            self.max_conv = Conv(c1, c2, 1, 1, act=act)
            self.avg_conv = Conv(c1, c2, 1, 1, act=act)
        self.attn = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = 0.0001


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        x2 = self.max_pool(x)
        x3 = self.avg_pool(x)
        attn = torch.sigmoid(self.attn)
        x = x1 * attn[0] + x2 * attn[1] + x3 * attn[2]
        return x


class RepMAConv(nn.Module):
    """
    RepConv + MaxPool + AvgPool + Conv for down-sampling
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            k: int = 1,
            s: int = 1,
            p: t.Optional[int] = None,
            g: int = 1,
            d: int = 1,
            act: t.Union[bool, nn.Module] = True
    ):
        super(RepMAConv, self).__init__()
        self.conv = RepConv(c1, c2, k, s, autopad(k, p=p, d=d), g, d, act)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c1 = c1
        self.c2 = c2
        if c1 != c2:
            self.max_conv = Conv(c1, c2, 1, 1, act=act)
            self.avg_conv = Conv(c1, c2, 1, 1, act=act)
        self.attn = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = 0.0001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        x2 = self.max_pool(x)
        x3 = self.avg_pool(x)
        attn = torch.sigmoid(self.attn)
        x = x1 * attn[0] + x2 * attn[1] + x3 * attn[2]
        return x

class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse


class PConv(nn.Module):
    """
    PConv + BN + act and allow fused in inference
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            k: int,
            s: int,
            n_div: int = 4,
            p: t.Optional[int] = None,
            g: int = 1,
            d: int = 1,
    ):
        super(PConv, self).__init__()
        self.dim_partial = c1 // n_div
        self.dim_untouched = c1 - self.dim_partial
        self.conv1 = nn.Conv2d(self.dim_partial, self.dim_partial, k, s, autopad(k, p, d), groups=g, dilation=d,
                              bias=False)
        self.conv2 = Conv(c1, c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, batch normalization and activation to input tensor."""
        x1, x2 = torch.split(x, [self.dim_partial, self.dim_untouched], dim=1)
        x1 = self.conv1(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv2(x)
        return x



class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))

class LightConv2(nn.Module):

    def __init__(
            self,
            c1: int,
            c2: int,
            k: int = 1,
            s: int = 1,
            p: t.Optional[int] = None,
            g: t.Optional[int] = None,
            d: int = 1,
            act: t.Union[bool, nn.Module] = True
    ):
        super(LightConv2, self).__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = Conv(c1, c1, k, s, p=p, g=g or math.gcd(c1, c2), d=d, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(self.conv2(x))

class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

class RepConv1x1(nn.Module):
    """
    RepConv with 1x1 Conv, BN, Act.
    when reasoning, merge 1x1 convolution and BN to form a new 1x1 convolution.

    note: compared RepConv above, it supports 1x1 Conv and BN reparameters
    """
    default_act = nn.SiLU()

    def __init__(
            self,
            c1: int,
            c2: int,
            k: int = 1,
            s: int = 1,
            p: int = 0,
            g: int = 1,
            d: int = 1,
            act: t.Union[nn.Module, bool] = True,
            bn: bool = False,
            deploy=False
    ):
        super(RepConv1x1, self).__init__()
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, 1, s, p=0, g=g, act=False)

    def forward_fuse(self, x) -> torch.Tensor:
        return self.act(self.conv(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + id_out)

    def get_equivalent_kernel_bias(self):

        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv1)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel1x1 + kernelid, bias1x1 + biasid

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 1, 1), dtype=np.float32)
                # 生成类单位矩阵
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 0, 0] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """
        fuse 1x1 Conv and BN
        """
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for param in self.parameters():
            param.detach_()
        self.__delattr__('conv1')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('nm')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)

class ConcatBiFPN(nn.Module):
    """
    weighted feature fusion, since different input features are at different resolutions, they usually contribute to
    the output feature unequally.
    """
    def __init__(self, dimension: int = 1, weight_num: int = 2):
        super(ConcatBiFPN, self).__init__()
        self.d = dimension
        self.weights = nn.Parameter(torch.ones(weight_num, dtype=torch.float32), requires_grad=True)
        self.eps = 0.0001
        self.weight_num = weight_num
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.relu(self.weights)
        w = w / (torch.sum(w, dim=0) + self.eps)
        x = [w[i] * x[i] for i in range(self.weight_num)]
        return torch.cat(x, dim=self.d)


class EnhancedConcat(nn.Module):
    """
    Enhancing feature fusion capability by introducing more path aggregation
    """

    def __init__(
            self,
            dimension: int = 1,
            mode: str = 'up',
    ):
        super(EnhancedConcat, self).__init__()
        self.d = dimension
        self.mode = mode
        self.trans = nn.Upsample(None, 2, 'nearest')

    def forward(self, x: torch.Tensor):
        temp = x[-1].clone()
        temp = self.trans(temp)
        return torch.cat([x[0], x[1], temp], dim=self.d)

class ConcatBiDirectFPN(nn.Module):
    """
    Add additional direct feature interaction between adjacent layers
    """

    def __init__(
            self,
            dimension: int = 1,
            weight_num: int = 2,
            mode: str = 'up',
            c1: int = 128,
            c2: int = 64,
    ):
        super(ConcatBiDirectFPN, self).__init__()
        self.d = dimension
        self.weights = nn.Parameter(torch.ones(weight_num, dtype=torch.float32), requires_grad=True)
        self.eps = 0.0001
        self.weight_num = weight_num
        self.mode = mode
        if self.mode == 'up':
            self.trans = nn.Upsample(None, 2, 'nearest')
        elif self.mode == 'down':
            self.trans = Conv(c1, c2, k=3, s=2)

    def forward(self, x: torch.Tensor):
        w = self.weights
        w = w / (torch.sum(w, dim=0) + self.eps)
        temp = x[-1].clone()
        temp = self.trans(temp)
        x = [w[i] * x[i] for i in range(self.weight_num - 1)]
        x.append(w[-1] * temp)
        return torch.cat(x, dim=self.d)


class PartialConv(nn.Module):
    """
    PConv
    """

    def __init__(
            self,
            in_chans: int,
            n_div: int,
            forward: str,
    ):
        """
        in_chans is equal to out_chans
        """
        super(PartialConv, self).__init__()
        self.dim_partial = in_chans // n_div
        self.dim_untouched = in_chans - self.dim_partial
        self.partial_conv3 = nn.Conv2d(self.dim_partial, self.dim_partial, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: torch.Tensor):
        # clone is used to some extent for identity, it returns an intermediate variable that can pass gradients.
        # It is equivalent to a deep copy and is suitable for scenarios where a variable is repeatedly used.
        x = x.clone()
        x[:, :self.dim_partial, :, :] = self.partial_conv(x[:, :self.dim_partial, :, :])
        return x

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        # for train/inference
        x1, x2 = torch.split(x, [self.dim_partial, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), dim=1)
        return x


class RoutingFunc(nn.Module):
    """
    example-dependent routing weights r(x) = Sigmoid(GAP(x) * FC)
    A Shadow of Attention Mechanism, like SENet, SKNet, SCConv, but not exactly the same as them，

    note:
    1.In my implementation of CondConv, I used 1x1 conv instead of fc layer mentioned in the paper.
    2.uss Sigmoid instead of Softmax reflect the advantages of multiple experts, while use Softmax, it only
    one expert has an advantage.
    """

    def __init__(
            self,
            in_chans: int,
            experts_num: int,
            drop_ratio: float = 0.,
            is_fc: bool = True,
    ):
        super(RoutingFunc, self).__init__()
        self.dropout = nn.Dropout(drop_ratio)
        self.experts_num = experts_num
        self.is_fc = is_fc
        if is_fc:
            self.attn = nn.Linear(in_chans, experts_num)
        else:
            self.attn = nn.Conv2d(in_chans, experts_num, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        b, _, _, _ = x.shape
        if self.is_fc:
            # (b, c)
            x = x.mean((2, 3), keepdim=False)
        else:
            # (b, c, 1, 1)
            x = x.mean((2, 3), keepdim=True)
        x = self.dropout(x)
        x = self.attn(x)
        x = self.sigmoid(x)
        x = x.view(b, self.experts_num, 1, 1, 1, 1)
        return x

class CondConv(nn.Module):
    """
    CondConv, which also called Dynamic Convolution. plug-and-play module, which can replace convolution in each layer.

    CondConv increases the number of experts --- the number of convolutional kernels, which not only
    increase the capacity of the model but also maintains low latency in inference. CondConv Only add
    a small amount of additional computation compared to mixture of experts which using more depth/width/channel
    to improve capacity and performance of the model.

    The idea of ConvConv is first compute the weight coefficients for each expert --- a, then make decision and
    perform linear combination to generate new kernel, in last, make general convolution by using new kernel.

    note:
    1.the routing weight is sample-dependency, it means the routing weights are different in different samples.
    2.CondConv with the dynamic property through one dimension of kernel space,
    regarding the number of convolution kernel(experts)
    """
    default_act = nn.SiLU()  # default activation
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            stride: int = 1,
            experts_num: int = 8,
            groups: int = 1,
            dilation: int = 1,
            drop_ratio: float = 0.2,
            act: t.Union[t.Callable, nn.Module, bool] = True,
    ):
        super(CondConv, self).__init__()
        self.experts_num = experts_num
        self.out_chans = out_chans
        self.in_chans = in_chans
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.routing = RoutingFunc(in_chans, experts_num, drop_ratio)
        # the standard dim of kernel is (out_chans, in_chans, kernel_size, kernel_size)
        # because new kernel is combined by the num_experts size kernels, so the first dim is num_experts
        self.kernel_weights = nn.Parameter(torch.randn(experts_num, out_chans, in_chans // groups,
                                                        kernel_size, kernel_size), requires_grad=True)

        self.bn = nn.BatchNorm2d(out_chans)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.experts_num):
            nn.init.kaiming_normal_(self.kernel_weights[i], mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor, routing_weights: t.Optional[torch.Tensor] = None):
        b, c, h, w = x.size()
        # compute weights of different experts in different samples
        if routing_weights is None:
            routing_weights = self.routing(x)
        x = x.reshape(1, b * c, h, w)
        # combine weights of all samples, so combined batch_size and out_chans,
        # then use group conv to split different samples
        combined_weights = routing_weights * self.kernel_weights.unsqueeze(dim=0)
        combined_weights = torch.sum(combined_weights, dim=1).view(
            -1, self.in_chans // self.groups, self.kernel_size, self.kernel_size)
        # weight dim is (b * out_chans, in_chans, k, k), bias dim is (b * out_chans), groups is self.groups * b
        # x dim is (1, b * in_chans, h, w)
        y = F.conv2d(x, weight=combined_weights, bias=None, stride=self.stride, dilation=self.dilation,
                     padding=autopad(self.kernel_size, d=self.dilation), groups=self.groups * b)
        # (1, b * out_chans, h, w) -> (b, out_chans, h, w)
        y = y.view(b, self.out_chans, y.size(-2), y.size(-1))
        y = self.bn(y)
        y = self.act(y)
        return y


if __name__ == '__main__':
    pconv = PConv(256, 256, k=3, s=1).to(0)
    conv1 = Conv(256, 256, k=3, s=1).to(0)
    _x = torch.randn((1, 256, 80, 80)).to(0)
    from torchsummary import summary
    from thop import profile
    summary(pconv, (256, 80, 80))
    summary(conv1, (256, 80, 80))

    flops, params = profile(pconv, (_x,))
    print(f"FLOPs={str(flops / 1e9)}G")
    print(f"params={str(params / 1e6)}M")

    flops, params = profile(conv1, (_x,))
    print(f"FLOPs={str(flops / 1e9)}G")
    print(f"params={str(params / 1e6)}M")

