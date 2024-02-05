import torch
import torch.nn as nn
import typing as t

from torch.nn import init

from ultralytics.nn.modules import Conv

__all__ = ['MV3Block']


class hswish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SEModule(nn.Module):
    def __init__(self, in_chans: int, reduction :int = 4):
        super(SEModule, self).__init__()
        expand_size = max(in_chans // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(in_chans, expand_size, k=1, act=nn.ReLU(inplace=True)),
            nn.Conv2d(expand_size, in_chans, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MV3Block(nn.Module):
    """
    Inverted Residual Block + Linear Bottleneck + SEModule + HardSwish
    """
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            expand_chans: int,
            kernel_size: int = 3,
            stride: int = 1,
            act_layer: t.Optional[nn.Module] = None,
            use_se: bool = True
    ):
        super(MV3Block, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.stride = stride

        if act_layer is None:
            act_layer = nn.ReLU
        self.conv1 = Conv(in_chans, expand_chans, k=1, act=act_layer(inplace=True))
        self.conv2 = Conv(expand_chans, expand_chans, k=kernel_size, s=stride, g=expand_chans,
                          act=act_layer(inplace=True))
        self.se = SEModule(expand_chans) if use_se else nn.Identity()
        self.conv3 = Conv(expand_chans, out_chans, k=1, act=False)
        self.act3 = act_layer(inplace=True)

        self.skip = None
        if stride == 1 and in_chans != out_chans:
            self.skip = nn.Sequential(
                Conv(in_chans, out_chans, k=1, act=False)
            )
        elif stride == 2 and in_chans != out_chans:
            self.skip = nn.Sequential(
                Conv(in_chans, in_chans, k=3, g=in_chans, s=2, act=False),
                Conv(in_chans, out_chans, k=1, act=False)
            )
        elif stride == 2 and in_chans == out_chans:
            self.skip = nn.Sequential(
                Conv(in_chans, out_chans, k=3, g=in_chans, s=2, act=False)
            )
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.conv3(out)

        if self.skip is not None:
            identity = self.skip(identity)
        return self.act3(out + identity)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)






