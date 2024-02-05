import torch
import typing as t

from timm.layers import DropPath
from torch import nn
from ultralytics.nn.modules import Conv

__all__ = ['PatchEmbedding', 'PoolFormerBlocks']

class PatchEmbedding(nn.Module):
    """
    Patch Embedding and Patch Merging of PoolFormer
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            stride: int,
    ):
        super(PatchEmbedding, self).__init__()
        self.conv = Conv(in_chans, out_chans, kernel_size, stride, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Pooling(nn.Module):

    def __init__(self, pool_size: int = 3):
        super(Pooling, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) - x


class MLP(nn.Module):

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            hidden_chans: int,
            act_layer: t.Optional[nn.Module] = None,
            drop_ratio: float = 0.
    ):
        super(MLP, self).__init__()
        if act_layer is None:
            act_layer = nn.GELU
        self.fc1 = Conv(in_chans, hidden_chans, 1, act=act_layer())
        self.fc2 = Conv(hidden_chans, out_chans, 1, act=False)
        self.drop = nn.Dropout(drop_ratio)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.fc1(x)))


class PoolFormerBlock(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            pool_size: int = 3,
            mlp_ratio: int = 4,
            drop_ratio: float = 0.,
            drop_path: float = 0.,
            act_layer: t.Optional[nn.Module] = nn.GELU,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,

    ):
        super(PoolFormerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(1, embed_dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = nn.GroupNorm(1, embed_dim)
        self.channel_mixer = MLP(
            in_chans=embed_dim,
            out_chans=embed_dim,
            hidden_chans=int(mlp_ratio * embed_dim),
            act_layer=act_layer,
            drop_ratio=drop_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dim, )), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dim, )), requires_grad=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x))
            )
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.channel_mixer(self.norm2(x))
            )
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.channel_mixer(self.norm2(x)))
        return x


class PoolFormerBlocks(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            depth: int,
            pool_size: int = 3,
            mlp_ratio: int = 4,
            drop_ratio: float = 0.,
            drop_paths: t.Optional[t.List[float]] = None,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
            act_layer: t.Optional[nn.Module] = nn.GELU,
    ):
        super(PoolFormerBlocks, self).__init__()
        blocks = []
        for i in range(depth):
            blocks.append(PoolFormerBlock(
                embed_dim=embed_dim,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
                drop_ratio=drop_ratio,
                drop_path=drop_paths[i],
                act_layer=act_layer,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class PoolFormer(nn.Module):
    """
    Using downsampling to increase receptive field and replace the self attention mechanism in token_mixer
    """

    def __init__(
            self,
            layers: t.List[int],
            embed_dims: t.List[int],
            mlp_ratios: int = 4,
            pool_size: int = 3,
            act_layer: nn.Module = nn.GELU,
            in_patch_size: int = 7,
            in_stride: int = 4,
            down_patch_size: int = 3,
            down_stride: int = 2,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
            num_classes: int = 1000,
            classifier: bool = False,
    ):
        super(PoolFormer, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        self.stem = PatchEmbedding(in_chans=3, out_chans=embed_dims[0], kernel_size=in_patch_size, stride=in_stride)
        network = []
        lens = len(layers)
        for i in range(lens):
            network.append(PoolFormerBlocks(
                embed_dim=embed_dims[i],
                depth=layers[i],
                pool_size=pool_size,
                mlp_ratio=mlp_ratios,
                drop_ratio=drop_rate,
                drop_paths=dpr[sum(layers[: i]): sum(layers[: i + 1])],
                act_layer=act_layer,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value
            ))
            if i != lens - 1:
                network.append(PatchEmbedding(
                    embed_dims[i],
                    embed_dims[i + 1],
                    down_patch_size,
                    down_stride
                ))
        self.network = nn.Sequential(*network)
        self.classifier = classifier
        if classifier:
            self.norm = nn.GroupNorm(1, embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input embedding
        x = self.stem(x)
        # through backbone
        x = self.network(x)

        # classify
        if self.classifier:
            x = self.norm(x)
            cls_out = self.head(x.mean([-2, -1]))
        # for image classification
            return cls_out
        else:
            return x
