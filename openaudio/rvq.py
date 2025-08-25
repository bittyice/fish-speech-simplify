import math
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from typing import Union

import numpy as np
from einops import rearrange
from torch.nn.utils import weight_norm


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = weight_norm(nn.Conv1d(input_dim, codebook_dim, kernel_size=1))
        self.out_proj = weight_norm(nn.Conv1d(codebook_dim, input_dim, kernel_size=1))
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zeros",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right
    before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class CausalConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
        padding=None,
    ):
        super(CausalConvNet, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def forward(self, x):
        pad = self.padding
        extra_padding = get_extra_padding_for_conv1d(
            x, self.kernel_size, self.stride, pad
        )
        x = pad1d(x, (pad, extra_padding), mode="constant", value=0)
        return self.conv(x).contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


class CausalTransConvNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, stride=1, padding=None
    ):
        super(CausalTransConvNet, self).__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


# ConvNeXt Block copied from https://github.com/fishaudio/fish-diffusion/blob/main/fish_diffusion/modules/convnext.py
class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        kernel_size (int): Kernel size for depthwise conv. Default: 7.
        dilation (int): Dilation for depthwise conv. Default: 1.
    """  # noqa: E501

    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        convnet_type = CausalConvNet
        self.dwconv = convnet_type(
            dim,
            dim,
            kernel_size=kernel_size,
            # padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim,
            dilation=dilation,
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, int(mlp_ratio * dim)
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, apply_residual: bool = True):
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)

        if apply_residual:
            x = input + x

        return x


@dataclass
class VQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    semantic_distill_z: torch.Tensor | None = None


class DownsampleResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        n_codebooks: int = 9,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 4096,
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: tuple[int] | None = None,
        pre_module: nn.Module | None = None,
        post_module: nn.Module | None = None
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        self.semantic_quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=1,
            codebook_size=semantic_codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=0.0,
        )

        self.quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        convnet_type = CausalConvNet
        transconvnet_type = CausalTransConvNet

        self.downsample = nn.Sequential(
            *[
                nn.Sequential(
                    convnet_type(
                        all_dims[idx],
                        all_dims[idx + 1],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    transconvnet_type(
                        all_dims[idx + 1],
                        all_dims[idx],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx]),
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        )
        self.apply(self._init_weights)
        self.pre_module = pre_module if pre_module is not None else nn.Identity()
        self.post_module = post_module if post_module is not None else nn.Identity()
        self.semantic_predictor_module = nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, z, n_quantizers: int = None, semantic_len: torch.Tensor = None, **kwargs
    ):
        # z: (B, D, T)
        original_shape = z.shape
        if semantic_len is None:
            semantic_len = torch.LongTensor([z.shape[-1]])
        z = self.downsample(z)
        z = self.pre_module(z)  # B, T, D

        (
            semantic_z,
            semantic_codes,
            semantic_latents,
            semantic_commitment_loss,
            semantic_codebook_loss,
        ) = self.semantic_quantizer(z)
        residual_z = z - semantic_z
        residual_z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            residual_z, n_quantizers=n_quantizers
        )
        z = semantic_z + residual_z
        commitment_loss = commitment_loss + semantic_commitment_loss
        codebook_loss = codebook_loss + semantic_codebook_loss
        codes = torch.cat([semantic_codes, codes], dim=1)
        latents = torch.cat([semantic_latents, latents], dim=1)
        z = self.post_module(z)
        z = self.upsample(z)

        # Pad or crop z to match original shape
        diff = original_shape[-1] - z.shape[-1]
        right = 0
        left = abs(diff) - right

        if diff > 0:
            z = F.pad(z, (left, right))
        elif diff < 0:
            z = z[..., left:]

        results = VQResult(
            z=z,
            codes=codes,
            latents=latents,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
        )

        return results

    # 量化操作
    def encode(
        self, z, n_quantizers: int = None, semantic_len: torch.Tensor = None, **kwargs
    ):
        if semantic_len is None:
            semantic_len = torch.LongTensor([z.shape[-1]])
        z = self.downsample(z)
        z = self.pre_module(z)  # B, T, D

        # 量化操作
        (
            semantic_z,
            semantic_codes,
            semantic_latents,
            semantic_commitment_loss,
            semantic_codebook_loss,
        ) = self.semantic_quantizer(z)
        residual_z = z - semantic_z
        residual_z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            residual_z, n_quantizers=n_quantizers
        )
        z = semantic_z + residual_z
        commitment_loss = commitment_loss + semantic_commitment_loss
        codebook_loss = codebook_loss + semantic_codebook_loss
        codes = torch.cat([semantic_codes, codes], dim=1)

        return codes

    # 反量化操作
    def decode(self, indices: torch.Tensor):
        # indices = rearrange(indices, "b (g r) l -> g b l r", g=self.residual_fsq.groups)

        new_indices = torch.zeros_like(indices)
        new_indices[:, 0] = torch.clamp(
            indices[:, 0], max=self.semantic_quantizer.codebook_size - 1
        )
        new_indices[:, 1:] = torch.clamp(
            indices[:, 1:], max=self.quantizer.codebook_size - 1
        )

        z_q_semantic = self.semantic_quantizer.from_codes(new_indices[:, :1])[0]
        z_q_residual = self.quantizer.from_codes(new_indices[:, 1:])[0]
        z_q = z_q_semantic + z_q_residual
        z_q = self.post_module(z_q)
        z_q = self.upsample(z_q)
        return z_q


if __name__ == "__main__":
    rvq = DownsampleResidualVectorQuantize(
        input_dim=512,
        n_codebooks=8,
        codebook_dim=8,
        codebook_size=1024,
        quantizer_dropout=0.5,
        downsample_factor=[2, 2],
    )
    rvq.eval()
    x = torch.randn(2, 512, 442)

    result = rvq(x)
    print(rvq)
    print(result.latents.shape, result.codes.shape, result.z.shape)

    # y = rvq.from_codes(result.codes)
    # print(y[0].shape)

    # y = rvq.from_latents(

    result1 = rvq(x[:, :, :40])
    print(result1.latents.shape, result1.codes.shape, result1.z.shape)

    assert torch.allclose(result.z[:, :, :40], result1.z, atol=1e-8)
    print("Success")

