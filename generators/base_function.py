import sys
import math

import torch
from torch import nn
from torch.nn import functional as F

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix

class ExtractionOperation(nn.Module):
    def __init__(self, in_channel, num_label, match_kernel):
        super(ExtractionOperation, self).__init__()
        self.value_conv = EqualConv2d(in_channel, in_channel, match_kernel, 1, match_kernel//2, bias=True)
        self.semantic_extraction_filter = EqualConv2d(in_channel, num_label, match_kernel, 1, match_kernel//2, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.num_label = num_label

    def forward(self, value, recoder):
        key = value
        b,c,h,w = value.shape
        key = self.semantic_extraction_filter(self.feature_norm(key))
        extraction_softmax = self.softmax(key.view(b, -1, h*w)) #bkm
        values_flatten = self.value_conv(value).view(b, -1, h*w)
        neural_textures = torch.einsum('bkm,bvm->bvk', extraction_softmax, values_flatten)
        recoder['extraction_softmax'].insert(0, extraction_softmax)
        recoder['neural_textures'].insert(0, neural_textures)
        return neural_textures, extraction_softmax


    def feature_norm(self, input_tensor):
        input_tensor = input_tensor - input_tensor.mean(dim=1, keepdim=True)
        norm = torch.norm(input_tensor, 2, 1, keepdim=True) + sys.float_info.epsilon
        out = torch.div(input_tensor, norm)
        return out   

class DistributionOperation(nn.Module):
    def __init__(self, num_label, input_dim, match_kernel=3):
        super(DistributionOperation, self).__init__()
        self.semantic_distribution_filter = EqualConv2d(input_dim, num_label, 
                                    kernel_size=match_kernel, 
                                    stride=1,
                                    padding=match_kernel//2)
        self.num_label = num_label

    def forward(self, query, extracted_feature, recoder):
        b,c,h,w = query.shape

        query = self.semantic_distribution_filter(query) 
        query_flatten = query.view(b, self.num_label, -1)
        query_softmax = F.softmax(query_flatten, 1)
        values_q = torch.einsum('bkm,bkv->bvm', query_softmax, extracted_feature.permute(0,2,1))
        attn_out = values_q.view(b,-1,h,w)  
        recoder['semantic_distribution'].append(query)
        return attn_out

class EncoderLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        use_extraction=False,
        num_label=None, 
        match_kernel=None,
        num_extractions=2
    ):
        super().__init__()

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

            stride = 2
            padding = 0

        else:
            self.blur = None
            stride = 1
            padding = kernel_size // 2

        
        self.conv = EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias and not activate,
            )

        self.activate = FusedLeakyReLU(out_channel, bias=bias) if activate else None
        self.use_extraction = use_extraction
        if self.use_extraction:
            self.extraction_operations = nn.ModuleList()
            for _ in range(num_extractions):
                self.extraction_operations.append(
                    ExtractionOperation(
                        out_channel, 
                        num_label, 
                        match_kernel
                    )
                )

    def forward(self, input, recoder=None):
        out = self.blur(input) if self.blur is not None else input
        out = self.conv(out) 
        out = self.activate(out) if self.activate is not None else out
        if self.use_extraction:
            for extraction_operation in self.extraction_operations:
                extraction_operation(out, recoder)
        return out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,    
        use_distribution=True, 
        num_label=16,
        match_kernel=3,
    ):
        super().__init__()
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
            self.conv = EqualTransposeConv2d(
                in_channel, 
                out_channel, 
                kernel_size, 
                stride=2, 
                padding=0, 
                bias=bias and not activate,
            )
        else:
            self.conv = EqualConv2d(
                in_channel, 
                out_channel, 
                kernel_size, 
                stride=1, 
                padding=kernel_size//2,
                bias=bias and not activate,
            )
            self.blur = None

        self.distribution_operation = DistributionOperation(
            num_label, 
            out_channel, 
            match_kernel=match_kernel
        ) if use_distribution else None
        self.activate = FusedLeakyReLU(out_channel, bias=bias) if activate else None
        self.use_distribution = use_distribution

    def forward(self, input, neural_texture=None, recoder=None):
        out = self.conv(input)
        out = self.blur(out) if self.blur is not None else out
        if self.use_distribution and neural_texture is not None:
            out_attn = self.distribution_operation(out, neural_texture, recoder)
            out = (out + out_attn) / math.sqrt(2)     

        out = self.activate(out.contiguous()) if self.activate is not None else out
        
        return out        

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualTransposeConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        weight = self.weight.transpose(0,1)
        out = conv2d_gradfix.conv_transpose2d(
            input,
            weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class ToRGB(nn.Module):
    def __init__(
        self, 
        in_channel, 
        upsample=True, 
        blur_kernel=[1, 3, 3, 1]
        ):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = EqualConv2d(in_channel, 3, 3, stride=1, padding=1)

    def forward(self, input, skip=None):
        out = self.conv(input)
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )        

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k