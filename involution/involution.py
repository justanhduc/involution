import os
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn.modules.utils import _pair

sources = ('involution_cuda.cpp', 'involution_kernel.cu')
sources = [os.path.join(os.path.dirname(__file__), f) for f in sources]
involution_cuda = load(name='involution_cuda', sources=sources)


class _Involution(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 6 and weight.is_cuda
        with torch.cuda.device_of(input):
            output = involution_cuda.involution_forward(input, weight, stride, padding, dilation)

        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda and grad_output.is_contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        grad_input, grad_weight = None, None
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = involution_cuda.involution_backward_grad_input(grad_output, weight, input.shape,
                                                                            stride, padding, dilation)
            if ctx.needs_input_grad[1]:
                grad_weight = involution_cuda.involution_backward_grad_weight(grad_output, input, weight.shape,
                                                                              stride, padding, dilation)

        return grad_input, grad_weight, None, None, None


def _involution_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """ involution kernel
    """
    assert input.size(0) == weight.size(0)
    assert input.size(-2) // stride == weight.size(-2)
    assert input.size(-1) // stride == weight.size(-1)
    if input.is_cuda:
        out = _Involution.apply(input, weight, _pair(stride), _pair(padding), _pair(dilation))
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        raise NotImplementedError
    return out


class Involution(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size ** 2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        out = _involution_cuda(x, weight, stride=self.stride,
                               padding=(self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1) // 2,
                               dilation=self.dilation)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.channels}, out_channels={self.channels}, ' \
               f'kernel_size={self.kernel_size}, stride={self.stride}, dilation={self.dilation})'
