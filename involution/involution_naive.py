import torch.nn as nn
from mmcv.cnn import ConvModule


class Involution(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride,
                 dilation=1):
        super(Involution, self).__init__()
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

        self.unfold = nn.Unfold(kernel_size, dilation,
                                (self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.channels}, out_channels={self.channels}, ' \
               f'kernel_size={self.kernel_size}, stride={self.stride}, dilation={self.dilation})'
