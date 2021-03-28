import torch
import torch.nn as nn


class ColoringTransform(nn.Module):

    def __init__(self, dim):
        super(ColoringTransform, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, kernel_size=1, stride=1,
                                    padding=0, bias=True)

    def forward(self, x):

        x = self.conv(x)

        return x


class ChannelAffineTransform(nn.Module):

    def __init__(self, dim):
        super(ChannelAffineTransform, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dim, 1, 1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)

    def forward(self, x):

        x = (x * self.gamma) + self.beta

        return x


class BatchWhitening(nn.Module):

    def __init__(self, dim, scale=False, coloring=False):
        super(BatchWhitening, self).__init__()
        self.coloring = coloring
        self.batch_standardization = nn.SyncBatchNorm(dim, affine=False)
        if self.coloring:
            self.coloring_layer = ColoringTransform(dim)

    def forward(self, x):

        x = self.batch_standardization(x)
        w = x
        c = None
        if self.coloring:
            c = x.detach().clone()
            x = self.coloring_layer(x)
            c = self.coloring_layer(c)

        return x, w, c


class InstanceWhitening(nn.Module):

    def __init__(self, dim, coloring=False):    # Coloring is not used
        super(InstanceWhitening, self).__init__()
        self.coloring = coloring
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)
        if self.coloring:
            self.coloring_layer = ColoringTransform(dim)

    def forward(self, x):

        x = self.instance_standardization(x)
        w = x
        c = None
        if self.coloring:   # Coloring is not used
            c = x.detach().clone()
            x = self.coloring_layer(x)
            c = self.coloring_layer(c)

        return x, w, c
