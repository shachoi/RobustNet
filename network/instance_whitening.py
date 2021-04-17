import torch
import torch.nn as nn


class InstanceWhitening(nn.Module):

    def __init__(self, dim):
        super(InstanceWhitening, self).__init__()
        self.coloring = coloring
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):

        x = self.instance_standardization(x)
        w = x

        return x, w
