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


def instance_whitening_loss(f_map, eye, mask_matrix, margin, num_remove_cov):
    f_cor = get_covariance_matrix(f_map, eye=eye)
    f_cor_masked = f_cor * mask_matrix

    off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1,2), keepdim=True) - margin # B X 1 X 1
    off_diag_reg = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
    off_diag_reg = torch.sum(off_diag_reg) / B

    return off_diag_reg


def get_covariance_matrix(f_map, eye=None):
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (self.eps * eye)  # C X C / HW

    return f_cor