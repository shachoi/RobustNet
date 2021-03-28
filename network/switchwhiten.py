import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class SwitchWhiten2d(Module):
    """Switchable Whitening.

    Args:
        num_features (int): Number of channels.
        num_pergroup (int): Number of channels for each whitening group.
        sw_type (int): Switchable whitening type, from {2, 3, 5}.
            sw_type = 2: BW + IW
            sw_type = 3: BW + IW + LN
            sw_type = 5: BW + IW + BN + IN + LN
        T (int): Number of iterations for iterative whitening.
        tie_weight (bool): Use the same importance weight for mean and
            covariance or not.
    """

    def __init__(self,
                 num_features,
                 num_pergroup=16,
                 sw_type=2,
                 T=5,
                 tie_weight=False,
                 eps=1e-5,
                 momentum=0.99,
                 affine=True):
        super(SwitchWhiten2d, self).__init__()
        if sw_type not in [2, 3, 5]:
            raise ValueError('sw_type should be in [2, 3, 5], '
                             'but got {}'.format(sw_type))
        assert num_features % num_pergroup == 0
        self.num_features = num_features
        self.num_pergroup = num_pergroup
        self.num_groups = num_features // num_pergroup
        self.sw_type = sw_type
        self.T = T
        self.tie_weight = tie_weight
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        num_components = sw_type

        self.sw_mean_weight = Parameter(torch.ones(num_components))
        if not self.tie_weight:
            self.sw_var_weight = Parameter(torch.ones(num_components))
        else:
            self.register_parameter('sw_var_weight', None)

        if self.affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean',
                             torch.zeros(self.num_groups, num_pergroup, 1))
        self.register_buffer(
            'running_cov',
            torch.eye(num_pergroup).unsqueeze(0).repeat(self.num_groups, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_cov.zero_()
        nn.init.ones_(self.sw_mean_weight)
        if not self.tie_weight:
            nn.init.ones_(self.sw_var_weight)
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def __repr__(self):
        return ('{name}({num_features}, num_pergroup={num_pergroup}, '
                'sw_type={sw_type}, T={T}, tie_weight={tie_weight}, '
                'eps={eps}, momentum={momentum}, affine={affine})'.format(
                    name=self.__class__.__name__, **self.__dict__))

    def forward(self, x):
        N, C, H, W = x.size()
        c, g = self.num_pergroup, self.num_groups

        in_data_t = x.transpose(0, 1).contiguous()
        # g x c x (N x H x W)
        in_data_t = in_data_t.view(g, c, -1)

        # calculate batch mean and covariance
        if self.training:
            # g x c x 1
            mean_bn = in_data_t.mean(-1, keepdim=True)
            in_data_bn = in_data_t - mean_bn
            # g x c x c
            cov_bn = torch.bmm(in_data_bn,
                               in_data_bn.transpose(1, 2)).div(H * W * N)

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * mean_bn.data)
            self.running_cov.mul_(self.momentum)
            self.running_cov.add_((1 - self.momentum) * cov_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            cov_bn = torch.autograd.Variable(self.running_cov)

        mean_bn = mean_bn.view(1, g, c, 1).expand(N, g, c, 1).contiguous()
        mean_bn = mean_bn.view(N * g, c, 1)
        cov_bn = cov_bn.view(1, g, c, c).expand(N, g, c, c).contiguous()
        cov_bn = cov_bn.view(N * g, c, c)

        # (N x g) x c x (H x W)
        in_data = x.view(N * g, c, -1)

        eye = in_data.data.new().resize_(c, c)
        eye = torch.nn.init.eye_(eye).view(1, c, c).expand(N * g, c, c)

        # calculate other statistics
        # (N x g) x c x 1
        mean_in = in_data.mean(-1, keepdim=True)
        x_in = in_data - mean_in
        # (N x g) x c x c
        cov_in = torch.bmm(x_in, torch.transpose(x_in, 1, 2)).div(H * W)
        if self.sw_type in [3, 5]:
            x = x.view(N, -1)
            mean_ln = x.mean(-1, keepdim=True).view(N, 1, 1, 1)
            mean_ln = mean_ln.expand(N, g, 1, 1).contiguous().view(N * g, 1, 1)
            var_ln = x.var(-1, keepdim=True).view(N, 1, 1, 1)
            var_ln = var_ln.expand(N, g, 1, 1).contiguous().view(N * g, 1, 1)
            var_ln = var_ln * eye
        if self.sw_type == 5:
            var_bn = torch.diag_embed(torch.diagonal(cov_bn, dim1=-2, dim2=-1))
            var_in = torch.diag_embed(torch.diagonal(cov_in, dim1=-2, dim2=-1))

        # calculate weighted average of mean and covariance
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.sw_mean_weight)
        if not self.tie_weight:
            var_weight = softmax(self.sw_var_weight)
        else:
            var_weight = mean_weight

        # BW + IW
        if self.sw_type == 2:
            # (N x g) x c x 1
            mean = mean_weight[0] * mean_bn + mean_weight[1] * mean_in
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                self.eps * eye
        # BW + IW + LN
        elif self.sw_type == 3:
            mean = mean_weight[0] * mean_bn + \
                mean_weight[1] * mean_in + mean_weight[2] * mean_ln
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                var_weight[2] * var_ln + self.eps * eye
        # BW + IW + BN + IN + LN
        elif self.sw_type == 5:
            mean = (mean_weight[0] + mean_weight[2]) * mean_bn + \
                (mean_weight[1] + mean_weight[3]) * mean_in + \
                mean_weight[4] * mean_ln
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                var_weight[0] * var_bn + var_weight[1] * var_in + \
                var_weight[4] * var_ln + self.eps * eye

        # perform whitening using Newton's iteration
        Ng, c, _ = cov.size()
        P = torch.eye(c).to(cov).expand(Ng, c, c)
        # reciprocal of trace of covariance
        rTr = (cov * P).sum((1, 2), keepdim=True).reciprocal_()
        cov_N = cov * rTr
        for k in range(self.T):
            P = torch.baddbmm(1.5, P, -0.5, torch.matrix_power(P, 3), cov_N)
        # whiten matrix: the matrix inverse of covariance, i.e., cov^{-1/2}
        wm = P.mul_(rTr.sqrt())

        x_hat = torch.bmm(wm, in_data - mean)
        x_hat = x_hat.view(N, C, H, W)
        if self.affine:
            x_hat = x_hat * self.weight.view(1, self.num_features, 1, 1) + \
                self.bias.view(1, self.num_features, 1, 1)

        return x_hat
