import torch
import torch.nn as nn
from network.mynn import initialize_embedding
import kmeans1d


def make_cov_index_matrix(dim):  # make symmetric matrix for embedding index
    matrix = torch.LongTensor()
    s_index = 0
    for i in range(dim):
        matrix = torch.cat([matrix, torch.arange(s_index, s_index + dim).unsqueeze(0)], dim=0)
        s_index += (dim - (2 + i))
    return matrix.triu(diagonal=1).transpose(0, 1) + matrix.triu(diagonal=1)


class CovMatrix_ISW:
    def __init__(self, dim, relax_denom=0, clusters=50):
        super(CovMatrix_ISW, self).__init__()

        self.dim = dim
        self.i = torch.eye(dim, dim).cuda()

        # print(torch.ones(16, 16).triu(diagonal=1))
        self.reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda()

        # num_off_diagonal = ((dim * dim - dim) // 2)  # number of off-diagonal
        self.num_off_diagonal = torch.sum(self.reversal_i)
        self.num_sensitive = 0
        self.var_matrix = None
        self.count_var_cov = 0
        self.mask_matrix = None
        self.clusters = clusters
        print("num_off_diagonal", self.num_off_diagonal)
        if relax_denom == 0:    # kmeans1d clustering setting for ISW
            print("relax_denom == 0!!!!!")
            print("cluster == ", self.clusters)
            self.margin = 0
        else:                   # do not use
            self.margin = self.num_off_diagonal // relax_denom

    def get_eye_matrix(self):
        return self.i, self.reversal_i

    def get_mask_matrix(self, mask=True):
        if self.mask_matrix is None:
            self.set_mask_matrix()
        return self.i, self.mask_matrix, 0, self.num_sensitive

    def reset_mask_matrix(self):
        self.mask_matrix = None

    def set_mask_matrix(self):
        # torch.set_printoptions(threshold=500000)
        self.var_matrix = self.var_matrix / self.count_var_cov
        var_flatten = torch.flatten(self.var_matrix)

        if self.margin == 0:    # kmeans1d clustering setting for ISW
            clusters, centroids = kmeans1d.cluster(var_flatten, self.clusters) # 50 clusters
            num_sensitive = var_flatten.size()[0] - clusters.count(0)  # 1: Insensitive Cov, 2~50: Sensitive Cov
            print("num_sensitive, centroids =", num_sensitive, centroids)
            _, indices = torch.topk(var_flatten, k=int(num_sensitive))
        else:                   # do not use
            num_sensitive = self.num_off_diagonal - self.margin
            print("num_sensitive = ", num_sensitive)
            _, indices = torch.topk(var_flatten, k=int(num_sensitive))
        mask_matrix = torch.flatten(torch.zeros(self.dim, self.dim).cuda())
        mask_matrix[indices] = 1

        if self.mask_matrix is not None:
            self.mask_matrix = (self.mask_matrix.int() & mask_matrix.view(self.dim, self.dim).int()).float()
        else:
            self.mask_matrix = mask_matrix.view(self.dim, self.dim)
        self.num_sensitive = torch.sum(self.mask_matrix)
        print("Check whether two ints are same", num_sensitive, self.num_sensitive)

        self.var_matrix = None
        self.count_var_cov = 0

        if torch.cuda.current_device() == 0:
            print("Covariance Info: (CXC Shape, Num_Off_Diagonal)", self.mask_matrix.shape, self.num_off_diagonal)
            print("Selective (Sensitive Covariance)", self.num_sensitive)


    def set_variance_of_covariance(self, var_cov):
        if self.var_matrix is None:
            self.var_matrix = var_cov
        else:
            self.var_matrix = self.var_matrix + var_cov
        self.count_var_cov += 1

class CovMatrix_IRW:
    def __init__(self, dim, relax_denom=0):
        super(CovMatrix_IRW, self).__init__()

        self.dim = dim
        self.i = torch.eye(dim, dim).cuda()
        self.reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda()

        self.num_off_diagonal = torch.sum(self.reversal_i)
        if relax_denom == 0:
            print("relax_denom == 0!!!!!")
            self.margin = 0
        else:
            self.margin = self.num_off_diagonal // relax_denom

    def get_mask_matrix(self):
        return self.i, self.reversal_i, self.margin, self.num_off_diagonal
