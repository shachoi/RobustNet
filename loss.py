"""
Loss.py
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from config import cfg


def get_loss(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """
    if args.cls_wt_loss:
        ce_weight = torch.Tensor([0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                    1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                    1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
    else:
        ce_weight = None

    if args.img_wt_loss:
        criterion = ImageBasedCrossEntropyLoss2d(
            classes=datasets.num_classes, size_average=True,
            ignore_index=datasets.ignore_label,
            upper_bound=args.wt_bound).cuda()
    elif args.jointwtborder:
        criterion = ImgWtLossSoftNLL(classes=datasets.num_classes,
                                     ignore_index=datasets.ignore_label,
                                     upper_bound=args.wt_bound).cuda()
    else:
        print("standard cross entropy")
        criterion = nn.CrossEntropyLoss(weight=ce_weight, reduction='mean',
                                       ignore_index=datasets.ignore_label).cuda()

    criterion_val = nn.CrossEntropyLoss(reduction='mean',
                                       ignore_index=datasets.ignore_label).cuda()
    return criterion, criterion_val

def get_loss_by_epoch(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """

    if args.img_wt_loss:
        criterion = ImageBasedCrossEntropyLoss2d(
            classes=datasets.num_classes, size_average=True,
            ignore_index=datasets.ignore_label,
            upper_bound=args.wt_bound).cuda()
    elif args.jointwtborder:
        criterion = ImgWtLossSoftNLL_by_epoch(classes=datasets.num_classes,
                                     ignore_index=datasets.ignore_label,
                                     upper_bound=args.wt_bound).cuda()
    else:
        criterion = CrossEntropyLoss2d(size_average=True,
                                       ignore_index=datasets.ignore_label).cuda()

    criterion_val = CrossEntropyLoss2d(size_average=True,
                                       weight=None,
                                       ignore_index=datasets.ignore_label).cuda()
    return criterion, criterion_val


def get_loss_aux(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """
    if args.cls_wt_loss:
        ce_weight = torch.Tensor([0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
    else:
        ce_weight = None

    print("standard cross entropy")
    criterion = nn.CrossEntropyLoss(weight=ce_weight, reduction='mean',
                                    ignore_index=datasets.ignore_label).cuda()

    return criterion

def get_loss_bcelogit(args):
    if args.cls_wt_loss:
        pos_weight = torch.Tensor([0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
    else:
        pos_weight = None
    print("standard bce with logit cross entropy")
    criterion = nn.BCEWithLogitsLoss(reduction='mean').cuda()

    return criterion

def weighted_binary_cross_entropy(output, target):
        
    weights = torch.Tensor([0.1, 0.9])
        
    loss = weights[1] * (target * torch.log(output)) + \
            weights[0] * ((1 - target) * torch.log(1 - output))

    return torch.neg(torch.mean(loss))


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __call__(self, in0, in1):
        return torch.sum(torch.abs(in0 - in1), dim=1, keepdim=True)


class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight=weight, reduction='mean', ignore_index=ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            loss += self.nll_loss(self.logsoftmax(inputs[i].unsqueeze(0)),
                                  targets[i].unsqueeze(0))
        return loss



class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss(weight=weight, reduction='mean', ignore_index=ignore_index)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.weight = weight

    def forward(self, inputs, targets):
        return self.nll_loss(self.logsoftmax(inputs), targets)

def customsoftmax(inp, multihotmask):
    """
    Custom Softmax
    """
    soft = F.softmax(inp, dim=1)
    # This takes the mask * softmax ( sums it up hence summing up the classes in border
    # then takes of summed up version vs no summed version
    return torch.log(
        torch.max(soft, (multihotmask * (soft * multihotmask).sum(1, keepdim=True)))
    )

class ImgWtLossSoftNLL(nn.Module):
    """
    Relax Loss
    """

    def __init__(self, classes, ignore_index=255, weights=None, upper_bound=1.0,
                 norm=False):
        super(ImgWtLossSoftNLL, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm
        self.batch_weights = cfg.BATCH_WEIGHTING

    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:-1]

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """
        if (cfg.REDUCE_BORDER_ITER != -1 and cfg.ITER > cfg.REDUCE_BORDER_ITER):
            border_weights = 1 / border_weights
            target[target > 1] = 1

        loss_matrix = (-1 / border_weights *
                        (target[:, :-1, :, :].float() *
                        class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                        customsoftmax(inputs, target[:, :-1, :, :].float())).sum(1)) * \
                        (1. - mask.float())

            # loss_matrix[border_weights > 1] = 0
        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] - mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target):
        weights = target[:, :-1, :, :].sum(1).float()
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1

        loss = 0
        target_cpu = target.data.cpu().numpy()

        if self.batch_weights:
            class_weights = self.calculate_weights(target_cpu)

        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                class_weights = self.calculate_weights(target_cpu[i])
            loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
                                          target[i].unsqueeze(0),
                                          class_weights=torch.Tensor(class_weights).cuda(),
                                          border_weights=weights[i], mask=ignore_mask[i])

        loss = loss / inputs.shape[0]
        return loss

class ImgWtLossSoftNLL_by_epoch(nn.Module):
    """
    Relax Loss
    """

    def __init__(self, classes, ignore_index=255, weights=None, upper_bound=1.0,
                 norm=False):
        super(ImgWtLossSoftNLL_by_epoch, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm
        self.batch_weights = cfg.BATCH_WEIGHTING
        self.fp16 = False


    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:-1]

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """
        if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
            border_weights = 1 / border_weights
            target[target > 1] = 1
        if self.fp16:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].half() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].half())).sum(1)) * \
                          (1. - mask.half())
        else:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].float() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].float())).sum(1)) * \
                          (1. - mask.float())

            # loss_matrix[border_weights > 1] = 0
        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] - mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target):
        if self.fp16:
            weights = target[:, :-1, :, :].sum(1).half()
        else:
            weights = target[:, :-1, :, :].sum(1).float()
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1

        loss = 0
        target_cpu = target.data.cpu().numpy()

        if self.batch_weights:
            class_weights = self.calculate_weights(target_cpu)

        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                class_weights = self.calculate_weights(target_cpu[i])
            loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
                                          target[i].unsqueeze(0),
                                          class_weights=torch.Tensor(class_weights).cuda(),
                                          border_weights=weights, mask=ignore_mask[i])

        return loss
