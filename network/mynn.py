"""
Custom Norm wrappers to enable sync BN, regular BN and for weight initialization
"""
import torch.nn as nn
import torch
from config import cfg

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = getattr(cfg.MODEL, 'BNFUNC')
    normalization_layer = layer(in_channels)
    return normalization_layer


def freeze_weights(*models):
    for model in models:
        for k in model.parameters():
            k.requires_grad = False

def unfreeze_weights(*models):
    for model in models:
        for k in model.parameters():
            k.requires_grad = True

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or \
                isinstance(module, nn.GroupNorm) or isinstance(module, nn.SyncBatchNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def initialize_embedding(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                module.weight.data.zero_() #original



def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)

def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            print("Skipped loading parameter", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

def Zero_Masking(input_tensor, mask_org):
    output = input_tensor.clone()
    output.mul_(mask_org)
    return output

def RandomPosZero_Masking(input_tensor, p=0.5):
    output = input_tensor.clone()
    noise_b = input_tensor.new().resize_(input_tensor.size(0), 1, input_tensor.size(2), input_tensor.size(3))
    noise_u = input_tensor.new().resize_(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), input_tensor.size(3))
    noise_b.bernoulli_(1 - p)
    noise_b = noise_b.expand_as(input_tensor)
    output.mul_(noise_b)
    return output

def RandomVal_Masking(input_tensor, mask_org):
    output = input_tensor.clone()
    noise_u = input_tensor.new().resize_(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), input_tensor.size(3))
    mask = (mask_org==0).type(input_tensor.type())
    mask = mask.expand_as(input_tensor)
    mask = torch.mul(mask, noise_u.uniform_(torch.min(input_tensor).item(), torch.max(input_tensor).item()))
    mask_org = mask_org.expand_as(input_tensor)
    output.mul_(mask_org)
    output.add_(mask)
    return output

def RandomPosVal_Masking(input_tensor, p=0.5):
    output = input_tensor.clone()
    noise_b = input_tensor.new().resize_(input_tensor.size(0), 1, input_tensor.size(2), input_tensor.size(3))
    noise_u = input_tensor.new().resize_(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), input_tensor.size(3))
    mask = noise_b.bernoulli_(1 - p)
    mask = (mask==0).type(input_tensor.type())
    mask = mask.expand_as(input_tensor)
    mask = torch.mul(mask, noise_u.uniform_(torch.min(input_tensor).item(), torch.max(input_tensor).item()))
    noise_b = noise_b.expand_as(input_tensor)
    output.mul_(noise_b)
    output.add_(mask)
    return output

def masking(input_tensor, p=0.5):
    output = input_tensor.clone()
    noise_b = input_tensor.new().resize_(input_tensor.size(0), 1, input_tensor.size(2), input_tensor.size(3))
    noise_u = input_tensor.new().resize_(input_tensor.size(0), 1, input_tensor.size(2), input_tensor.size(3))
    mask = noise_b.bernoulli_(1 - p)
    mask = (mask==0).type(input_tensor.type())
    mask.mul_(noise_u.uniform_(torch.min(input_tensor).item(), torch.max(input_tensor).item()))
    # mask.mul_(noise_u.uniform_(5, 10))
    noise_b = noise_b.expand_as(input_tensor)
    mask = mask.expand_as(input_tensor)
    output.mul_(noise_b)
    output.add_(mask)
    return output
