"""
# Code Adapted from:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import network.mynn as mynn
from network.sync_switchwhiten import SyncSwitchWhiten2d
from network.instance_whitening import InstanceWhitening


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnet_adapt101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Basic Block for Resnet
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, iw=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = mynn.Norm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = mynn.Norm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.iw = iw
        if self.iw == 1:
            self.instance_norm_layer = InstanceWhitening(planes * self.expansion)
            self.relu = nn.ReLU(inplace=False)
        elif self.iw == 2:
            self.instance_norm_layer = InstanceWhitening(planes * self.expansion)
            self.relu = nn.ReLU(inplace=False)
        elif self.iw == 3:
            self.instance_norm_layer = nn.InstanceNorm2d(planes * self.expansion, affine=False)
            self.relu = nn.ReLU(inplace=True)
        elif self.iw == 4:
            self.instance_norm_layer = nn.InstanceNorm2d(planes * self.expansion, affine=True)
            self.relu = nn.ReLU(inplace=True)
        elif self.iw == 5:
            self.instance_norm_layer = SyncSwitchWhiten2d(planes * self.expansion,
                                                          num_pergroup=16,
                                                          sw_type=2,
                                                          T=5,
                                                          tie_weight=False,
                                                          eps=1e-5,
                                                          momentum=0.99,
                                                          affine=True)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x_tuple):
        if len(x_tuple) == 2:
            w_arr = x_tuple[1]
            x = x_tuple[0]
        else:
            print("error!!!")
            return

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.iw >= 1:
            if self.iw == 1 or self.iw == 2:
                out, w = self.instance_norm_layer(out)
                w_arr.append(w)
            else:
                out = self.instance_norm_layer(out)

        out = self.relu(out)

        return [out, w_arr]


class Bottleneck(nn.Module):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, iw=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = mynn.Norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = mynn.Norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = mynn.Norm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.iw = iw
        if self.iw == 1:
            self.instance_norm_layer = InstanceWhitening(planes * self.expansion)
            self.relu = nn.ReLU(inplace=False)
        elif self.iw == 2:
            self.instance_norm_layer = InstanceWhitening(planes * self.expansion)
            self.relu = nn.ReLU(inplace=False)
        elif self.iw == 3:
            self.instance_norm_layer = nn.InstanceNorm2d(planes * self.expansion, affine=False)
            self.relu = nn.ReLU(inplace=True)
        elif self.iw == 4:
            self.instance_norm_layer = nn.InstanceNorm2d(planes * self.expansion, affine=True)
            self.relu = nn.ReLU(inplace=True)
        elif self.iw == 5:
            self.instance_norm_layer = SyncSwitchWhiten2d(planes * self.expansion,
                                                          num_pergroup=16,
                                                          sw_type=2,
                                                          T=5,
                                                          tie_weight=False,
                                                          eps=1e-5,
                                                          momentum=0.99,
                                                          affine=True)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x_tuple):
        if len(x_tuple) == 2:
            w_arr = x_tuple[1]
            x = x_tuple[0]
        else:
            print("error!!!")
            return

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.iw >= 1:
            if self.iw == 1 or self.iw == 2:
                out, w = self.instance_norm_layer(out)
                w_arr.append(w)
            else:
                out = self.instance_norm_layer(out)

        out = self.relu(out)

        return [out, w_arr]


class ResNet3X3(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, wt_layer=None, num_classes=1000):
        self.inplanes = 128
        super(ResNet3X3, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = mynn.Norm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                                bias=False)

        if wt_layer[0] == 1:
            self.bn1 = InstanceWhitening(64)
            self.relu1 = nn.ReLU(inplace=False)
        elif wt_layer[0] == 2:
            self.bn1 = InstanceWhitening(64)
            self.relu1 = nn.ReLU(inplace=False)
        elif wt_layer[0] == 3:
            self.bn1 = nn.InstanceNorm2d(64, affine=False)
            self.relu1 = nn.ReLU(inplace=True)
        elif wt_layer[0] == 4:
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
            self.relu1 = nn.ReLU(inplace=True)
        elif wt_layer[0] == 5:
            self.bn1 = SyncSwitchWhiten2d(64,
                                          num_pergroup=16,
                                          sw_type=2,
                                          T=5,
                                          tie_weight=False,
                                          eps=1e-5,
                                          momentum=0.99,
                                          affine=True)
            self.relu1 = nn.ReLU(inplace=True)
        else:
            self.bn1 = mynn.Norm2d(64)
            self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        if wt_layer[1] == 1:
            self.bn2 = InstanceWhitening(64)
            self.relu2 = nn.ReLU(inplace=False)
        elif wt_layer[1] == 2:
            self.bn2 = InstanceWhitening(64)
            self.relu2 = nn.ReLU(inplace=False)
        elif wt_layer[1] == 3:
            self.bn2 = nn.InstanceNorm2d(64, affine=False)
            self.relu2 = nn.ReLU(inplace=True)
        elif wt_layer[1] == 4:
            self.bn2 = nn.InstanceNorm2d(64, affine=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif wt_layer[1] == 5:
            self.bn2 = SyncSwitchWhiten2d(64,
                                          num_pergroup=16,
                                          sw_type=2,
                                          T=5,
                                          tie_weight=False,
                                          eps=1e-5,
                                          momentum=0.99,
                                          affine=True)
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.bn2 = mynn.Norm2d(64)
            self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,
                               bias=False)
        if wt_layer[2] == 1:
            self.bn3 = InstanceWhitening(self.inplanes)
            self.relu3 = nn.ReLU(inplace=False)
        elif wt_layer[2] == 2:
            self.bn3 = InstanceWhitening(self.inplanes)
            self.relu3 = nn.ReLU(inplace=False)
        elif wt_layer[2] == 3:
            self.bn3 = nn.InstanceNorm2d(self.inplanes, affine=False)
            self.relu3 = nn.ReLU(inplace=True)
        elif wt_layer[2] == 4:
            self.bn3 = nn.InstanceNorm2d(self.inplanes, affine=True)
            self.relu3 = nn.ReLU(inplace=True)
        elif wt_layer[2] == 5:
            self.bn3 = SyncSwitchWhiten2d(self.inplanes,
                                          num_pergroup=16,
                                          sw_type=2,
                                          T=5,
                                          tie_weight=False,
                                          eps=1e-5,
                                          momentum=0.99,
                                          affine=True)
            self.relu3 = nn.ReLU(inplace=True)
        else:
            self.bn3 = mynn.Norm2d(self.inplanes)
            self.relu3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], wt_layer=wt_layer[3])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, wt_layer=wt_layer[4])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, wt_layer=wt_layer[5])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, wt_layer=wt_layer[6])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.wt_layer = wt_layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, wt_layer=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                mynn.Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, iw=0))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                iw=0 if (wt_layer > 0 and index < blocks - 1) else wt_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        w_arr = []
        x_size = x.size()  # 800

        x = self.conv1(x)
        if self.wt_layer[0] == 1 or self.wt_layer[0] == 2:
            x, w = self.bn1(x)
            w_arr.append(w)
        else:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        if self.wt_layer[1] == 1 or self.wt_layer[1] == 2:
            x, w = self.bn2(x)
            w_arr.append(w)
        else:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        if self.wt_layer[2] == 1 or self.wt_layer[2] == 2:
            x, w = self.bn3(x)
            w_arr.append(w)
        else:
            x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool(x)

        x_tuple = self.layer1([x, w_arr])  # 400
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        aux_out = x_tuple[0]
        x_tuple = self.layer4(x_tuple)  # 100

        x = x_tuple[0]
        w_arr = x_tuple[1]
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x

class ResNet(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, wt_layer=None, num_classes=1000):
        self.inplanes = 64
        # self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if wt_layer[2] == 1:
            self.bn1 = InstanceWhitening(64)
            self.relu = nn.ReLU(inplace=False)
        elif wt_layer[2] == 2:
            self.bn1 = InstanceWhitening(64)
            self.relu = nn.ReLU(inplace=False)
        elif wt_layer[2] == 3:
            self.bn1 = nn.InstanceNorm2d(64, affine=False)
            self.relu = nn.ReLU(inplace=True)
        elif wt_layer[2] == 4:
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
            self.relu = nn.ReLU(inplace=True)
        elif wt_layer[2] == 5:
            self.bn1 = SyncSwitchWhiten2d(self.inplanes,
                                          num_pergroup=16,
                                          sw_type=2,
                                          T=5,
                                          tie_weight=False,
                                          eps=1e-5,
                                          momentum=0.99,
                                          affine=True)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.bn1 = mynn.Norm2d(64)
            self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], wt_layer=wt_layer[3])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, wt_layer=wt_layer[4])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, wt_layer=wt_layer[5])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, wt_layer=wt_layer[6])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.wt_layer = wt_layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, wt_layer=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                mynn.Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, iw=0))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                iw=0 if (wt_layer > 0 and index < blocks - 1) else wt_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        w_arr = []
        x_size = x.size()  # 800

        x = self.conv1(x)
        if self.wt_layer[2] == 1 or self.wt_layer[2] == 2:
            x, w = self.bn1(x)
            w_arr.append(w)
        else:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_tuple = self.layer1([x, w_arr])  # 400
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        aux_out = x_tuple[0]
        x_tuple = self.layer4(x_tuple)  # 100

        x = x_tuple[0]
        w_arr = x_tuple[1]

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x



def resnet18(pretrained=True, wt_layer=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if wt_layer is None:
        wt_layer = [0, 0, 0, 0, 0, 0, 0]
    model = ResNet(BasicBlock, [2, 2, 2, 2], wt_layer=wt_layer, **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        print("########### pretrained ##############")
        mynn.forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=True, wt_layer=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if wt_layer is None:
        wt_layer = [0, 0, 0, 0, 0, 0, 0]
    model = ResNet(Bottleneck, [3, 4, 6, 3], wt_layer=wt_layer, **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        print("########### pretrained ##############")
        mynn.forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=True, wt_layer=None, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param pretrained:
    """
    if wt_layer is None:
        wt_layer = [0, 0, 0, 0, 0, 0, 0]
    model = ResNet3X3(Bottleneck, [3, 4, 23, 3], wt_layer=wt_layer, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        print("########### pretrained ##############")
        # model.load_state_dict(torch.load('./pretrained/resnet101-imagenet.pth', map_location="cpu"))
        mynn.forgiving_state_restore(model, torch.load('./pretrained/resnet101-imagenet.pth', map_location="cpu"))
    return model


def resnet_adapt101(args, pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        args: arguments that contain adapt_layer information
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param pretrained:
    """
    model = ResNet3X3(args, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        print("########### pretrained ##############")
        model.load_state_dict(torch.load('./pretrained/resnet_adapt101-imagenet.pth', map_location="cpu"))
        # mynn.forgiving_state_restore(model, torch.load('./pretrained/resnet101-imagenet.pth', map_location="cpu"))
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
