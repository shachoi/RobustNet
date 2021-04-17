from torch import nn
from torch import Tensor
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Callable, Any, Optional, List
from network.instance_whitening import InstanceWhitening
from network.mynn import forgiving_state_restore

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        iw: int = 0,
    ) -> None:

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.iw = iw

        if iw == 1:
            instance_norm_layer = InstanceWhitening(out_planes)
        elif iw == 2:
            instance_norm_layer = InstanceWhitening(out_planes)
        elif iw == 3:
            instance_norm_layer = nn.InstanceNorm2d(out_planes, affine=False)
        elif iw == 4:
            instance_norm_layer = nn.InstanceNorm2d(out_planes, affine=True)
        else:
            instance_norm_layer = nn.Sequential()

        super(ConvBNReLU, self).__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                norm_layer(out_planes),
                nn.ReLU6(inplace=True),
                instance_norm_layer
            )


    def forward(self, x_tuple):
        if len(x_tuple) == 2:
            w_arr = x_tuple[1]
            x = x_tuple[0]
        else:
            print("error in BN forward path")
            return

        for i, module in enumerate(self):
            if i == len(self) - 1:
                if self.iw >= 1:
                    if self.iw == 1 or self.iw == 2:
                        x, w = self.instance_norm_layer(x)
                        w_arr.append(w)
                    else:
                        x = self.instance_norm_layer(x)
            else:
                x = module(x)

        return [x, w_arr]


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        iw: int = 0,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.expand_ratio = expand_ratio
        self.iw = iw

        if iw == 1:
            self.instance_norm_layer = InstanceWhitening(oup)
        elif iw == 2:
            self.instance_norm_layer = InstanceWhitening(oup)
        elif iw == 3:
            self.instance_norm_layer = nn.InstanceNorm2d(oup, affine=False)
        elif iw == 4:
            self.instance_norm_layer = nn.InstanceNorm2d(oup, affine=True)
        else:
            self.instance_norm_layer = nn.Sequential()

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)


    def forward(self, x_tuple):
        if len(x_tuple) == 2:
            x = x_tuple[0]
        else:
            print("error in invert residual forward path")
            return
        if self.expand_ratio != 1:
            x_tuple = self.conv[0](x_tuple)
            x_tuple = self.conv[1](x_tuple)
            conv_x = x_tuple[0]
            w_arr = x_tuple[1]
            conv_x = self.conv[2](conv_x)
            conv_x = self.conv[3](conv_x)
        else:
            x_tuple = self.conv[0](x_tuple)
            conv_x = x_tuple[0]
            w_arr = x_tuple[1]
            conv_x = self.conv[1](conv_x)
            conv_x = self.conv[2](conv_x)

        if self.use_res_connect:
            x = x + conv_x
        else:
            x = conv_x

        if self.iw >= 1:
            if self.iw == 1 or self.iw == 2:
                x, w = self.instance_norm_layer(x)
                w_arr.append(w)
            else:
                x = self.instance_norm_layer(x)

        return [x, w_arr]


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        iw: list = [0, 0, 0, 0, 0, 0, 0],
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],  # feature 1
                [6, 24, 2, 2],  # feature 2, 3
                [6, 32, 3, 2],  # feature 4, 5, 6
                [6, 64, 4, 2],  # feature 7, 8, 9, 10
                [6, 96, 3, 1],  # feature 11, 12, 13
                [6, 160, 3, 2], # feature 14, 15, 16
                [6, 320, 1, 1], # feature 17
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        # feature 0
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        feature_count = 0
        iw_layer = [1, 6, 10, 17, 18]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                feature_count += 1
                stride = s if i == 0 else 1
                if feature_count in iw_layer:
                    layer = iw_layer.index(feature_count)
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, iw=iw[layer + 2]))
                else:
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, iw=0))
                input_channel = output_channel
        # building last several layers
        # feature 18
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        #model.load_state_dict(state_dict)
        forgiving_state_restore(model, state_dict)
    return model
