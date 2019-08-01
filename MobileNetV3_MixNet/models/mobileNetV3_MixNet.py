import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def _split_channels(num_channels, num_groups, mode='equal'):
    if mode is 'exponential':
        split = [int(num_channels // math.pow(2, exp+1)) for exp in range(num_groups)]
        split[-1] += num_channels - sum(split)
    elif mode is 'equal':
        split = [int(num_channels // num_groups) for _ in range(num_groups)]
        split[0] += num_channels - sum(split)
        pass
    else:
        return [num_channels]

    return split


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileBottleneck_Mixed(nn.Module):
    def __init__(self, inp, oup, kernels, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck_Mixed, self).__init__()
        # assert stride in [1, 2]
        # assert kernel in [3, 5]
        assert isinstance(kernels, list)

        self.groups = len(kernels)
        self.paddings = [(kernel - 1) // 2 for kernel in kernels]
        self.channels = _split_channels(exp, self.groups)
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv1 = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
        )

        self.conv_mixed = nn.Sequential()
        for idx, (channel, kernel, padding) in enumerate(zip(self.channels, kernels, self.paddings)):
            self.conv_mixed.add_module(str(idx), conv_layer(channel, channel, kernel, stride, padding, groups=channel, bias=False))

        self.conv2 = nn.Sequential(
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        y = self.conv1(x)

        y = torch.split(y, self.channels, 1)  # split y to several groups
        y = [c(y_) for y_, c in zip(y, self.conv_mixed)]
        y = torch.cat(y, 1)

        y = self.conv2(y)

        if self.use_res_connect:
            return x + y
        else:
            return y


class MobileNetV3_MixNet(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='large', width_mult=1.0):
        super(MobileNetV3_MixNet, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            # shuffleNetV2提到组卷积太多会影响速度, 因为增加了内存访问开销(MAC)
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [[3, 5],  16,  16,  False, 'RE', 1],
                [[3, 5],  64,  24,  False, 'RE', 2],
                [[3, 5],  72,  24,  False, 'RE', 1],
                [[3, 5, 7], 72,  40,  True,  'RE', 2],
                [[3, 5, 7], 120, 40,  True,  'RE', 1],
                [[3, 5, 7], 120, 40,  True,  'RE', 1],
                [[3, 5],  240, 80,  False, 'HS', 2],
                [[3, 5],  200, 80,  False, 'HS', 1],
                [[3, 5],  184, 80,  False, 'HS', 1],
                [[3, 5],  184, 80,  False, 'HS', 1],
                [[5, 7],  480, 112, True,  'HS', 1],
                [[5, 7, 9],  672, 112, True,  'HS', 1],
                [[5, 7, 9], 672, 160, True,  'HS', 2],
                [[5, 7, 9], 960, 160, True,  'HS', 1],
                [[3], 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [[3, 5], 16,  16,  True,  'RE', 2],
                [[3, 5], 72,  24,  False, 'RE', 2],
                [[3, 5], 88,  24,  False, 'RE', 1],
                [[5, 7], 96,  40,  True,  'HS', 2],
                [[5, 7], 240, 40,  True,  'HS', 1],
                [[5, 7], 240, 40,  True,  'HS', 1],
                [[5, 7], 120, 48,  True,  'HS', 1],
                [[5, 7], 144, 48,  True,  'HS', 1],
                [[5, 7], 288, 96,  True,  'HS', 2],
                [[5, 7], 576, 96,  True,  'HS', 1],
                [[5, 7], 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck_Mixed(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
