import math

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as cp

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, with_cp=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.with_cp = with_cp

    def forward(self, x):
        def _inner_forward(x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, output_channels=512):
        super(ResNet, self).__init__()
        channels = [output_channels//(2**i) for i in reversed(range(5))]
        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=1)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=1)
        self.layer5 = self._make_layer(block, channels[4], layers[4], stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, extra_feats=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if extra_feats is not None:
            if extra_feats[0].shape[1]>0:
                x = x+F.interpolate(extra_feats[0], x.shape[2:], mode='nearest')
        x = self.layer1(x)
        if extra_feats is not None:
            if extra_feats[1].shape[1]>0:
                x = x+F.interpolate(extra_feats[1], x.shape[2:], mode='nearest')
        x = self.layer2(x)
        if extra_feats is not None:
            if extra_feats[2].shape[1]>0:
                x = x+F.interpolate(extra_feats[2], x.shape[2:], mode='nearest')
        x = self.layer3(x)
        if extra_feats is not None:
            if extra_feats[3].shape[1]>0:
                x = x+F.interpolate(extra_feats[3], x.shape[2:], mode='nearest')
        x = self.layer4(x)
        if extra_feats is not None:
            if extra_feats[4].shape[1]>0:
                x = x+F.interpolate(extra_feats[4], x.shape[2:], mode='nearest')
        x = self.layer5(x)
        if extra_feats is not None:
            if extra_feats[5].shape[1]>0:
                x = x+F.interpolate(extra_feats[5], x.shape[2:], mode='nearest')
        return x

def resnet45(alpha_d, output_channels=512):
    layers = [int(round(x*alpha_d)) for x in [3, 4, 6, 6, 3]]
    return ResNet(BasicBlock, layers, output_channels=output_channels)
