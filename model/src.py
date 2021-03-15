"""
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import torch.nn as nn
import torch.nn.functional as F
from config import *
from utils import weights_init


__all__ = ['ResNet', 'BasicBlock']


class LambdaLayer(nn.Module):
    """
    Creates LambdaLayer for shortcut for ResNet from [1]
    """

    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class BasicBlock(nn.Module):
    """
    Creates BasicBlock for ResNet from [1]
    """

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=KERNEL_SIZE, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=KERNEL_SIZE, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        # If there are some problems with dims while changing between levels
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2],
                                              [0, 0, 0, 0, planes//4, planes//4],
                                              "constant",
                                              0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    General ResNet class based on definition from [1]
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=KERNEL_SIZE, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, BLOCKS_PLANES[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, BLOCKS_PLANES[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, BLOCKS_PLANES[2], num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        :param block: Type of basic block of ResNet (I have implemented only BasicBlock
                      from [1])
        :param planes: Number of planes (convolution filters)
        :param num_blocks: Number of blocks in level
        :param stride: List of strides for each block
        :return: nn.Sequential of layers in current level
        """

        strides = [stride] + [1]*(num_blocks-1)

        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
