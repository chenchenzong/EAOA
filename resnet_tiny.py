import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet_fea(nn.Module):
    def __init__(self, block, num_blocks):
        super(resnet_fea, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride, flag=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, img_size, intermediate=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        out = out.view(out.size(0), -1)
        return out

class resnet_clf(nn.Module):
    def __init__(self, block, n_class=10):
        super(resnet_clf, self).__init__()
        self.linear = nn.Linear(4 * 128 * block.expansion, n_class)

    def forward(self, emb):
        # emb = x.view(x.size(0), -1)
        logit = self.linear(emb)
        return logit


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_class=10):
        super(ResNet, self).__init__()

        self.embDim = 4 * 128 * block.expansion

        self.feature_extractor = resnet_fea(block, num_blocks)
        self.linear = resnet_clf(block, n_class)


    def forward(self, x):
        emb = self.feature_extractor(x, x.shape[2])
        logit = self.linear(emb)
        return logit, emb

    def get_embedding_dim(self):
        return self.embDim


def ResNet18_Tiny(n_class):
    return ResNet(BasicBlock, [2,2,2,2], n_class=n_class)

def ResNet34_Tiny(n_class):
    return ResNet(BasicBlock, [3,4,6,3], n_class=n_class)

def ResNet50_Tiny(n_class):
    return ResNet(Bottleneck, [3,4,6,3], n_class=n_class)

def ResNet101_Tiny(n_class):
    return ResNet(Bottleneck, [3,4,23,3], n_class=n_class)

def ResNet152_Tiny(n_class):
    return ResNet(Bottleneck, [3,8,36,3], n_class=n_class)


# test()
