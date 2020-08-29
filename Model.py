import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)

        return out

class CnnBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1):
        super(CnnBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.bn = nn.BatchNorm2d(planes)
        self.mish = Mish()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.mish(out)
        out = F.dropout(out, 0.1)
        return out


class FaceNet(nn.Module):
    IMAGE_SHAPE = (96, 128)

    def __init__(self):
        super(FaceNet, self).__init__()
        self.cnn1 = CnnBlock(3, 32)
        self.cnn2 = BasicBlock(32, 64)
        self.cnn3 = BasicBlock(64, 128)
        self.cnn4 = BasicBlock(128, 256)
        self.cnn5 = BasicBlock(256, 512)
        self.cnn6 = nn.Conv2d(512, 1024, kernel_size=1, padding=0)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x1 = self.cnn1(x)  # 96*128*32
        x2 = self.max_pool(x1)  # 48*64*32
        x3 = self.cnn2(x2)  # 48*64*64
        x4 = self.max_pool(x3)  # 24*32*64
        x5 = self.cnn3(x4)  # 24*32*128
        x6 = self.max_pool(x5)  # 12*16*128
        x7 = self.cnn4(x6)  # 10*14*256
        x8 = self.max_pool(x7)  # 5*7*256
        x9 = self.cnn5(x8)  # 3*5*512
        x10 = self.avg_pool(x9)  # 1*1*512
        x11 = self.cnn6(x10)  # 1*1*1024
        return x11.view(-1, 1024)


class ResnetFaceModel(nn.Module):

    IMAGE_SHAPE = (96, 128)

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.extract_feature = nn.Linear(
            self.feature_dim*4*3, self.feature_dim)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = x.view(x.size(0), -1)
        feature = self.extract_feature(x)
        feature_normed = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

        return feature_normed

class Resnet18FaceModel(ResnetFaceModel):

    FEATURE_DIM = 512

    def __init__(self):
        super().__init__(self.FEATURE_DIM)
        self.base = resnet18(pretrained=False)

def test_model():
    net = Resnet18FaceModel()
    x = torch.randn(2, 3, 96, 128)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test_model()
