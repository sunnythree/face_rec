import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


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
    def __init__(self):
        super(FaceNet, self).__init__()
        self.cnn1 = CnnBlock(3, 32)
        self.cnn2 = CnnBlock(32, 64)
        self.cnn3 = CnnBlock(64, 128)
        self.cnn4 = CnnBlock(128, 256, padding=0)
        self.cnn5 = CnnBlock(256, 512, padding=0)
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


def test_model():
    net = FaceNet()
    x = torch.randn(2, 3, 96, 128)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test_model()
