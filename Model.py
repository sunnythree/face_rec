import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
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



class CnnAlign(nn.Module):
    def __init__(self):
        super(CnnAlign, self).__init__()
        self.cnn1 = CnnBlock(3, 16)
        self.cnn2 = CnnBlock(16, 32)
        self.cnn3 = CnnBlock(32, 64)
        self.cnn4 = CnnBlock(64, 128)
        self.cnn5 = CnnBlock(128, 64)
        self.max_pool = nn.MaxPool2d(2)
        self.drop_3 = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 2*194)

    def forward(self, x):
        x1 = self.cnn1(x)         #224
        x2 = self.max_pool(x1)    #112
        x3 = self.cnn2(x2)        #112
        x4 = self.max_pool(x3)    #56
        x5 = self.cnn3(x4)        #56
        x6 = self.max_pool(x5)    #28
        x7 = self.cnn4(x6)        #28
        x8 = self.max_pool(x7)    #14
        x9 = self.cnn5(x8)        #14
        x10 = self.max_pool(x9)   #7
        x11 = self.fc1(x10.view(-1, 7*7*64))
        x12 = self.drop_3(x11)
        x13 = self.fc2(x12)
        return x13


class CnnAlignHard(nn.Module):
    def __init__(self):
        super(CnnAlignHard, self).__init__()
        self.cnn1 = CnnBlock(3, 16)
        self.cnn2 = CnnBlock(16, 32)
        self.cnn3 = CnnBlock(32, 64)
        self.cnn4 = CnnBlock(64, 128)
        self.cnn5 = CnnBlock(128, 256)
        self.cnn6 = CnnBlock(256, 512, padding=0)
        self.cnn7 = nn.Conv2d(512, 388, kernel_size=1, padding=0)
        self.max_pool = nn.MaxPool2d(2)
        self.mean_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        x1 = self.cnn1(x)          # 224
        x2 = self.max_pool(x1)     # 112
        x3 = self.cnn2(x2)         # 112
        x4 = self.max_pool(x3)     # 56
        x5 = self.cnn3(x4)         # 56
        x6 = self.max_pool(x5)     # 28
        x7 = self.cnn4(x6)         # 28
        x8 = self.max_pool(x7)     # 14
        x9 = self.cnn5(x8)         # 14
        x10 = self.max_pool(x9)    # 7
        x11 = self.cnn6(x10)       # 5
        x12 = self.mean_pool(x11)  # 1
        x13 = self.cnn7(x12)       # 5
        return torch.flatten(x13, 1)


def test_model():
    net = CnnAlignHard()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())
if __name__=='__main__':
    test_model()