from fontTools.unicodedata import block
from torch import nn as nn
import torch
import torch.nn.functional as F

#18和34 没有用1*1卷积,改变通道数
class BasicBlock(nn.Module):
    #膨胀因子，表示通道数不变
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.downsample = downsample
        self.sequential = nn.Sequential(
            #第一个stride是传进的参数，可能会降采样
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            #stride=1，padding=1，kernel_size=3*3，得出的特征图大小不变
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            #注意downsample是论文里面的，是否有降采样
        )
    def forward(self, x):
        #恒等映射
        identity = x
        #残差部分 如果发生过降采样，那么恒等映射部分也需要降采样，通道数减半
        if self.downsample is not None:
            identity = self.downsample(x)
        x=self.sequential(x)
        x+=identity
        x=F.relu(x)
        return x

class Bottleneck(nn.Module):
    #设置膨胀因子，最后通道个数与开始通道个数比值
    expansion = 4
    def __init__(self, in_channel, out_channel,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.sequential = nn.Sequential(
            #1*1卷积，改变通道数
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel*self.expansion),
        )
        self.downsample = downsample
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x=self.sequential(x)
        x+=identity
        x=F.relu(x)
        return x

class MyResNet(nn.Module):
    def __init__(self,layer,num_classes):
        # include_top 分类头，为线性层
        super().__init__()
        self.layers_dict={
            18:[2,2,2,2],
            34:[3,4,6,3],
            50:[3,4,6,3],
            101:[3,4,23,3],
            152:[3,8,36,3],
        }
        self.layers = layer
        self.in_channel = 64
        self.sequential = nn.Sequential(
            #从224到112 所以padding一定为3 【w-k+2p】/s+1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #又减半
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #残差层
        )
        if layer >=50:
            block = Bottleneck
        else:
            block = BasicBlock

        self.block1=self.make_layer(block,self.layers_dict[layer][0],channel=64,stride=1)
        self.block2=self.make_layer(block,self.layers_dict[layer][1],channel=128,stride=2)
        self.block3=self.make_layer(block,self.layers_dict[layer][2],channel=256,stride=2)
        self.block4=self.make_layer(block,self.layers_dict[layer][3],channel=512,stride=2)

        #最后平均池化后，全连接输出
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512*block.expansion,num_classes)

        #初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def forward(self, x):
        x = self.sequential(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    #残差层的创建,注意channel为第一层卷积的卷积核个数,block为resenet层数用于选择那个残差模块，
    def make_layer(self,block,blocknum,channel,stride=1):
        downsample = None
        #下采样的条件，stride不为1和在50 101 152的情况下
        if stride != 1 or self.in_channel != channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel*block.expansion),
            )
        #存下每个层
        layers=[]
        layers.append(block(self.in_channel,channel,stride,downsample))
        #更改in_channel,准备下一次残差
        self.in_channel = channel*block.expansion
        for layer in range(1,blocknum):
            layers.append(block(self.in_channel,channel))

        return nn.Sequential(*layers)


