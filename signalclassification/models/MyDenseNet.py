
from torch import nn as nn
import torch


class DenseLayer(nn.Module):
    def __init__(self, in_channels, bn_size,growth_rate=32,drop_rate=0.0):
        #bn_size 控制bottleneck的超参数，主要是1*1卷积的输出
        super(DenseLayer,self).__init__()
        self.in_channels = in_channels
        self.drop_rate = drop_rate
        # BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv
        self.denselayer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,out_channels=bn_size*growth_rate,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(bn_size*growth_rate),
            nn.ReLU(inplace=True),
            #注意输出要保持一致 都是growth_rate
            nn.Conv2d(in_channels=bn_size*growth_rate,out_channels=growth_rate,kernel_size=3,stride=1,padding=1,bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)
    def forward(self,x):
        new_x = self.denselayer(x)
        if self.drop_rate > 0:
            new_x = self.dropout(new_x)
        return torch.cat([x,new_x],1)




class DenseBlock(nn.Module):
    #添加bottleneck
    def __init__(self, in_channels,num_layers,growth_rate,bn_size,drop_rate=0.0):
        super(DenseBlock,self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels+i*growth_rate,bn_size,growth_rate,drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)

class Transition(nn.Module):
    #降低分辨率
    #BN + Relu + 1*1Conv + 2*2AvgPool
    def __init__(self,in_channels,compression):
        super(Transition,self).__init__()
        self.sequential = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,out_channels=int(in_channels*compression),kernel_size=1,stride=1,padding=0,bias=False),
            nn.AvgPool2d(kernel_size=2,stride=2),
        )

    def forward(self,x):
        return self.sequential(x)

class DenseNet(nn.Module):

    def __init__(self,layer,growth_rate,num_classes=58):
        super().__init__()
        self.layers_dict={
            121:[6,12,24,16],
            169: [6,16,32,32],
            201: [6,12,48,32],
            264: [6,12,64,48]
        }
        bn_size=4
        drop_rate=0.2
        compression=0.5
        #num_features为处理后进入DEnseBlock的通道数
        num_features=64
        self.sequential=nn.Sequential(
            nn.Conv2d(3,num_features,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.block1=DenseBlock(num_features,self.layers_dict[layer][0],growth_rate,bn_size,drop_rate)
        num_features+=self.layers_dict[layer][0]*growth_rate
        self.transition1=Transition(num_features,compression)
        num_features=int(num_features*compression)

        self.block2 = DenseBlock(num_features, self.layers_dict[layer][1], growth_rate,bn_size, drop_rate)
        num_features += self.layers_dict[layer][1] * growth_rate
        self.transition2 = Transition(num_features, compression)
        num_features=int(num_features*compression)

        self.block3 = DenseBlock(num_features, self.layers_dict[layer][2], growth_rate,bn_size, drop_rate)
        num_features += self.layers_dict[layer][2] * growth_rate
        self.transition3 = Transition(num_features, compression)
        num_features = int(num_features * compression)

        self.block4 = DenseBlock(num_features, self.layers_dict[layer][3], growth_rate,bn_size, drop_rate)
        num_features += self.layers_dict[layer][3] * growth_rate

        # self.avgpool = nn.AvgPool2d(7,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc=nn.Linear(num_features,num_classes)

    def forward(self,x):
        x = self.sequential(x)
        x = self.block1(x)
        x = self.transition1(x)
        x = self.block2(x)
        x = self.transition2(x)
        x = self.block3(x)
        x = self.transition3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


