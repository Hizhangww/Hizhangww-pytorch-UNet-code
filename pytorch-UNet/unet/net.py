import torch
from torch import nn
from torch.nn import functional as F


"""整个网络主要分为 卷积层 下采样 上采样 根据Unet图去看下面代码更清晰点（见Unet结构图.png）"""

class Conv_Block(nn.Module):    #定义一个卷积层类
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        """
        该卷积类包括两个卷积层
        """
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),    #使均值为0 方差为1的正态分布上 使数据分布一致，避免梯度消失
            nn.Dropout2d(0.3),  #防止过拟合
            nn.LeakyReLU(),     #激活函数
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),   #重复上一卷积层
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class DownSample(nn.Module):    #定义下采样类
    def __init__(self,channel):
        super(DownSample, self).__init__()
        """
        原本下采样是一个池化层，由于池化层没有特征提取能力 特征丢地太多 原代码作者将池化层换成了卷积
        """
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):  #定义上采样层类
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')   #上采样采用最近邻插值法（保证图片信息不丢失的情况下，图像放大为原来两倍）
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)   #看Unet图 这里有一个拼接


class UNet(nn.Module):  #定义Unet网络结构
    def __init__(self):
        super(UNet, self).__init__()    #定义每个层
        self.c1=Conv_Block(3,64)
        self.d1=DownSample(64)
        self.c2=Conv_Block(64,128)
        self.d2=DownSample(128)
        self.c3=Conv_Block(128,256)
        self.d3=DownSample(256)
        self.c4=Conv_Block(256,512)
        self.d4=DownSample(512)
        self.c5=Conv_Block(512,1024)
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out=nn.Conv2d(64,3,3,1,1)
        self.Th=nn.Sigmoid()    #最后采用Sigmoid函数作为激活函数

    def forward(self,x):    #传播过程，根据Unet图去看
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5,R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))

if __name__ == '__main__':  #测试代码 如果形状相同就正确
    x=torch.randn(2,3,256,256)
    net=UNet()
    print(net(x).shape)




