import os

from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path='params/unet.pth'   #保存训练权值的路径
data_path=r'C:\Users\LJY\Desktop\VOCdevkit\VOC2007'
save_path='train_image'     #保存训练好的图片的路径

Summarywriter = SummaryWriter('logs')
if __name__ == '__main__':
    data_loader=DataLoader(MyDataset(data_path),batch_size=2,shuffle=True)
    net=UNet().to(device)
    if os.path.exists(weight_path):         #如果有已经有训练好的权值文件，就用训练好的权值
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt=optim.Adam(net.parameters())    #采用Adam优化器 学习率为0.001
    loss_fun=nn.BCELoss()

    index = 0
    epoch=1
    while True:
        for i,(image,segment_image) in enumerate(data_loader):
            image, segment_image=image.to(device),segment_image.to(device)

            out_image=net(image)
            loss=loss_fun(out_image,segment_image)
            running_loss = loss.item()

            opt.zero_grad()     #清空梯度
            loss.backward()   #反向传播得到梯度值
            opt.step()      #更新梯度

            if i%5==0:
                print(f'{epoch}-{i}-train_loss===>>{running_loss}')

            if i%50==0:
                torch.save(net.state_dict(),weight_path)





            _image=image[0]
            _segment_image=segment_image[0]
            _out_image=out_image[0]

            img=torch.stack([_image,_segment_image,_out_image],dim=0)
            save_image(img,f'{save_path}/{i}.png')


        epoch+=1

