import torch
from utils import keep_image_size_open
from net import *
import os
from torchvision import transforms
from data import *
from torchvision.utils import save_image

transform=transforms.Compose([
    transforms.ToTensor()
])

net = UNet()

weight = 'params/unet.pth'      #加载权重
if os.path.exists(weight):
    net.load_state_dict(torch.load(weight))
    print('successfully')
else:
    print('no loading')

_input = input('please input image path:')

img = keep_image_size_open(_input)      #图像预处理
img_data = transform(img)
img_data = torch.unsqueeze(img_data,dim=0)
out = net(img_data)
save_image(out,'text_result/result.jpg')
print(out)