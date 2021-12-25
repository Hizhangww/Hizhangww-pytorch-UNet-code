import os

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
transform=transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path=path
        self.name=os.listdir(os.path.join(path,'SegmentationClass'))
        """获取到SegmentationClass文件夹下的所有图片名称  SegmentationClass文件夹下是标签"""

    def __len__(self):
        return len(self.name)   #返回所有图片的数量

    def __getitem__(self, index):
        segment_name=self.name[index]  #图片标签格式为xxx.png
        segment_path=os.path.join(self.path,'SegmentationClass',segment_name)   #得到每个图片标签的路径
        image_path=os.path.join(self.path,'JPEGImages',segment_name.replace('png','jpg'))
        """得到需要训练的图片标签 （需要注意有标签的图片比实际图片少，只需要带有标签的图片）    并将实际图片（PNG格式）转换成JPG"""
        segment_image=keep_image_size_open(segment_path)
        """标签的图片size是不一样的，需要进行等比缩放，keep_image_size_open函数看utils.py"""
        image=keep_image_size_open(image_path)  #原图也一样进行等比缩放
        return transform(image),transform(segment_image)    #原始图和标签图都进行归一化

if __name__ == '__main__':  #测试代码 如果size相同则正确
    data=MyDataset('C:\\Users\\honor\\Desktop\\VOCdevkit\\VOC2007')
    print(data[0][0].shape)
    print(data[0][1].shape)