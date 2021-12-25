from PIL import Image


def keep_image_size_open(path, size=(256, 256)):
    """等比缩放 先找出以图片最长边为边长的正方形,作为幕布 然后将原图粘到幕布上 最后再进行缩放 这样会使原图的长和宽等比缩放"""
    img = Image.open(path)  #将图片读取进来
    temp = max(img.size)    #取到图片的最长边
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))    #创造幕布 边长为temp的正方形 颜色为黑（0，0，0）代表全黑
    mask.paste(img, (0, 0))     #将原图粘到幕布上
    mask = mask.resize(size)    #将幕布resize一下 变成（256*256）
    return mask
