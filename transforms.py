import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    """
        如果图像的最小边长小于给定的size，对图像进行填充。

        :param img: 输入的图像，通常是PIL Image对象
        :param size: 用于比较的目标尺寸，图像最小边长需要达到这个值
        :param fill: 填充的值，默认为0
        :return: 经过填充后的图像，如果图像最小边长不小于size，则返回原图像
        """
    min_size = min(img.size) # 获取图像的最小边长
    if min_size < size:
        ow, oh = img.size# 获取图像的原始宽度和高度
        padh = size - oh if oh < size else 0  # 计算高度方向需要填充的像素数
        padw = size - ow if ow < size else 0  # 计算宽度方向需要填充的像素数
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
        # 使用torch.nn.functional.pad对图像进行填充
        # (0, 0, padw, padh)表示在图像的左、上、右、下四个方向填充的像素数
    return img


class Compose(object): # 将多个数据变换操作组合成一个单一的可调用对象
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        """
            使Compose类的实例可调用，对输入数据执行组合的变换操作。
            :param image: 输入的数据，例如图像。
            :param target: 与输入数据对应的目标数据，例如图像的标签或掩码。
            :return: 经过所有变换操作处理后的图像和目标数据。
        """
        for t in self.transforms:
            image, target = t(image, target) # 依次对图像和目标数据应用每个变换操作
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    """
        用于对图像进行归一化处理的类。
        该类实现了对输入图像按照给定的均值和标准差进行归一化操作，
        通常在深度学习图像处理中用于标准化数据，以帮助模型更快收敛。
        """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        """
            使Normalize类的实例可调用，对输入图像进行归一化处理。
            :param image: 输入的图像张量，通常是经过ToTensor等操作转换后的张量。
            :param target: 与图像对应的目标数据，例如图像的标签或掩码。在本操作中目标数据不做处理。
            :return: 归一化后的图像和原始的目标数据。
        """
        # 使用F.normalize函数对图像进行归一化处理
        # 按照预先设定的均值和标准差对图像的每个通道进行归一化
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
