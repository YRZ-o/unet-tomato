import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class TomatoDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(TomatoDataset, self).__init__()  # 调用父类的构造函数
        self.flag = "training" if train else "test" # 根据train参数确定数据集是用于训练还是测试
        data_root = os.path.join(root, "tomatoNewDataset", self.flag) # 构建数据集的根目录路径
        assert os.path.exists(data_root), f"path '{data_root}' does not exists." # 断言检查数据集根目录是否存在，如果不存在则报错
        self.transforms = transforms # 存储数据增强变换函数

        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]  # 构建所有图像文件的完整路径列表
        self.manual = [os.path.join(data_root, "masks", i.split(".")[0] + ".png")
                       for i in img_names] # 构建与图像对应的手动标注文件的完整路径列表
        # check files
        # 检查手动标注文件是否都存在，如果有不存在的则抛出异常
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')  # 打开索引为idx的图像文件，并将其转换为RGB格式
        manual = Image.open(self.manual[idx]).convert('L') # 打开索引为idx的手动标注文件，并将其转换为灰度图
        manual = np.array(manual) / 255  # 将手动标注的图像转换为numpy数组，并归一化到0-1范围
        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(manual)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list) # 主要功能是返回数据集的大小，即数据集中样本的数量。

    @staticmethod
    def collate_fn(batch):
        """
            静态方法，用于将数据集中的多个样本整理成一个批次。
            :param batch: 一个包含多个样本的列表，每个样本是一个元组 (image, target)
            :return: 整理后的一个批次的图像和目标，分别为 batched_imgs 和 batched_targets
            """

        images, targets = list(zip(*batch)) # 将batch中的样本解压缩为两个独立的列表，images包含所有图像，targets包含所有目标
        batched_imgs = cat_list(images, fill_value=0)
        # 使用cat_list函数将多个图像整理成一个批次的图像
        # fill_value=0 表示如果图像大小不一致，用0填充
        batched_targets = cat_list(targets, fill_value=255)
        # 使用cat_list函数将多个目标整理成一个批次的目标
        # fill_value=255 表示如果目标大小不一致，用255填充
        return batched_imgs, batched_targets  # 返回整理后的一个批次的图像和目标


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

