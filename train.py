import os
import time
import datetime

import torch

from src import AttentionResUNet, UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import TomatoDataset
import transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)  # 计算随机缩放的尺寸范围
        max_size = int(1.2 * base_size)
        # 初始化一个空的变换列表
        trans = [T.RandomResize(min_size, max_size)]  # 随机缩放图像尺寸
        if hflip_prob > 0:  # 如果水平翻转概率大于0，则添加水平翻转变换
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:  # 如果垂直翻转概率大于0，则添加垂直翻转变换
            trans.append(T.RandomVerticalFlip(vflip_prob))
        # 添加随机裁剪、转换为张量和归一化操作
        trans.extend([
            T.RandomCrop(crop_size),  # 随机裁剪图像
            T.ToTensor(),  # 将图像转换为张量
            T.Normalize(mean=mean, std=std),  # 对图像进行归一化
        ])
        self.transforms = T.Compose(trans)  # 将所有变换组合成一个整体的变换操作

    """
           初始化函数，用于设置图像分割训练时的数据增强和预处理操作。

           Args:
               base_size (int): 图像的基础大小，用于计算随机缩放的尺寸范围。
               crop_size (int): 随机裁剪后的图像大小。
               hflip_prob (float): 随机水平翻转的概率，默认为0.5。
               vflip_prob (float): 随机垂直翻转的概率，默认为0.5。
               mean (tuple): 图像归一化时的均值，默认为(0.485, 0.456, 0.406)。
               std (tuple): 图像归一化时的标准差，默认为(0.229, 0.224, 0.225)。
           """

    def __call__(self, img, target):
        return self.transforms(img, target)

    """
    调用函数，对输入的图像和目标进行变换。
    Args:
        img (PIL.Image): 输入的图像。
        target (PIL.Image): 输入的目标（通常是分割标签）。

        Returns:
            tuple: 返回变换后的图像和目标。
    SegmentationPresetTrain 包含训练时的数据增强操作，以提高模型的泛化能力。
    """


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 初始化一个包含评估阶段所需变换操作的列表
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


"""
    初始化函数，用于设置图像分割评估时的数据预处理操作。
    Args:
        mean (tuple): 图像归一化时的均值，默认为(0.485, 0.456, 0.406)。
        std (tuple): 图像归一化时的标准差，默认为(0.229, 0.224, 0.225)。
    包含必要的预处理操作（如转换为张量和归一化），而不包含数据增强操作（如随机缩放、翻转、裁剪等），因为评估阶段不需要数据增强。
"""


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 640
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    # model = AttentionResUNet(in_channels=3, num_classes=num_classes, base_c=32)
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes+1

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 它用于加载和访问数据集中的样本（如图像和标签）
    train_dataset = TomatoDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = TomatoDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 获取当前cpu核心数，batch_size 大于 1，则取 batch_size 的值；否则取 0,最大值为8
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    model.to(device)
    # 从模型的参数中筛选出需要优化的参数，并将它们放入一个列表中，方便传递给优化器进行训练
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    # 优化器
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    """
    momentum 动量参数，用于加速 SGD 的收敛。它通过引入前几次更新的方向来平滑梯度更新，避免陷入局部最优解
    weight_decay 是权重衰减（L2 正则化）参数，用于防止模型过拟合。它通过在损失函数中添加参数的 L2 范数惩罚项，限制参数的大小。
    args.weight_decay 是从外部传入的权重衰减值，通常是一个较小的浮点数（如 0.0001）
    """
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # 混合精度训练通过使用半精度（float16）和单精度（float32）的结合，来加速训练并减少显存占用。

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    """
    从一个检查点文件（checkpoint file）中恢复模型的训练状态，
    包括模型参数、优化器状态、学习率调度器状态以及混合精度训练的梯度缩放器状态（如果启用）

    """
    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 调用 train_one_epoch 函数进行一轮训练，返回平均损失和当前学习率
        # model 是要训练的模型，optimizer 是优化器，train_loader 是训练数据加载器
        # lr_scheduler 是学习率调度器，print_freq 控制打印频率，scaler 用于混合精度训练
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        # 调用 evaluate 函数在验证集上评估模型，返回混淆矩阵和 Dice 系数
        # val_loader 是验证数据加载器
        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)  # 将混淆矩阵转换为字符串
        print(val_info)  # 打印混淆矩阵信息
        print(f"dice coefficient: {dice:.3f}")  # 打印 Dice 系数，保留三位小数
        # write into txt
        # 打开文件 results_file，以追加模式写入训练和验证信息
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            # 构建训练信息字符串，包含当前 epoch、训练损失、学习率和 Dice 系数
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")  # 将训练信息、验证信息写入文件，并添加两个换行符分隔

        if args.save_best is True:  # 如果 args.save_best 为 True，表示只保存最佳模型
            if best_dice < dice:
                best_dice = dice
            else:
                continue  # 如果当前 Dice 系数不是最佳，跳过本次保存

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        # 构建保存文件的字典，包含模型参数、优化器参数、学习率调度器参数、当前 epoch 和参数对象 args
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))  # 将总时间转换为 timedelta 对象，并格式化为字符串
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse  # 导入argparse模块，用于处理命令行参数

    parser = argparse.ArgumentParser(description="pytorch unet training")
    # 创建一个ArgumentParser对象，description参数用于描述程序功能

    parser.add_argument("--data-path", default="./", help=" root")
    # 添加一个名为"--data-path"的命令行参数
    # default参数指定了该参数的默认值为"./"
    # help参数对该参数的作用进行了说明，即指定DRIVE数据集的根目录

    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    # 添加一个名为"--num-classes"的命令行参数
    # default参数指定默认值为1
    # type参数指定该参数的类型为整数
    # 此参数用于指定（不包括背景的）类别数量
    parser.add_argument("--device", default="cuda:1", help="training device")
    # 添加一个名为"--device"的命令行参数
    # default参数指定默认值为"cuda"
    # help参数说明该参数用于指定训练使用的设备
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    # 添加一个名为"-b" 或 "--batch-size"的命令行参数
    # default参数指定默认值为4
    # type参数指定该参数的类型为整数
    # 该参数用于指定训练时的批量大小
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")
    # 添加一个名为"--epochs"的命令行参数
    # default参数指定默认值为200
    # type参数指定该参数的类型为整数
    # metavar参数用于在帮助信息中显示参数的名称
    # help参数说明该参数用于指定总的训练轮数

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    # 添加一个名为"--lr"的命令行参数
    # default参数指定默认值为0.01
    # type参数指定该参数的类型为浮点数
    # help参数说明该参数用于指定初始学习率
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # 添加一个名为"--momentum"的命令行参数
    # default参数指定默认值为0.9
    # type参数指定该参数的类型为浮点数
    # metavar参数用于在帮助信息中显示参数的名称
    # help参数说明该参数用于指定动量
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 添加一个名为"--wd" 或 "--weight-decay"的命令行参数
    # default参数指定默认值为1e-4
    # type参数指定该参数的类型为浮点数
    # metavar参数用于在帮助信息中显示参数的名称
    # help参数说明该参数用于指定权重衰减
    # dest参数指定该参数在解析后的属性名
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    # 添加一个名为"--print-freq"的命令行参数
    # default参数指定默认值为1
    # type参数指定该参数的类型为整数
    # help参数说明该参数用于指定打印训练信息的频率
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 添加一个名为"--resume"的命令行参数
    # default参数指定默认值为空字符串
    # help参数说明该参数用于指定从哪个检查点恢复训练
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # 添加一个名为"--start-epoch"的命令行参数
    # default参数指定默认值为0
    # type参数指定该参数的类型为整数
    # metavar参数用于在帮助信息中显示参数的名称
    # help参数说明该参数用于指定开始训练的轮数
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    # 添加一个名为"--save-best"的命令行参数
    # default参数指定默认值为True
    # type参数指定该参数的类型为布尔值
    # help参数说明该参数用于指定是否只保存具有最佳Dice系数的模型权重
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    # 添加一个名为"--amp"的命令行参数
    # default参数指定默认值为False
    # type参数指定该参数的类型为布尔值
    # help参数说明该参数用于指定是否使用torch.cuda.amp进行混合精度训练

    args = parser.parse_args()  # 解析命令行参数

    return args


# 这段代码定义了一个名为 parse_args 的函数，用于解析命令行参数。通过 argparse 模块，
# 用户可以在运行程序时通过命令行指定各种训练参数，
# 如数据集路径、类别数、训练设备、批量大小、训练轮数、学习率等。函数最后返回解析后的参数对象，
# 以便在程序的其他部分使用这些参数来配置训练过程。

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    # 如果不存在，则创建该文件夹
    main(args)
