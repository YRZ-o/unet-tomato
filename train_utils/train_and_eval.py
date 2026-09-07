import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval() # 将模型设置为评估模式
    confmat = utils.ConfusionMatrix(num_classes)
    # 创建一个混淆矩阵对象，用于记录分类结果的混淆情况
    # num_classes参数指定类别数
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    # 创建一个Dice系数对象，用于计算Dice系数
    # num_classes参数指定类别数，ignore_index指定在计算时要忽略的标签值
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 创建一个MetricLogger对象，用于记录评估过程中的指标
    # delimiter参数指定指标之间的分隔符
    header = 'Test:' # 构建评估过程的头部信息
    with torch.no_grad(): # 在不计算梯度的情况下进行评估
        for image, target in metric_logger.log_every(data_loader, 100, header):
            # 遍历数据加载器中的数据
            # metric_logger.log_every函数会按照指定的频率打印评估进度
            image, target = image.to(device), target.to(device)
            output = model(image)  # 模型前向传播，得到输出
            output = output['out'] # 假设模型输出是一个字典，提取其中的'out'键对应的值作为最终输出

            confmat.update(target.flatten(), output.argmax(1).flatten())
            # 更新混淆矩阵
            # target.flatten()将目标张量展平
            # output.argmax(1).flatten()对输出结果在维度1上取最大值的索引，并展平
            dice.update(output, target) # 更新Dice系数

        confmat.reduce_from_all_processes() # 在多进程环境下，将各个进程的混淆矩阵结果进行合并
        dice.reduce_from_all_processes() # 在多进程环境下，将各个进程的Dice系数结果进行合并

    return confmat, dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train() # 将模型设置为训练模式
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 创建一个MetricLogger对象，用于记录训练过程中的指标
    # delimiter参数指定了指标之间的分隔符
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 为MetricLogger添加一个用于记录学习率的指标
    # window_size表示平滑窗口大小，fmt指定了输出格式
    header = 'Epoch: [{}]'.format(epoch) # 构建当前训练轮次的头部信息

    if num_classes == 2: # 如果类别数为2，设置交叉熵损失中背景和前景的损失权重 , 这里的权重可以根据数据集的情况进行调整
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None # 如果类别数不为2，不设置损失权重
    # 遍历数据加载器中的数据
    # metric_logger.log_every函数会按照指定的频率打印训练进度

    # VOC 数据集有 21 个类别（包括背景）
    # loss_weight = None
    # 假设已经统计好每个类别的权重，存储在一个列表中
    # class_weights = [0.5, 0.8, 1.2, 1.0, 0.9, 1.1, 0.7, 1.3, 0.6, 1.4, 0.75, 1.25, 0.85, 1.15, 0.95, 1.05, 0.65, 1.35,
    #                  0.55, 1.45, 0.7]
    # loss_weight = torch.as_tensor(class_weights, device=device)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)  # 将图像和目标数据移动到指定的设备上（如GPU）
        # 使用torch.cuda.amp.autocast进行自动混合精度训练
        # 如果scaler不为None，则启用自动混合精度
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image) # 模型前向传播，得到输出
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)
            # 计算损失，这里的criterion应该是一个已经定义好的损失函数
            # loss_weight用于指定损失权重，num_classes为类别数，ignore_index指定忽略的标签值
        optimizer.zero_grad() # 清空优化器的梯度

        if scaler is not None:
            scaler.scale(loss).backward()
            # 如果使用自动混合精度训练
            # 对损失进行缩放后反向传播计算梯度
            scaler.step(optimizer) # 调用scaler.step更新优化器参数
            scaler.update()  # 更新scaler

        else: # 如果不使用自动混合精度训练，直接反向传播计算梯度
            loss.backward()
            optimizer.step() # 更新优化器参数

        lr_scheduler.step() # 更新学习率调度器

        lr = optimizer.param_groups[0]["lr"] # 获取当前的学习率
        metric_logger.update(loss=loss.item(), lr=lr) # 更新MetricLogger中的损失和学习率指标

    return metric_logger.meters["loss"].global_avg, lr # 返回训练过程中的平均损失和当前学习率


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0  # 断言确保num_step和epochs都大于0
    if warmup is False:
        warmup_epochs = 0 # 如果不进行warmup，将warmup_epochs设为0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step): # 如果处于warmup阶段且当前step数小于等于warmup_epochs * num_step
            alpha = float(x) / (warmup_epochs * num_step) # 计算当前step在warmup阶段的比例
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
            # 计算当前step在总训练step数（除去warmup阶段）中的比例
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
    # 返回一个LambdaLR学习率调度器对象