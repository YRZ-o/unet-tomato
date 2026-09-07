# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with identity shortcut"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class AttentionGate(nn.Module):
    """重构的注意力门，确保通道一致性"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # 门控信号处理
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        # 特征信号处理
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        # 注意力系数生成
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 统一空间尺寸
        if g.size()[2:] != x.size()[2:]:
            diff_y = x.size()[2] - g.size()[2]
            diff_x = x.size()[3] - g.size()[3]
            g = F.pad(g, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        # 处理门控信号
        W_g = self.W_g(g)
        W_x = self.W_x(x)

        # 确保尺寸匹配
        if W_g.size() != W_x.size():
            diff_y = W_x.size()[2] - W_g.size()[2]
            diff_x = W_x.size()[3] - W_g.size()[3]
            W_g = F.pad(W_g, [diff_x // 2, diff_x - diff_x // 2,
                              diff_y // 2, diff_y - diff_y // 2])

        # 组合信号并计算注意力
        combined = self.relu(W_g + W_x)
        psi = self.psi(combined)

        # 应用注意力
        return x * psi


class Down(nn.Sequential):
    """下采样模块 - 最大池化+残差块"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            ResidualBlock(in_channels, out_channels)
        )


class Up(nn.Module):
    """上采样模块 - 含注意力门机制"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.bilinear = bilinear

        # 上采样方法选择
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # 注意力门
        # g 信号来自上采样后的低层特征 (in_channels // 2)
        # x 信号来自编码器路径 (out_channels)
        self.attention = AttentionGate(
            F_g=in_channels if bilinear else in_channels // 2,
            F_l=out_channels,
            F_int=out_channels // 2
        )

        # 卷积块处理
        # 输入 = 上采样信号通道 + 注意力门输出通道
        conv_in_channels = (in_channels if bilinear else in_channels // 2) + out_channels
        self.conv = ResidualBlock(conv_in_channels, out_channels)

    def forward(self, g, x):
        # g: 来自解码器较低层的特征
        # x: 来自编码器对应层的特征

        # 上采样信号
        g_up = self.up(g)

        # 确保空间尺寸匹配
        if g_up.size()[2:] != x.size()[2:]:
            diff_y = x.size()[2] - g_up.size()[2]
            diff_x = x.size()[3] - g_up.size()[3]
            g_up = F.pad(g_up, [diff_x // 2, diff_x - diff_x // 2,
                                diff_y // 2, diff_y - diff_y // 2])

        # 应用注意力机制
        x_att = self.attention(g=g_up, x=x)

        # 连接特征
        x_cat = torch.cat([g_up, x_att], dim=1)

        # 通过卷积块
        return self.conv(x_cat)


class OutConv(nn.Sequential):
    """输出卷积块"""

    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class AttentionResUNet(nn.Module):
    """完全重构的AttentionResUNet，确保通道一致性"""

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(AttentionResUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1  # 用于通道调整因子

        # 编码器路径 (下采样)
        self.inc = ResidualBlock(in_channels, base_c)  # [B, base_c, H, W]
        self.down1 = Down(base_c, base_c * 2)  # [B, base_c*2, H/2, W/2]
        self.down2 = Down(base_c * 2, base_c * 4)  # [B, base_c*4, H/4, W/4]
        self.down3 = Down(base_c * 4, base_c * 8)  # [B, base_c*8, H/8, W/8]
        self.down4 = Down(base_c * 8, base_c * 16 // factor)  # [B, base_c*16//factor, H/16, W/16]

        # 解码器路径 (上采样+注意力门)
        self.up1 = Up(base_c * 16 // factor, base_c * 8, bilinear)  # 输入通道 = 瓶颈层输出通道
        self.up2 = Up(base_c * 8, base_c * 4, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        # 输出层
        self.outc = OutConv(base_c, num_classes)

    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)  # [B, base_c, H, W]
        x2 = self.down1(x1)  # [B, base_c*2, H/2, W/2]
        x3 = self.down2(x2)  # [B, base_c*4, H/4, W/4]
        x4 = self.down3(x3)  # [B, base_c*8, H/8, W/8]
        x5 = self.down4(x4)  # [B, base_c*16//factor, H/16, W/16]

        # 解码器路径
        d = self.up1(x5, x4)  # 上采样并应用注意力门
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)

        # 输出
        logits = self.outc(d)
        return {"out": logits}