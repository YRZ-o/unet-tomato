import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import warnings
from datetime import datetime

from src import AttentionResUNet

# 忽略PyTorch关于权重的警告
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


# ============================== 模型定义 ==============================
# [模型定义保持不变]

# ============================== 预测函数 ==============================
def predict_image(model, image_path, device, show_results=True, save_results=False, num_classes=2):
    """
    使用训练好的模型对单张图像进行预测

    参数:
    model -- 训练好的模型
    image_path -- 要预测的图像路径
    device -- 使用的设备 (CPU或GPU)
    show_results -- 是否显示结果（默认True）
    save_results -- 是否保存结果（默认False）
    num_classes -- 模型输出的类别数（包含背景）
    """
    # 1. 加载图像
    original_image = Image.open(image_path).convert("RGB")
    original_width, original_height = original_image.size
    print(f"Original image size: {original_width}x{original_height}")

    # 2. 预处理（与训练时相同）
    # 使用训练时的均值和标准差
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    target_size = (480, 480)  # 训练时的图像大小

    # 创建转换管道
    transform = transforms.Compose([
        transforms.Resize(target_size),  # 调整到训练时的大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=mean, std=std)  # 归一化
    ])

    # 应用转换
    image_tensor = transform(original_image).unsqueeze(0)  # 增加batch维度

    # 3. 模型预测
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)

    # 4. 处理输出
    logits = output["out"]

    # 根据类别数采用不同的处理方式
    if num_classes > 2:  # 多分类
        probs = torch.softmax(logits, dim=1)
        # 去掉batch维度 -> [C, H, W]
        pred_mask = torch.argmax(probs, dim=1).cpu().squeeze(0).numpy()  # [H, W]
    else:  # 二分类
        # 使用sigmoid处理（适用于单类分割）
        probs = torch.sigmoid(logits)
        # 获取前景掩码 - 确保得到二维数组
        pred_mask =1 - (probs > 0.5).float().cpu().squeeze(0).squeeze(0).numpy()  # [H, W]

    print(f"Predicted mask shape: {pred_mask.shape}")

    # 5. 确保预测掩码是二维的
    if len(pred_mask.shape) > 2:
        # 取通道0作为前景掩码
        pred_mask = pred_mask[0, :, :]
        print(f"Adjusted predicted mask shape: {pred_mask.shape}")

    # 6. 将预测掩码调整回原始图像大小
    # 将预测掩码转换为PIL图像以调整大小
    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    pred_mask_resized = pred_mask_img.resize((original_width, original_height), Image.NEAREST)
    pred_mask_resized_np = np.array(pred_mask_resized) / 255.0

    # 7. 创建带有透明度的叠加效果 - 使用原始图像尺寸
    # 原始图像转换为数组
    original_np = np.array(original_image) / 255.0
    # 将预测掩码转换为彩色遮罩（这里用红色表示前景）
    mask_red = np.zeros_like(original_np)
    mask_red[..., 0] = 1.0  # 红色通道设为全饱和

    # 只在前景区域应用红色
    overlay = original_np.copy()
    # 使用调整大小后的掩码
    overlay[pred_mask_resized_np > 0] = overlay[pred_mask_resized_np > 0] * 0.5 + mask_red[
        pred_mask_resized_np > 0] * 0.5

    # 8. 保存结果
    base_name = os.path.basename(image_path).split('.')[0]
    output_files = []

    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if save_results:
        # 保存原始图像
        orig_path = f"results/{base_name}_{current_time}_original.png"
        original_image.save(orig_path)
        output_files.append(orig_path)

        # 保存预测掩码（原始大小）
        mask_path = f"results/{base_name}_{current_time}_prediction.png"
        pred_mask_resized.save(mask_path)
        output_files.append(mask_path)

        # 保存叠加图像（原始大小）
        overlay_path = f"results/{base_name}_{current_time}_overlay.png"
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        overlay_img.save(overlay_path)
        output_files.append(overlay_path)

        print(f"Saved results to: {', '.join(output_files)}")

    # 9. 显示结果 - 现在总是尝试显示
    try:
        # 创建图像显示
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # 原始图像
        axs[0].imshow(original_image)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # 预测结果
        axs[1].imshow(pred_mask_resized_np, cmap='gray')
        axs[1].set_title("Segmentation Mask")
        axs[1].axis('off')

        # 叠加效果
        axs[2].imshow(overlay)
        axs[2].set_title("Overlay")
        axs[2].axis('off')

        plt.tight_layout()

        if show_results:
            print("Showing result visualization...")
            plt.show()
        else:
            # 即使不主动显示，也尝试保存图像
            fig_path = f"results/{base_name}_{current_time}_visualization.png"
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization to: {fig_path}")
            output_files.append(fig_path)
    except Exception as e:
        print(f"Could not display results: {e}. Saving visualization instead.")
        fig_path = f"{base_name}_{current_time}_visualization.png"
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to: {fig_path}")
        output_files.append(fig_path)
        # 10. 添加直接展示预测结果的代码（新添加部分）
        if show_results:
            print("Showing segmentation visualization in a new window...")
            plt.figure(figsize=(15, 5))

            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis('off')

            # 预测结果
            plt.subplot(1, 3, 2)
            plt.imshow(pred_mask_resized_np, cmap='gray')
            plt.title("Segmentation Mask")
            plt.axis('off')

            # 叠加效果
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title("Overlay")
            plt.axis('off')

            plt.tight_layout()
            plt.show()
        else:
            print("To see visualization, run with --show option")
    return output_files

# ============================== 主函数 ==============================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Predict tomato segmentation')
    parser.add_argument('--image', type=str, default='./tomato/predict/tomato_0711.jpg',
                        help='Path to the image to predict')
    parser.add_argument('--model', type=str, default='save_weights/best_model_AttentionResUNet_tomatoNewDataset.pth',
                        help='Path to the trained model weights')
    parser.add_argument('--show', default='true',
                        help='Show the results')
    parser.add_argument('--save', default='true',
                        help='Save the results')
    parser.add_argument('--num-classes', type=int, default=1,
                        help='Number of object classes (excluding background)')
    args = parser.parse_args()

    # 设备配置
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 重要：加载模型时使用与训练时相同的类别数
    # 训练时：num_classes_total = args.num_classes + 1
    # 这里我们同样处理：真实类别数 + 背景
    num_classes_total = args.num_classes + 1

    # 创建模型 - 使用正确的输出通道数
    model = AttentionResUNet(in_channels=3, num_classes=num_classes_total, base_c=32)

    # 加载整个检查点
    print(f"Loading checkpoint from {args.model}")

    # 使用正确的设备加载
    if torch.cuda.is_available():
        checkpoint = torch.load(args.model, map_location=device)
    else:
        checkpoint = torch.load(args.model, map_location=torch.device('cpu'))

    # 提取模型权重并加载
    model.load_state_dict(checkpoint['model'])

    model = model.to(device)
    print(f"Model weights successfully loaded. Output classes: {num_classes_total}")

    # 检查图像文件是否存在
    if not os.path.exists(args.image):
        print(f"Warning: Image file '{args.image}' not found. Using default test image.")
        # 创建简单的测试图像
        test_image = np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8)
        Image.fromarray(test_image).save('test.jpg')
        args.image = 'test.jpg'

    # 进行预测
    predict_image(
        model=model,
        image_path=args.image,
        device=device,
        show_results=args.show,
        save_results=args.save,
        num_classes=num_classes_total
    )


if __name__ == '__main__':
    main()