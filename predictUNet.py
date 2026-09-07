import os
import time
import warnings
from datetime import datetime

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet

# 忽略PyTorch关于权重的警告
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def predict_image(model, image_path, device, show_results=True, save_results=True, num_classes=2):
    """
    使用训练好的模型对单张图像进行预测

    参数:
    model -- 训练好的模型
    image_path -- 要预测的图像路径
    device -- 使用的设备 (CPU或GPU)
    show_results -- 是否显示结果（默认True）
    save_results -- 是否保存结果（默认True）
    num_classes -- 模型输出的类别数（包含背景）
    """
    # 1. 加载图像
    original_image = Image.open(image_path).convert("RGB")
    original_width, original_height = original_image.size
    print(f"Original image size: {original_width}x{original_height}")

    # 2. 预处理（与训练时相同）
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
        # 初始化模型（可选，用于预热）
        img_height, img_width = image_tensor.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        # 计时预测
        t_start = time_synchronized()
        output = model(image_tensor.to(device))
        t_end = time_synchronized()
        print(f"Inference time: {t_end - t_start:.4f}s")

    # 4. 处理输出
    logits = output['out']

    # 根据类别数采用不同的处理方式
    if num_classes > 2:  # 多分类
        probs = torch.softmax(logits, dim=1)
        # 去掉batch维度 -> [C, H, W]
        pred_mask = torch.argmax(probs, dim=1).cpu().squeeze(0).numpy()  # [H, W]
    else:  # 二分类
        # 使用argmax处理（适用于二分类）
        prediction = logits.argmax(1).squeeze(0)
        pred_mask = prediction.cpu().numpy().astype(np.uint8)
        # 将前景对应的像素值改成1
        pred_mask[pred_mask == 1] = 1

    print(f"Predicted mask shape: {pred_mask.shape}")

    # 5. 确保预测掩码是二维的
    if len(pred_mask.shape) > 2:
        pred_mask = pred_mask[0, :, :]
        print(f"Adjusted predicted mask shape: {pred_mask.shape}")

    # 6. 将预测掩码调整回原始图像大小
    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    pred_mask_resized = pred_mask_img.resize((original_width, original_height), Image.NEAREST)
    pred_mask_resized_np = np.array(pred_mask_resized) / 255.0

    # 7. 创建带有透明度的叠加效果
    original_np = np.array(original_image) / 255.0
    mask_red = np.zeros_like(original_np)
    mask_red[..., 0] = 1.0  # 红色通道设为全饱和

    # 只在前景区域应用红色
    overlay = original_np.copy()
    overlay[pred_mask_resized_np > 0] = overlay[pred_mask_resized_np > 0] * 0.5 + mask_red[
        pred_mask_resized_np > 0] * 0.5

    # 8. 保存结果
    base_name = os.path.basename(image_path).split('.')[0]
    output_files = []

    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)

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

    # 9. 显示结果
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
            # 即使不主动显示，也保存可视化图像
            fig_path = f"results/{base_name}_{current_time}_visualization.png"
            plt.savefig(fig_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved visualization to: {fig_path}")
            output_files.append(fig_path)
    except Exception as e:
        print(f"Could not display results: {e}. Saving visualization instead.")
        fig_path = f"results/{base_name}_{current_time}_visualization.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved visualization to: {fig_path}")
        output_files.append(fig_path)

    return output_files


def main():
    classes = 1  # exclude background
    weights_path = "./save_weights/best_model_UNet_tomatoNewDataset.pth"
    img_path = "./tomato/predict/tomato_2976.jpg"

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    # get devices
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False)['model'])
    model.to(device)
    print("Model loaded successfully.")

    # 进行预测
    predict_image(
        model=model,
        image_path=img_path,
        device=device,
        show_results=True,
        save_results=True,
        num_classes=classes + 1
    )


if __name__ == '__main__':
    main()