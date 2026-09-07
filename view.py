import torch

# 加载.pth文件
checkpoint = torch.load('./save_weights/best_model.pth', map_location='cpu')  # 使用CPU加载更安全

# 查看文件包含的键（keys）
print("Checkpoint keys:", checkpoint.keys())

# 如果有模型参数（通常是'state_dict'或'model'）
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint  # 如果直接保存的是state_dict

# 打印模型结构（参数名）
print("\nModel layers:")
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")  # 打印每一层的名称和维度