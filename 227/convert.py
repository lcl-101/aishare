import io
import os
import sys
import numpy as np
import torch
import torch.onnx

# 添加 U-2-Net 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'U-2-Net'))

from model import U2NET

# 配置路径
base_dir = os.path.dirname(os.path.abspath(__file__))
model_input_path = os.path.join(base_dir, 'models', 'custom.pth')  # 输入的 PyTorch 模型
model_output_path = os.path.join(base_dir, 'models', 'custom.onnx')  # 输出的 ONNX 模型

batch_size = 1

# 创建模型
torch_model = U2NET(3, 1)

# 加载权重
print(f"正在加载模型: {model_input_path}")
torch_model.load_state_dict(torch.load(model_input_path, map_location='cpu'))
torch_model.eval()

# 创建示例输入
x = torch.randn(batch_size, 3, 320, 320, requires_grad=True)

# 测试前向传播
print("测试模型前向传播...")
torch_out = torch_model(x)
print(f"输出形状: {torch_out[0].shape}")

# 导出 ONNX 模型
print(f"正在导出 ONNX 模型到: {model_output_path}")
torch.onnx.export(
    torch_model, 
    x, 
    model_output_path, 
    export_params=True, 
    opset_version=16, 
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"✅ ONNX 模型导出成功: {model_output_path}")
print(f"文件大小: {os.path.getsize(model_output_path) / (1024*1024):.2f} MB")