import os
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import functional as TF 
from detectron2.checkpoint import DetectionCheckpointer
import warnings
import time

warnings.filterwarnings("ignore")

def load_model(model_name):
    """加载指定的模型"""
    try:
        torch.set_grad_enabled(False)
        
        # 选择权重文件
        if model_name == "SDMatte":
            checkpoint_path = "checkpoints/SDMatte/SDMatte.pth"
            # 直接导入模型类
            from modeling.SDMatte import SDMatte
            
            # 手动创建模型参数，避免加载配置文件
            model_kwargs = {
                'pretrained_model_name_or_path': "checkpoints/SDMatte",
                'load_weight': False,
                'conv_scale': 3,
                'num_inference_steps': 1,
                'aux_input': "bbox_mask",
                'add_noise': False,
                'use_dis_loss': True,
                'use_aux_input': True,
                'use_coor_input': True,
                'use_attention_mask': True,
                'residual_connection': False,
                'use_encoder_hidden_states': True,
                'use_attention_mask_list': [True, True, True],
                'use_encoder_hidden_states_list': [False, True, False],
            }
            model = SDMatte(**model_kwargs)
            
        elif model_name == "LiteSDMatte":
            checkpoint_path = "checkpoints/LiteSDMatte/LiteSDMatte.pth"
            # 直接导入模型类
            from modeling.LiteSDMatte import LiteSDMatte
            
            # 手动创建模型参数，避免加载配置文件
            model_kwargs = {
                'pretrained_model_name_or_path': "checkpoints/LiteSDMatte",
                'load_weight': False,
                'conv_scale': 3,
                'num_inference_steps': 1,
                'aux_input': "bbox_mask",
                'add_noise': False,
                'use_dis_loss': True,
                'use_aux_input': True,
                'use_coor_input': True,
                'use_attention_mask': True,
                'residual_connection': False,
                'use_encoder_hidden_states': True,
                'use_attention_mask_list': [True, True, True],
                'use_encoder_hidden_states_list': [False, True, False],
            }
            model = LiteSDMatte(**model_kwargs)
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # 加载权重
        DetectionCheckpointer(model).load(checkpoint_path)
        model.eval()
        
        print(f"✅ {model_name} 模型加载成功")
        return model, None  # 不返回cfg，因为我们不需要它
        
    except Exception as e:
        print(f"❌ {model_name} 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def preprocess_image(image):
    """预处理输入图像"""
    try:
        # 确保图像是PIL Image
        if isinstance(image, str):
            # 如果是文件路径，尝试打开
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be PIL Image, numpy array, or file path")
        
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 获取原始尺寸
        original_size = image.size
        
        # 转换为numpy array，保持0-255范围（后续在prepare_model_input中处理归一化）
        image_np = np.array(image).astype(np.float32)
        
        # 调整大小到512x512（与模型训练尺寸一致）
        image_resized = cv2.resize(image_np, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        return image_resized, original_size
        
    except Exception as e:
        print(f"图像预处理错误: {str(e)}")
        return None, None

def generate_simple_mask(image):
    """生成简单的前景掩码（基于图像中心区域）"""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    # 创建一个中心区域的掩码（椭圆形）
    center_h, center_w = h // 2, w // 2
    mask_h, mask_w = int(h * 0.6), int(w * 0.6)
    
    # 创建椭圆掩码
    y, x = np.ogrid[:h, :w]
    ellipse_mask = ((y - center_h) ** 2) / (mask_h/2) ** 2 + ((x - center_w) ** 2) / (mask_w/2) ** 2 <= 1
    mask[ellipse_mask] = 1.0
    
    return mask

def inference_single_image(image, model, model_name):
    """对单张图像进行推理"""
    try:
        start_time = time.time()
        
        # 预处理图像
        processed_image, original_size = preprocess_image(image)
        
        if processed_image is None or original_size is None:
            return None, 0
        
        # 生成简单掩码
        mask = generate_simple_mask(processed_image)
        
        # 准备模型输入（简化版，不使用trimap）
        model_input = prepare_model_input(processed_image, mask, None)
        
        if model_input is None:
            return None, 0
        
        # 模型推理
        inference_start = time.time()
        with torch.no_grad():
            output = model(model_input)
        inference_time = time.time() - inference_start
        
        # 后处理输出
        result_image = postprocess_output(output, original_size)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        print(f"✅ {model_name} 推理完成 - 总时间: {total_time:.3f}s (模型推理: {inference_time:.3f}s)")
        
        return result_image, total_time
        
    except Exception as e:
        print(f"推理错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0

def generate_bbox_from_mask(mask):
    """从掩码生成边界框坐标"""
    h, w = mask.shape
    coords = np.nonzero(mask)
    
    if coords[0].size == 0 or coords[1].size == 0:
        return np.array([0, 0, 1, 1])
    
    y_min, x_min = np.argwhere(mask).min(axis=0)
    y_max, x_max = np.argwhere(mask).max(axis=0)
    
    # 归一化坐标
    y_min, y_max = y_min / h, y_max / h
    x_min, x_max = x_min / w, x_max / w
    
    return np.array([x_min, y_min, x_max, y_max])

def prepare_model_input(image, mask, trimap):
    """准备模型输入数据"""
    try:
        # 图像从0-255范围直接标准化到[-1,1]
        image_normalized = (image / 255.0 - 0.5) / 0.5
        
        # 转换图像到tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float().cuda()
        
        # 转换掩码到tensor
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().cuda()
        
        # 生成边界框坐标
        bbox_coords = generate_bbox_from_mask(mask)
        bbox_coords_tensor = torch.from_numpy(bbox_coords).unsqueeze(0).float().cuda()
        
        # 创建基本数据字典
        data = {
            "image": image_tensor,
            "bbox_mask": mask_tensor,
            "bbox_coords": bbox_coords_tensor,
            "is_trans": torch.tensor([0]).cuda(),  # 0表示不透明物体
            "hw": torch.tensor([image.shape[0], image.shape[1]]).unsqueeze(0).cuda()
        }
        
        return data
        
    except Exception as e:
        print(f"准备模型输入错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_point_coords(mask, num_points=10):
    """生成点坐标（模拟GenPoint功能）"""
    height, width = mask.shape
    
    # 找到掩码中的前景像素
    mask_binary = (mask > 0.5).astype(np.float32)
    y_coords, x_coords = np.where(mask_binary == 1)
    
    if len(y_coords) < num_points:
        # 如果前景像素不够，返回零点坐标
        return np.zeros(20, dtype=np.float32)
    
    # 随机选择点
    selected_indices = np.random.choice(len(y_coords), size=num_points, replace=False)
    
    point_coords = []
    for idx in selected_indices:
        y_center = y_coords[idx]
        x_center = x_coords[idx]
        
        # 归一化坐标
        y_norm = y_center / height
        x_norm = x_center / width
        point_coords.extend([x_norm, y_norm])
    
    # 确保总是20个坐标（10个点，每个点2个坐标）
    if len(point_coords) < 20:
        point_coords.extend([0.0] * (20 - len(point_coords)))
    
    return np.array(point_coords, dtype=np.float32)

def generate_point_mask(mask, point_coords, radius=25):
    """生成点掩码（模拟GenPoint功能）"""
    import scipy.ndimage
    
    height, width = mask.shape
    point_mask = np.zeros_like(mask, dtype=np.float32)
    
    # 处理点坐标（每两个值代表一个点的x,y坐标）
    for i in range(0, len(point_coords), 2):
        if i + 1 < len(point_coords):
            x_norm, y_norm = point_coords[i], point_coords[i + 1]
            if x_norm == 0 and y_norm == 0:
                continue
                
            # 转换为像素坐标
            x_center = int(x_norm * width)
            y_center = int(y_norm * height)
            
            # 创建高斯掩码
            tmp_mask = np.zeros_like(mask, dtype=np.float32)
            if 0 <= y_center < height and 0 <= x_center < width:
                tmp_mask[y_center, x_center] = 1
                tmp_mask = scipy.ndimage.gaussian_filter(tmp_mask, sigma=radius)
                if np.max(tmp_mask) > 0:
                    tmp_mask /= np.max(tmp_mask)
                point_mask = np.maximum(point_mask, tmp_mask)
    
    return point_mask

def prepare_model_input(image, mask, trimap):
    """准备模型输入数据"""
    try:
        # 图像从0-255范围直接标准化到[-1,1]
        image_normalized = (image / 255.0 - 0.5) / 0.5
        
        # 转换图像到tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float().cuda()
        
        # 转换掩码到tensor
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().cuda()
        
        # 生成边界框坐标
        bbox_coords = generate_bbox_from_mask(mask)
        bbox_coords_tensor = torch.from_numpy(bbox_coords).unsqueeze(0).float().cuda()
        
        # 创建基本数据字典
        data = {
            "image": image_tensor,
            "bbox_mask": mask_tensor,
            "bbox_coords": bbox_coords_tensor,
            "is_trans": torch.tensor([0]).cuda(),  # 0表示不透明物体
            "hw": torch.tensor([image.shape[0], image.shape[1]]).unsqueeze(0).cuda()
        }
        
        return data
        
        return data
        
    except Exception as e:
        print(f"准备模型输入错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def postprocess_output(output, original_size):
    """后处理模型输出"""
    try:
        # 按照官方inference.py的处理方式
        # output = pred.flatten(0, 2) * 255
        output = output.flatten(0, 2) * 255
        
        # 将输出转换为numpy array
        output = output.detach().cpu().numpy()
        
        # 确保输出在正确范围内
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # 调整回原始尺寸
        output_resized = cv2.resize(output, original_size, interpolation=cv2.INTER_LINEAR)
        
        # 转换为PIL Image（官方使用F.to_pil_image(output).convert("RGB")）
        from torchvision.transforms import functional as F
        output_pil = F.to_pil_image(output_resized).convert("RGB")
        
        return output_pil
        
    except Exception as e:
        print(f"后处理错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def matting_inference(image):
    """主推理函数，同时使用SDMatte和LiteSDMatte处理图像"""
    if image is None:
        return None, None, "请上传图像"
    
    try:
        total_start_time = time.time()
        print("🚀 开始推理...")
        
        # 加载SDMatte模型
        print("📥 加载 SDMatte 模型...")
        sdmatte_model, sdmatte_cfg = load_model("SDMatte")
        
        if sdmatte_model is None:
            return None, None, "SDMatte 模型加载失败"
        
        print("🔄 使用 SDMatte 进行推理...")
        sdmatte_result, sdmatte_time = inference_single_image(image, sdmatte_model, "SDMatte")
        
        # 清理内存
        del sdmatte_model
        torch.cuda.empty_cache()
        
        if sdmatte_result is None:
            return None, None, "SDMatte 推理失败"
        
        # 加载LiteSDMatte模型
        print("📥 加载 LiteSDMatte 模型...")
        litesdmatte_model, litesdmatte_cfg = load_model("LiteSDMatte")
        
        if litesdmatte_model is None:
            status_msg = f"LiteSDMatte 模型加载失败，但 SDMatte 推理成功\n⏱️ SDMatte 推理时间: {sdmatte_time:.3f}s"
            return sdmatte_result, None, status_msg
        
        print("🔄 使用 LiteSDMatte 进行推理...")
        litesdmatte_result, litesdmatte_time = inference_single_image(image, litesdmatte_model, "LiteSDMatte")
        
        # 清理内存
        del litesdmatte_model
        torch.cuda.empty_cache()
        
        total_time = time.time() - total_start_time
        
        if litesdmatte_result is None:
            status_msg = f"LiteSDMatte 推理失败，但 SDMatte 推理成功\n⏱️ SDMatte 推理时间: {sdmatte_time:.3f}s"
            return sdmatte_result, None, status_msg
        
        # 计算速度比较
        speed_comparison = ""
        if sdmatte_time > 0 and litesdmatte_time > 0:
            if sdmatte_time > litesdmatte_time:
                ratio = sdmatte_time / litesdmatte_time
                speed_comparison = f"\n🚀 LiteSDMatte 比 SDMatte 快 {ratio:.1f}x"
            else:
                ratio = litesdmatte_time / sdmatte_time
                speed_comparison = f"\n🚀 SDMatte 比 LiteSDMatte 快 {ratio:.1f}x"
        
        status_msg = f"""✅ 推理完成
⏱️ 推理时间统计:
   • SDMatte: {sdmatte_time:.3f}s
   • LiteSDMatte: {litesdmatte_time:.3f}s
   • 总时间: {total_time:.3f}s{speed_comparison}"""
        
        print("✅ 推理完成")
        return sdmatte_result, litesdmatte_result, status_msg
        
    except Exception as e:
        error_msg = f"推理过程中出现错误: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, error_msg

# 创建Gradio界面
def create_interface():
    """创建Gradio Web界面"""
    
    with gr.Blocks(title="SDMatte & LiteSDMatte Web UI", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; margin: 20px;">
            <h1>🎭 SDMatte & LiteSDMatte 图像抠图系统</h1>
            <p>上传图像，使用SDMatte和LiteSDMatte模型进行自动抠图处理</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📤 输入</h3>")
                input_image = gr.Image(
                    label="上传图像",
                    type="pil",
                    height=400
                )
                
                inference_button = gr.Button(
                    "🚀 开始推理",
                    variant="primary",
                    size="lg"
                )
                
                status_text = gr.Textbox(
                    label="状态信息",
                    value="等待上传图像...",
                    interactive=False,
                    lines=6,
                    max_lines=10
                )
            
            with gr.Column(scale=2):
                gr.HTML("<h3>📤 输出结果</h3>")
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h4>SDMatte 结果</h4>")
                        sdmatte_output = gr.Image(
                            label="SDMatte 抠图结果",
                            type="pil",
                            height=300
                        )
                    
                    with gr.Column():
                        gr.HTML("<h4>LiteSDMatte 结果</h4>")
                        litesdmatte_output = gr.Image(
                            label="LiteSDMatte 抠图结果", 
                            type="pil",
                            height=300
                        )
        
        # 使用说明
        gr.HTML("""
        <div style="margin: 20px; padding: 15px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 10px; backdrop-filter: blur(10px);">
            <h3 style="color: var(--body-text-color, #333); margin-bottom: 15px;">📋 使用说明</h3>
            <ul style="color: var(--body-text-color, #666); line-height: 1.6;">
                <li>🖼️ 支持常见图像格式：JPG、PNG、JPEG等</li>
                <li>🎯 系统会自动生成前景区域掩码</li>
                <li>⚡ 同时使用两个模型进行处理，对比效果</li>
                <li>🔄 处理完成后会显示两个模型的抠图结果</li>
                <li>💾 可以右键保存结果图像</li>
                <li>⏱️ 推理时间会显示在状态栏中</li>
            </ul>
        </div>
        """)
        
        # 绑定事件
        inference_button.click(
            fn=matting_inference,
            inputs=[input_image],
            outputs=[sdmatte_output, litesdmatte_output, status_text],
            show_progress=True
        )
        
        # 示例
        gr.Examples(
            examples=[],
            inputs=input_image,
            label="📚 示例图像（如果有的话）"
        )
    
    return demo

if __name__ == "__main__":
    # 检查必要的文件是否存在
    required_files = [
        "checkpoints/SDMatte/SDMatte.pth",
        "checkpoints/LiteSDMatte/LiteSDMatte.pth"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ 缺少必要文件:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n请确保模型文件存在后重新运行。")
        exit(1)
    
    print("✅ 所有必要文件已找到")
    print("🚀 启动 Gradio Web UI...")
    
    # 创建并启动界面
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 默认端口
        share=False,            # 不创建公共链接
        debug=True              # 调试模式
    )
