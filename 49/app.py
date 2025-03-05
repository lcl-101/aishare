import gradio as gr
import torch
from PIL import Image
import numpy as np
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from gradio_imageslider import ImageSlider
import tempfile
import os  # 新增导入


# Helper function to load model from Hugging Face
def load_model_by_name(arch_name, checkpoint_path, device):
    model = None
    if arch_name == 'depthanything':
        # 使用 safetensors 加载模型权重
        model_weights = load_file(checkpoint_path)  # safetensors 加载方式
        
        # 初始化模型
        model = DepthAnything(checkpoint_path=None).to(device)
        model.load_state_dict(model_weights)  # 将加载的权重应用到模型

        model = model.to(device)  # 确保模型在正确的设备上
    else:
        raise NotImplementedError(f"Unknown architecture: {arch_name}")
    return model

# Image processing function
def process_image(image, model, device):
    if model is None:
        return None
    
    # Preprocess the image
    image_np = np.array(image)[..., ::-1] / 255
    
    transform = Compose([
        Resize(756, 756, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])
    
    image_tensor = transform({'image': image_np})['image']
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():  # Disable autograd since we don't need gradients on CPU
        pred_disp, _ = model(image_tensor)

    # Ensure the depth map is in the correct shape before colorization
    pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]  # Remove extra singleton dimensions
    
    # Normalize depth map
    pred_disp = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())
    
    # Colorize depth map
    cmap = "Spectral_r"
    depth_colored = colorize_depth_maps(pred_disp[None, ..., None], 0, 1, cmap=cmap).squeeze()  # Ensure correct dimension
    
    # Convert to uint8 for image display
    depth_colored = (depth_colored * 255).astype(np.uint8)
    
    # Convert to HWC format (height, width, channels)
    depth_colored_hwc = chw2hwc(depth_colored)
    
    # Resize to match the original image dimensions (height, width)
    h, w = image_np.shape[:2]
    depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)
    
    # Convert to a PIL image
    depth_image = Image.fromarray(depth_colored_hwc)
    return image, depth_image

# Gradio interface function with GPU support
def gradio_interface(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = dict(
        vitb=dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        vitl=dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        )
    )
    # Load model
    model = DepthAnything(**model_kwargs['vitl']).to(device)
    #checkpoint_path = hf_hub_download(repo_id=f"checkpoint/model.safetensors", filename=f"large/model.safetensors", repo_type="model")

    # 使用 safetensors 加载模型权重
    model_weights = load_file("checkpoint/model.safetensors")  # safetensors 加载方式
    model.load_state_dict(model_weights)
    model = model.to(device)  # 确保模型在正确的设备上
    
    if model is None:
        return None
    
    # Process image and return output
    _, depth = process_image(image, model, device)
    # 设置输出目录，确保该目录被 Gradio 服务可访问
    output_dir = "/home/softice/Distill-Any-Depth/tmp"
    os.makedirs(output_dir, exist_ok=True)
    # 保存图片到指定目录，并返回文件路径
    depth_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=output_dir)
    depth.save(depth_tmp.name)
    return depth_tmp.name

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),  # Only image input, no mode selection
    outputs=gr.Image(type="filepath", label="Depth Image"),  # 显示单一图片
    title="Depth Estimation Demo",
    description="Upload an image to see the depth estimation results. Our model is running on GPU for faster processing.",
    examples=["1.jpg", "2.jpg", "4.png", "5.jpg", "6.jpg"],
    cache_examples=False,  # 不预运行示例，等待处理完成后再更新结果
)

# Launch the Gradio interface
iface.launch(server_name='0.0.0.0')