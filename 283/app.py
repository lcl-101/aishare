import os
import math
import cv2
import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
from skimage import color, img_as_float32, img_as_ubyte
from diffusers import FluxPipeline
from Genfocus.pipeline.flux import Condition, generate, seed_everything

import depth_pro


# ============== 模型路径配置 ==============
# 使用本地已下载的模型
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.path.join(BASE_PATH, "checkpoints", "FLUX.1-dev")
GENFOCUS_MODEL_PATH = os.path.join(BASE_PATH, "checkpoints", "Genfocus-Model")
DEBLUR_LORA_PATH = GENFOCUS_MODEL_PATH
DEBLUR_WEIGHT_NAME = "deblurNet.safetensors"
BOKEH_LORA_DIR = GENFOCUS_MODEL_PATH
BOKEH_WEIGHT_NAME = "bokehNet.safetensors"

# Depth Pro 模型路径
DEPTH_PRO_CKPT_PATH = os.path.join(GENFOCUS_MODEL_PATH, "checkpoints", "depth_pro.pt")
DEPTH_PRO_DEFAULT_PATH = os.path.join(BASE_PATH, "checkpoints", "depth_pro.pt")

# 检查模型文件是否存在
if not os.path.exists(os.path.join(DEBLUR_LORA_PATH, DEBLUR_WEIGHT_NAME)):
    print(f"❌ 警告: 未找到 {DEBLUR_WEIGHT_NAME}")
if not os.path.exists(os.path.join(BOKEH_LORA_DIR, BOKEH_WEIGHT_NAME)):
    print(f"❌ 警告: 未找到 {BOKEH_WEIGHT_NAME}")
if not os.path.exists(DEPTH_PRO_CKPT_PATH):
    print(f"❌ 警告: 未找到 Depth Pro 模型: {DEPTH_PRO_CKPT_PATH}")

# 创建 depth_pro.pt 的符号链接到默认位置
if os.path.exists(DEPTH_PRO_CKPT_PATH) and not os.path.exists(DEPTH_PRO_DEFAULT_PATH):
    try:
        os.symlink(DEPTH_PRO_CKPT_PATH, DEPTH_PRO_DEFAULT_PATH)
        print(f"✅ 已创建 Depth Pro 模型符号链接: {DEPTH_PRO_DEFAULT_PATH}")
    except Exception as e:
        print(f"⚠️ 创建符号链接失败: {e}")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"🚀 检测到设备: {device}")


# ============== 加载 FLUX 模型 ==============
print("🔄 正在加载 FLUX 模型...")
pipe_flux = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
current_adapter = None

if device == "cuda":
    print("🚀 正在将 FLUX 移动到 CUDA...")
    pipe_flux.to("cuda")


# ============== 加载 Depth Pro 模型 ==============
print("🔄 正在加载 Depth Pro 模型...")
try:
    depth_model, depth_transform = depth_pro.create_model_and_transforms(
        device=torch.device(device)
    )
    depth_model.eval()
    print("✅ Depth Pro 加载完成。")
except Exception as e:
    print(f"❌ 加载 Depth Pro 失败: {e}")
    depth_model = None
    depth_transform = None


def resize_and_pad_image(img: Image.Image, force_512: bool) -> Image.Image:
    """
    控制图像预处理:
    - 如果 force_512=True: 将长边调整为 512，然后裁剪到最近的 16 的倍数。
    - 如果 force_512=False: 将宽高上采样/调整到最近的 16 的倍数上限。
    """
    w, h = img.size

    if force_512:
        target_max = 512
        if w >= h:
            new_w = target_max
            scale = target_max / w
            new_h = int(h * scale)
        else:
            new_h = target_max
            scale = target_max / h
            new_w = int(w * scale)
        
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        final_w = (new_w // 16) * 16
        final_h = (new_h // 16) * 16
        
        left = (new_w - final_w) // 2
        top = (new_h - final_h) // 2
        right = left + final_w
        bottom = top + final_h
        
        return img.crop((left, top, right, bottom))
    
    else:
        final_w = ((w + 15) // 16) * 16
        final_h = ((h + 15) // 16) * 16
        
        if final_w == w and final_h == h:
            return img
        
        return img.resize((final_w, final_h), Image.LANCZOS)


def switch_lora(target_mode):
    """切换 LoRA 适配器"""
    global pipe_flux, current_adapter
    if current_adapter == target_mode:
        return
    print(f"🔄 正在切换 LoRA 到 [{target_mode}]...")
    pipe_flux.unload_lora_weights()
    if target_mode == "deblur":
        try:
            pipe_flux.load_lora_weights(DEBLUR_LORA_PATH, weight_name=DEBLUR_WEIGHT_NAME, adapter_name="deblurring")
            pipe_flux.set_adapters(["deblurring"])
            current_adapter = "deblur"
        except Exception as e:
            print(f"❌ 加载 Deblur LoRA 失败: {e}")
    elif target_mode == "bokeh":
        try:
            pipe_flux.load_lora_weights(BOKEH_LORA_DIR, weight_name=BOKEH_WEIGHT_NAME, adapter_name="bokeh")
            pipe_flux.set_adapters(["bokeh"])
            current_adapter = "bokeh"
        except Exception as e:
            print(f"❌ 加载 Bokeh LoRA 失败: {e}")


def preprocess_input_image(raw_img, force_512):
    """预处理输入图像"""
    if raw_img is None:
        return None, None, None
    
    mode_str = "调整最长边为 512" if force_512 else "原始分辨率 (对齐到 16)"
    print(f"🔄 正在预处理输入... 模式: {mode_str}")
    
    final_input = resize_and_pad_image(raw_img, force_512)
    
    return final_input, final_input, None


def draw_red_dot_on_preview(clean_img, evt: gr.SelectData):
    """在预览图上绘制红点标记焦点"""
    if clean_img is None:
        return None, None
    
    img_copy = clean_img.copy()
    draw = ImageDraw.Draw(img_copy)
    x, y = evt.index
    r = 8
    draw.ellipse((x-r, y-r, x+r, y+r), outline="red", width=2)
    draw.line((x-r, y, x+r, y), fill="red", width=2)
    draw.line((x, y-r, x, y+r), fill="red", width=2)
    
    return img_copy, evt.index


def run_genfocus_pipeline(clean_input_processed, click_coords, K_value, cached_latents):
    """运行 Genfocus 流水线"""
    if clean_input_processed is None:
        raise gr.Error("请先完成步骤 1（上传图像）。")

    w, h = clean_input_processed.size
    print(f"🚀 开始 Genfocus 流水线... (尺寸: {w}x{h})")
    
    # ========== 阶段 1: 去模糊 ==========
    print("   ► 运行阶段 1: DeblurNet")
    switch_lora("deblur")
    
    condition_0_img = Image.new("RGB", (w, h), (0, 0, 0))
    cond0 = Condition(condition_0_img, "deblurring", [0, 32], 1.0)
    cond1 = Condition(clean_input_processed, "deblurring", [0, 0], 1.0)
    
    seed_everything(42)
    deblurred_img = generate(
        pipe_flux,
        height=h, width=w,
        prompt="a sharp photo with everything in focus",
        conditions=[cond0, cond1]
    ).images[0]
    
    if K_value == 0:
        print("✅ K=0，返回去模糊结果。")
        return deblurred_img, cached_latents

    # ========== 阶段 2: 散景生成 ==========
    print(f"   ► 运行阶段 2: BokehNet (K={K_value})")
    
    if click_coords is None:
        click_coords = [w // 2, h // 2]
        print("   ⚠️ 未选择焦点。默认使用中心点。")

    # 深度估计
    try:
        img_t = depth_transform(deblurred_img)
        if device == "cuda":
            img_t = img_t.to("cuda")
        with torch.no_grad():
            pred = depth_model.infer(img_t, f_px=None)
        depth_map = pred["depth"].cpu().numpy().squeeze()
        safe_depth = np.where(depth_map > 0.0, depth_map, np.finfo(np.float32).max)
        disp_orig = 1.0 / safe_depth
        disp = cv2.resize(disp_orig, (w, h), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"❌ 深度估计错误: {e}")
        return deblurred_img, cached_latents

    # 计算散焦图
    tx, ty = click_coords
    tx = min(max(int(tx), 0), w - 1)
    ty = min(max(int(ty), 0), h - 1)
    
    disp_focus = float(disp[ty, tx])
    dmf = disp - np.float32(disp_focus)
    defocus_abs = np.abs(K_value * dmf)
    MAX_COC = 100.0
    defocus_t = torch.from_numpy(defocus_abs).unsqueeze(0).float()
    cond_map = (defocus_t / MAX_COC).clamp(0, 1).repeat(3, 1, 1).unsqueeze(0)

    # 准备潜变量
    if cached_latents is None:
        print("      正在生成新的固定潜变量...")
        seed_everything(42)
        gen = torch.Generator(device=pipe_flux.device).manual_seed(1234)
        latents, _ = pipe_flux.prepare_latents(
            batch_size=1, num_channels_latents=16,
            height=h, width=w,
            dtype=pipe_flux.dtype, device=pipe_flux.device, generator=gen, latents=None
        )
        current_latents = latents
    else:
        print("      使用缓存的潜变量...")
        current_latents = cached_latents

    # 运行散景生成
    switch_lora("bokeh")
    cond_img = Condition(deblurred_img, "bokeh")
    cond_dmf = Condition(cond_map, "bokeh", [0, 0], 1.0, No_preprocess=True)
    
    seed_everything(42)
    gen = torch.Generator(device=pipe_flux.device).manual_seed(1234)
    
    with torch.no_grad():
        res = generate(
            pipe_flux,
            height=h, width=w,
            prompt="an excellent photo with a large aperture",
            conditions=[cond_img, cond_dmf],
            guidance_scale=1.0, kv_cache=False, generator=gen,
            latents=current_latents,
        )
    generated_bokeh = res.images[0]
    return generated_bokeh, current_latents


# ============== CSS 样式 ==============
css = """
#col-container { margin: 0 auto; max-width: 1400px; }
"""

# ============== 示例图像加载 ==============
example_dir = os.path.join(BASE_PATH, "example")

valid_examples = []
allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if os.path.exists(example_dir):
    files = sorted(os.listdir(example_dir))
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in allowed_extensions:
            full_path = os.path.join(example_dir, filename)
            valid_examples.append([full_path])
    print(f"✅ 从 '{example_dir}' 加载了 {len(valid_examples)} 个示例")
else:
    print(f"⚠️ 警告: 示例目录 '{example_dir}' 未找到。")


# ============== Gradio 界面 ==============
with gr.Blocks(css=css) as demo:
    # 状态变量
    clean_processed_state = gr.State(value=None)
    click_coords_state = gr.State(value=None)
    latents_state = gr.State(value=None)
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# 📷 Genfocus 流水线：交互式重对焦")
        
        gr.Markdown("""
        ### 📖 使用指南
        **生成式重对焦** 支持两种主要应用：
        
        * **全焦估计 (AIF)：** 设置 **K = 0**。模型将从模糊输入中恢复全焦图像。
          
        * **重对焦：** 
          1. 在**步骤 2** 的图像预览中**点击**您想要聚焦的主体。
          2. 增加 **K**（模糊强度）以根据场景深度生成逼真的散景效果。
        
        > ⚠️ **预处理说明：**
        > - **勾选（默认关闭）：** 将长边调整为 512 像素，并裁剪到最近的 16 的倍数。
        > - **不勾选：** 将高度和宽度上采样到最近的 16 的倍数（保留原始分辨率），并使用分块推理。
        """)
        
        with gr.Row():
            # 步骤 1: 上传图像
            with gr.Column(scale=1):
                gr.Markdown("### 步骤 1：上传图像")
                gr.Markdown("点击示例或上传您自己的图像。")
                
                input_raw = gr.Image(label="原始输入图像", type="pil")
                
                resize_512_check = gr.Checkbox(label="将长边调整为 512？", value=False)
                
                if valid_examples:
                    gr.Examples(examples=valid_examples, inputs=input_raw, label="示例（点击加载）", cache_examples=False)

            # 步骤 2: 设置焦点和 K 值
            with gr.Column(scale=1):
                gr.Markdown("### 步骤 2：设置焦点和 K 值")
                gr.Markdown("下图显示模型的实际输入。**点击图像**设置焦点。")
                
                focus_preview_img = gr.Image(label="模型输入（已处理）- 点击此处", type="pil", interactive=False)
                
                with gr.Row():
                    click_status = gr.Textbox(label="选定坐标", value="中心（默认）", interactive=False, scale=1)
                    k_slider = gr.Slider(minimum=0, maximum=50, value=20, step=1, label="模糊强度 (K)", scale=2)
                
                run_btn = gr.Button("✨ 运行 Genfocus", variant="primary", scale=1)

        # 结果显示
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 结果")
                output_img = gr.Image(label="最终输出", type="pil", interactive=False, elem_id="output_image")

        # 事件绑定
        input_raw.change(
            fn=preprocess_input_image,
            inputs=[input_raw, resize_512_check],
            outputs=[focus_preview_img, clean_processed_state, latents_state]
        )
        input_raw.upload(
            fn=preprocess_input_image,
            inputs=[input_raw, resize_512_check],
            outputs=[focus_preview_img, clean_processed_state, latents_state]
        )
        
        resize_512_check.change(
            fn=preprocess_input_image,
            inputs=[input_raw, resize_512_check],
            outputs=[focus_preview_img, clean_processed_state, latents_state]
        )

        focus_preview_img.select(
            fn=draw_red_dot_on_preview,
            inputs=[clean_processed_state],
            outputs=[focus_preview_img, click_coords_state]
        ).then(
            fn=lambda x: f"x={x[0]}, y={x[1]}",
            inputs=[click_coords_state],
            outputs=[click_status]
        )

        run_btn.click(
            fn=run_genfocus_pipeline,
            inputs=[clean_processed_state, click_coords_state, k_slider, latents_state],
            outputs=[output_img, latents_state]
        )


if __name__ == "__main__":
    allowed_dir = os.path.join(BASE_PATH, "example")
    allowed_paths = [allowed_dir]
    demo.launch(server_name="0.0.0.0", share=True, allowed_paths=allowed_paths)
