"""
Qwen-Image 图像生成与编辑 Gradio Web 应用
- Tab 1: Qwen-Image-2512 图像生成
- Tab 2: Qwen-Image-Edit-2511 + Multi-Angles LoRA 多视角编辑
"""

import gradio as gr
import torch
from PIL import Image
import os

# 模型路径
IMAGE_GEN_MODEL_PATH = "./checkpoints/Qwen-Image-2512"
IMAGE_EDIT_MODEL_PATH = "./checkpoints/Qwen-Image-Edit-2511"
MULTI_ANGLES_LORA_PATH = "./checkpoints/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/qwen-image-edit-2511-multiple-angles-lora.safetensors"

# 预设的宽高比选项
ASPECT_RATIOS = {
    "1:1 (1328×1328)": (1328, 1328),
    "16:9 (1664×928)": (1664, 928),
    "9:16 (928×1664)": (928, 1664),
    "4:3 (1472×1104)": (1472, 1104),
    "3:4 (1104×1472)": (1104, 1472),
    "3:2 (1584×1056)": (1584, 1056),
    "2:3 (1056×1584)": (1056, 1584),
}

# 默认负面提示词
DEFAULT_NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

# 多视角 LoRA 配置
AZIMUTHS = {
    "正面 (front view)": "front view",
    "右前45° (front-right quarter view)": "front-right quarter view",
    "右侧 (right side view)": "right side view",
    "右后135° (back-right quarter view)": "back-right quarter view",
    "背面 (back view)": "back view",
    "左后225° (back-left quarter view)": "back-left quarter view",
    "左侧 (left side view)": "left side view",
    "左前315° (front-left quarter view)": "front-left quarter view",
}

ELEVATIONS = {
    "仰视 -30° (low-angle shot)": "low-angle shot",
    "平视 0° (eye-level shot)": "eye-level shot",
    "俯视 30° (elevated shot)": "elevated shot",
    "高角度 60° (high-angle shot)": "high-angle shot",
}

DISTANCES = {
    "特写 (close-up)": "close-up",
    "中景 (medium shot)": "medium shot",
    "远景 (wide shot)": "wide shot",
}

# 示例提示词 - 按功能分类 [功能增强类别, 提示词]
EXAMPLE_PROMPTS = [
    # 人物写实
    [
        "🧑 人物写实 - 精细发丝和自然表情",
        "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm.",
    ],
    [
        "🧑 人物写实 - 面部细节和环境背景",
        "A Chinese female college student, around 20 years old, with a very short haircut that conveys a gentle, artistic vibe. Her hair naturally falls to partially cover her cheeks, projecting a tomboyish yet charming demeanor. She has cool-toned fair skin and delicate features, with a slightly shy yet subtly confident expression—her mouth crooked in a playful, youthful smirk. She wears an off-shoulder top, revealing one shoulder, with a well-proportioned figure. The image is framed as a close-up selfie: she dominates the foreground, while the background clearly shows her dormitory—a neatly made bed with white linens on the top bunk, a tidy study desk with organized stationery, and wooden cabinets and drawers. The photo is captured on a smartphone under soft, even ambient lighting, with natural tones, high clarity, and a bright, lively atmosphere full of youthful, everyday energy.",
    ],
    [
        "🧑 人物写实 - 精确姿态语义遵循",
        "An East Asian teenage boy, aged 15–18, with soft, fluffy black short hair and refined facial contours. His large, warm brown eyes sparkle with energy. His fair skin and sunny, open smile convey an approachable, friendly demeanor—no makeup or blemishes. He wears a blue-and-white summer uniform shirt, slightly unbuttoned, made of thin breathable fabric, with black headphones hanging around his neck. His hands are in his pockets, body leaning slightly forward in a relaxed pose, as if engaged in conversation. Behind him lies a summer school playground: lush green grass and a red rubber track in the foreground, blurred school buildings in the distance, a clear blue sky with fluffy white clouds. The bright, airy lighting evokes a joyful, carefree adolescent atmosphere.",
    ],
    [
        "🧑 人物写实 - 年龄特征（皱纹）渲染",
        "An elderly Chinese couple in their 70s in a clean, organized home kitchen. The woman has a kind face and a warm smile, wearing a patterned apron; the man stands behind her, also smiling, as they both gaze at a steaming pot of buns on the stove. The kitchen is bright and tidy, exuding warmth and harmony. The scene is captured with a wide-angle lens to fully show the subjects and their surroundings.",
    ],
    # 自然风景
    [
        "🌿 自然纹理 - 水流植被雾气渲染",
        "A turquoise river winds through a lush canyon. Thick moss and dense ferns blanket the rocky walls; multiple waterfalls cascade from above, enveloped in mist. At noon, sunlight filters through the dense canopy, dappling the river surface with shimmering light. The atmosphere is humid and fresh, pulsing with primal jungle vitality. No humans, text, or artificial traces present.",
    ],
    [
        "🌿 自然纹理 - 海浪与晨雾渲染",
        "At dawn, a thin mist veils the sea. An ancient stone lighthouse stands at the cliff's edge, its beacon faintly visible through the fog. Black rocks are pounded by waves, sending up bursts of white spray. The sky glows in soft blue-purple hues under cool, hazy light—evoking solitude and solemn grandeur.",
    ],
    # 动物毛发
    [
        "🐕 动物毛发 - 精细毛发纹理",
        "An ultra-realistic close-up of a golden retriever outdoors under soft daylight. Hair is exquisitely detailed: strands distinct, color transitioning naturally from warm gold to light cream, light glinting delicately at the tips; a gentle breeze adds subtle volume. Undercoat is soft and dense; guard hairs are long and well-defined, with visible layering. Eyes are moist, expressive; nose is slightly damp with fine specular highlights. Background is softly blurred to emphasize the dog's tangible texture and vivid expression.",
    ],
    [
        "🐕 动物毛发 - 粗糙野生动物纹理",
        "A male argali stands atop a barren, rocky mountainside. Its coarse, dense grey-brown coat covers a powerful, muscular body. Most striking are its massive, thick, outward-spiraling horns—a symbol of wild strength. Its gaze is alert and sharp. The background reveals steep alpine terrain: jagged peaks, sparse low vegetation, and abundant sunlight—conveying the harsh yet majestic wilderness and the animal's resilient vitality.",
    ],
    # 文字渲染
    [
        "📝 文字渲染 - PPT时间轴图文混排",
        '这是一张现代风格的科技感幻灯片，整体采用深蓝色渐变背景。标题是"Qwen-Image发展历程"。下方一条水平延伸的发光时间轴，轴线中间写着"生图路线"。由左侧淡蓝色渐变为右侧深紫色，并以精致的箭头收尾。时间轴上每个节点通过虚线连接至下方醒目的蓝色圆角矩形日期标签，标签内为清晰白色字体，从左向右依次写着："2025年5月6日 Qwen-Image 项目启动""2025年8月4日 Qwen-Image 开源发布""2025年12月31日 Qwen-Image-2512 开源发布"',
    ],
    [
        "📝 文字渲染 - 产品对比图混合渲染",
        '这是一张现代风格的科技感幻灯片，整体采用深蓝色渐变背景。顶部中央为白色无衬线粗体大字标题"Qwen-Image-2512重磅发布"。画面主体为横向对比图，视觉焦点集中于中间的升级对比区域。左侧为面部光滑没有任何细节的女性人像，质感差；右侧为高度写实的年轻女性肖像，皮肤呈现真实毛孔纹理与细微光影变化，发丝根根分明，眼眸透亮，表情自然，整体质感接近写实摄影。',
    ],
]

# 全局变量存储模型管道
gen_pipe = None
edit_pipe = None
lora_loaded = False


def load_gen_model():
    """加载 Qwen-Image-2512 图像生成模型"""
    global gen_pipe
    
    if gen_pipe is not None:
        return gen_pipe
    
    from diffusers import DiffusionPipeline
    
    print("正在加载 Qwen-Image-2512 图像生成模型...")
    
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        torch_dtype = torch.float32
        device = "cpu"
        print("CUDA 不可用，使用 CPU（生成速度会较慢）")
    
    gen_pipe = DiffusionPipeline.from_pretrained(
        IMAGE_GEN_MODEL_PATH, 
        torch_dtype=torch_dtype
    ).to(device)
    
    print("Qwen-Image-2512 模型加载完成！")
    return gen_pipe


def load_edit_model():
    """加载 Qwen-Image-Edit-2511 图像编辑模型"""
    global edit_pipe, lora_loaded
    
    if edit_pipe is not None:
        return edit_pipe
    
    from diffusers import QwenImageEditPlusPipeline
    
    print("正在加载 Qwen-Image-Edit-2511 图像编辑模型...")
    
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    
    edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
        IMAGE_EDIT_MODEL_PATH, 
        torch_dtype=torch_dtype
    )
    edit_pipe.to(device)
    edit_pipe.set_progress_bar_config(disable=None)
    
    # 加载 Multi-Angles LoRA
    print("正在加载 Multi-Angles LoRA...")
    edit_pipe.load_lora_weights(MULTI_ANGLES_LORA_PATH)
    lora_loaded = True
    
    print("Qwen-Image-Edit-2511 + Multi-Angles LoRA 加载完成！")
    return edit_pipe


def generate_image(
    prompt: str,
    negative_prompt: str,
    aspect_ratio: str,
    num_inference_steps: int,
    true_cfg_scale: float,
    seed: int,
    progress=gr.Progress()
):
    """生成图像"""
    global gen_pipe
    
    if gen_pipe is None:
        gen_pipe = load_gen_model()
    
    if not prompt.strip():
        return None, "错误：请输入提示词。"
    
    try:
        width, height = ASPECT_RATIOS[aspect_ratio]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)
        
        progress(0, desc="开始生成图像...")
        
        result = gen_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator
        )
        
        image = result.images[0]
        info = f"✅ 生成成功！\n尺寸: {width}×{height}\n步数: {num_inference_steps}\nCFG: {true_cfg_scale}\n种子: {seed}"
        
        return image, info
        
    except Exception as e:
        error_msg = f"❌ 生成失败：{str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def edit_image_multi_angle(
    input_image,
    azimuth: str,
    elevation: str,
    distance: str,
    num_inference_steps: int,
    true_cfg_scale: float,
    seed: int,
    progress=gr.Progress()
):
    """使用多视角 LoRA 编辑图像"""
    global edit_pipe
    
    if edit_pipe is None:
        edit_pipe = load_edit_model()
    
    if input_image is None:
        return None, "错误：请先上传或从生成 Tab 传入图像。"
    
    try:
        # 构建 LoRA 提示词
        azimuth_text = AZIMUTHS[azimuth]
        elevation_text = ELEVATIONS[elevation]
        distance_text = DISTANCES[distance]
        
        prompt = f"<sks> {azimuth_text} {elevation_text} {distance_text}"
        
        progress(0, desc=f"正在生成多视角图像: {prompt}")
        
        # 确保输入图像是 PIL Image
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)
        
        generator = torch.manual_seed(seed)
        
        with torch.inference_mode():
            result = edit_pipe(
                image=[input_image],
                prompt=prompt,
                generator=generator,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=" ",
                num_inference_steps=num_inference_steps,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            )
        
        output_image = result.images[0]
        info = f"✅ 多视角编辑成功！\n提示词: {prompt}\n步数: {num_inference_steps}\nCFG: {true_cfg_scale}\n种子: {seed}"
        
        return output_image, info
        
    except Exception as e:
        error_msg = f"❌ 编辑失败：{str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def send_to_edit_tab(generated_image):
    """将生成的图像发送到编辑 Tab"""
    if generated_image is None:
        gr.Warning("没有生成的图像可以发送")
        return None
    return generated_image


def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(
        title="Qwen-Image 图像生成与多视角编辑"
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1>🎨 Qwen-Image 图像生成与多视角编辑</h1>
            <p>Tab 1: 使用 Qwen-Image-2512 生成图像 | Tab 2: 使用 Qwen-Image-Edit-2511 + Multi-Angles LoRA 多视角编辑</p>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            # ==================== Tab 1: 图像生成 ====================
            with gr.TabItem("🖼️ 图像生成", id=0):
                with gr.Row():
                    # 左侧输入区域
                    with gr.Column(scale=1):
                        gen_prompt = gr.Textbox(
                            label="✏️ 提示词 (Prompt)",
                            placeholder="请输入图像描述...",
                            lines=5,
                            max_lines=10
                        )
                        
                        gen_negative_prompt = gr.Textbox(
                            label="🚫 负面提示词 (Negative Prompt)",
                            value=DEFAULT_NEGATIVE_PROMPT,
                            lines=3,
                            max_lines=5
                        )
                        
                        with gr.Row():
                            gen_aspect_ratio = gr.Dropdown(
                                label="📐 宽高比",
                                choices=list(ASPECT_RATIOS.keys()),
                                value="16:9 (1664×928)"
                            )
                            
                            gen_seed = gr.Number(
                                label="🎲 随机种子",
                                value=42,
                                precision=0
                            )
                        
                        with gr.Row():
                            gen_steps = gr.Slider(
                                label="🔄 推理步数",
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=1
                            )
                            
                            gen_cfg = gr.Slider(
                                label="🎯 CFG 强度",
                                minimum=1.0,
                                maximum=10.0,
                                value=4.0,
                                step=0.5
                            )
                        
                        with gr.Row():
                            gen_btn = gr.Button(
                                "🚀 生成图像",
                                variant="primary",
                                size="lg"
                            )
                            send_to_edit_btn = gr.Button(
                                "📤 发送到多视角编辑",
                                variant="secondary",
                                size="lg"
                            )
                    
                    # 右侧输出区域
                    with gr.Column(scale=1):
                        gen_output_image = gr.Image(
                            label="🖼️ 生成结果",
                            type="pil"
                        )
                        
                        gen_output_info = gr.Textbox(
                            label="📊 生成信息",
                            lines=5,
                            interactive=False
                        )
                
                # 示例区域
                gr.HTML("""
                <div style="margin-top: 2rem;">
                    <h3>📚 示例提示词</h3>
                    <p style="color: #666; font-size: 0.9em;">点击下方示例快速体验不同功能场景</p>
                </div>
                """)
                
                feature_category = gr.Textbox(visible=False)
                
                gr.Examples(
                    examples=EXAMPLE_PROMPTS,
                    inputs=[feature_category, gen_prompt],
                    label="",
                    examples_per_page=10
                )
            
            # ==================== Tab 2: 多视角编辑 ====================
            with gr.TabItem("🔄 多视角编辑", id=1):
                gr.HTML("""
                <div style="margin-bottom: 1rem; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
                    <h3 style="margin: 0;">📷 Multi-Angles LoRA 多视角相机控制</h3>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9em;">
                        支持 96 种精确相机位置：8 个水平角度 × 4 个垂直角度 × 3 个距离
                    </p>
                </div>
                """)
                
                with gr.Row():
                    # 左侧输入区域
                    with gr.Column(scale=1):
                        edit_input_image = gr.Image(
                            label="📥 输入图像（可从生成Tab发送或直接上传）",
                            type="pil"
                        )
                        
                        gr.HTML("<h4>📷 相机位置设置</h4>")
                        
                        with gr.Row():
                            edit_azimuth = gr.Dropdown(
                                label="🔄 水平角度 (Azimuth)",
                                choices=list(AZIMUTHS.keys()),
                                value="正面 (front view)"
                            )
                        
                        with gr.Row():
                            edit_elevation = gr.Dropdown(
                                label="📐 垂直角度 (Elevation)",
                                choices=list(ELEVATIONS.keys()),
                                value="平视 0° (eye-level shot)"
                            )
                        
                        with gr.Row():
                            edit_distance = gr.Dropdown(
                                label="📏 拍摄距离 (Distance)",
                                choices=list(DISTANCES.keys()),
                                value="中景 (medium shot)"
                            )
                        
                        with gr.Row():
                            edit_steps = gr.Slider(
                                label="🔄 推理步数",
                                minimum=10,
                                maximum=80,
                                value=40,
                                step=1
                            )
                            
                            edit_cfg = gr.Slider(
                                label="🎯 CFG 强度",
                                minimum=1.0,
                                maximum=10.0,
                                value=4.0,
                                step=0.5
                            )
                        
                        edit_seed = gr.Number(
                            label="🎲 随机种子",
                            value=0,
                            precision=0
                        )
                        
                        edit_btn = gr.Button(
                            "🔄 生成多视角图像",
                            variant="primary",
                            size="lg"
                        )
                    
                    # 右侧输出区域
                    with gr.Column(scale=1):
                        edit_output_image = gr.Image(
                            label="🖼️ 多视角编辑结果",
                            type="pil"
                        )
                        
                        edit_output_info = gr.Textbox(
                            label="📊 编辑信息",
                            lines=5,
                            interactive=False
                        )
                
                # 相机位置参考
                with gr.Accordion("📖 相机位置参考", open=False):
                    gr.Markdown("""
### 🔄 水平角度 (Azimuth) - 8个方向
```
                         0° 
                    (正面 front)
                         │
         315°            │            45°
      (左前)             │          (右前)
              ╲          │          ╱
               ╲         │         ╱
                ╲        │        ╱
   270° ─────────────── ● ─────────────── 90°
   (左侧)             物体            (右侧)
                ╱        │        ╲
               ╱         │         ╲
              ╱          │          ╲
         225°            │            135°
       (左后)            │          (右后)
                         │
                        180°
                     (背面 back)
```

### 📐 垂直角度 (Elevation) - 4个高度
| 角度 | 描述 | 说明 |
|------|------|------|
| -30° | 仰视 (low-angle) | 相机在下方，向上看 |
| 0° | 平视 (eye-level) | 相机与物体同高 |
| 30° | 俯视 (elevated) | 相机略高于物体 |
| 60° | 高角度 (high-angle) | 相机在高处向下看 |

### 📏 拍摄距离 (Distance) - 3种距离
| 类型 | 描述 | 用途 |
|------|------|------|
| ×0.6 | 特写 (close-up) | 细节、纹理 |
| ×1.0 | 中景 (medium shot) | 平衡、标准 |
| ×1.8 | 远景 (wide shot) | 环境、全景 |
                    """)
        
        # ==================== 事件绑定 ====================
        
        # 生成按钮
        gen_btn.click(
            fn=generate_image,
            inputs=[gen_prompt, gen_negative_prompt, gen_aspect_ratio, gen_steps, gen_cfg, gen_seed],
            outputs=[gen_output_image, gen_output_info]
        )
        
        # 发送到编辑 Tab
        send_to_edit_btn.click(
            fn=send_to_edit_tab,
            inputs=[gen_output_image],
            outputs=[edit_input_image]
        ).then(
            fn=lambda: gr.Tabs(selected=1),
            outputs=[tabs]
        )
        
        # 多视角编辑按钮
        edit_btn.click(
            fn=edit_image_multi_angle,
            inputs=[edit_input_image, edit_azimuth, edit_elevation, edit_distance, edit_steps, edit_cfg, edit_seed],
            outputs=[edit_output_image, edit_output_info]
        )
    
    return demo


if __name__ == "__main__":
    # 预加载模型（可选，也可以在首次使用时加载）
    print("=" * 50)
    print("Qwen-Image 图像生成与多视角编辑")
    print("=" * 50)
    
    # 可以选择预加载模型
    # load_gen_model()
    # load_edit_model()
    
    # 创建并启动界面
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
