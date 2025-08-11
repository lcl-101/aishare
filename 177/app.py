import gradio as gr
import numpy as np
import random
import os
import time
import torch
import math
import torch.nn as nn
from PIL import Image
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.models import QwenImageTransformer2DModel
from safetensors.torch import safe_open

# 配置参数
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1440

# 全局变量存储管道
base_pipe = None
lora_pipe = None

# 示例提示词
EXAMPLE_PROMPTS = [
    'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".',
    "一个咖啡店门口有一个黑板，上面写着通义千问咖啡，2美元一杯，旁边有个霓虹灯，写着阿里巴巴，旁边有个海报，海报上面是一个中国美女，海报下方写着qwen newbee",
    "一幅精致细腻的工笔画，画面中心是一株蓬勃生长的红色牡丹，花朵繁茂，既有盛开的硕大花瓣，也有含苞待放的花蕾，层次丰富，色彩艳丽而不失典雅。牡丹枝叶舒展，叶片浓绿饱满，脉络清晰可见，与红花相映成趣。一只蓝紫色蝴蝶仿佛被画中花朵吸引，停驻在画面中央的一朵盛开牡丹上，流连忘返，蝶翼轻展，细节逼真，仿佛随时会随风飞舞。整幅画作笔触工整严谨，色彩浓郁鲜明，展现出中国传统工笔画的精妙与神韵，画面充满生机与灵动之感。",
    "A capybara wearing a suit holding a sign that reads Hello World",
    '一个会议室，墙上写着"3.14159265-358979-32384626-4338327950"，一个小陀螺在桌上转动',
    'A young girl wearing school uniform stands in a classroom, writing on a chalkboard. The text "Introducing Qwen-Image, a foundational image generation model that excels in complex text rendering and precise image editing" appears in neat white chalk at the center of the blackboard. Soft natural light filters through windows, casting gentle shadows. The scene is rendered in a realistic photography style with fine details, shallow depth of field, and warm tones. The girl\'s focused expression and chalk dust in the air add dynamism. Background elements include desks and educational posters, subtly blurred to emphasize the central action. Ultra-detailed 32K resolution, DSLR-quality, soft bokeh effect, documentary-style composition',
    "Realistic still life photography style: A single, fresh apple resting on a clean, soft-textured surface. The apple is slightly off-center, softly backlit to highlight its natural gloss and subtle color gradients—deep crimson red blending into light golden hues. Fine details such as small blemishes, dew drops, and a few light highlights enhance its lifelike appearance. A shallow depth of field gently blurs the neutral background, drawing full attention to the apple. Hyper-detailed 8K resolution, studio lighting, photorealistic render, emphasizing texture and form."
]

def build_lora_names(key, lora_down_key, lora_up_key, is_native_weight):
    """构建LoRA权重名称"""
    base = "diffusion_model." if is_native_weight else ""
    lora_down = base + key.replace(".weight", lora_down_key)
    lora_up = base + key.replace(".weight", lora_up_key)
    lora_alpha = base + key.replace(".weight", ".alpha")
    return lora_down, lora_up, lora_alpha

def load_and_merge_lora_weight(
    model: nn.Module,
    lora_state_dict: dict,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
):
    """加载并合并LoRA权重"""
    is_native_weight = any("diffusion_model." in key for key in lora_state_dict)
    for key, value in model.named_parameters():
        lora_down_name, lora_up_name, lora_alpha_name = build_lora_names(
            key, lora_down_key, lora_up_key, is_native_weight
        )
        if lora_down_name in lora_state_dict:
            lora_down = lora_state_dict[lora_down_name]
            lora_up = lora_state_dict[lora_up_name]
            lora_alpha = float(lora_state_dict[lora_alpha_name])
            rank = lora_down.shape[0]
            scaling_factor = lora_alpha / rank
            assert lora_up.dtype == torch.float32
            assert lora_down.dtype == torch.float32
            delta_W = scaling_factor * torch.matmul(lora_up, lora_down)
            value.data = (value.data + delta_W).type_as(value.data)
    return model

def load_and_merge_lora_weight_from_safetensors(
    model: nn.Module,
    lora_weight_path: str,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
):
    """从SafeTensors文件加载LoRA权重"""
    lora_state_dict = {}
    with safe_open(lora_weight_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_state_dict[key] = f.get_tensor(key)
    model = load_and_merge_lora_weight(
        model, lora_state_dict, lora_down_key, lora_up_key
    )
    return model

def load_base_model():
    """加载基础模型"""
    global base_pipe
    if base_pipe is None:
        model_name = "checkpoints"
        
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
        else:
            torch_dtype = torch.float32
            device = "cpu"
        
        base_pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        base_pipe = base_pipe.to(device)
        print(f"基础模型已加载到 {device}")
    return base_pipe

def load_lora_model():
    """加载LoRA模型"""
    global lora_pipe
    if lora_pipe is None:
        model_name = "checkpoints"
        lora_path = "checkpoints/lora/Qwen-Image-Lightning-8steps-V1.0.safetensors"
        
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
        else:
            torch_dtype = torch.float32
            device = "cpu"
        
        if os.path.exists(lora_path):
            # 加载LoRA模型
            model = QwenImageTransformer2DModel.from_pretrained(
                model_name, subfolder="transformer", torch_dtype=torch_dtype
            )
            model = load_and_merge_lora_weight_from_safetensors(model, lora_path)
            
            # 设置调度器配置
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
            lora_pipe = DiffusionPipeline.from_pretrained(
                model_name, transformer=model, scheduler=scheduler, torch_dtype=torch_dtype
            )
            lora_pipe = lora_pipe.to(device)
            print(f"LoRA模型已加载到 {device}")
        else:
            print(f"LoRA文件 {lora_path} 不存在，将使用基础模型")
            lora_pipe = load_base_model()
    return lora_pipe

def detect_language(text):
    """检测文本语言"""
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    return "zh" if chinese_chars > 0 else "en"

def generate_single_image(pipe, prompt, width, height, num_steps, cfg_scale, seed):
    """生成单张图片"""
    # 检测语言并添加魔法词
    lang = detect_language(prompt)
    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition. ",
        "zh": "超清，4K，电影级构图。"
    }
    
    full_prompt = positive_magic[lang] + prompt
    negative_prompt = " "  # 推荐的空负面提示词
    
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        true_cfg_scale=cfg_scale,
        generator=generator
    ).images[0]
    
    return image

def generate_comparison(prompt, width, height, seed, progress=gr.Progress()):
    """生成对比图片"""
    if not prompt:
        return None, None, "请输入提示词"
    
    if seed == -1:
        seed = random.randint(0, MAX_SEED)
    
    progress(0.1, desc="正在加载模型...")
    
    # 加载模型
    base_model = load_base_model()
    lora_model = load_lora_model()
    
    try:
        # 生成基础模型图片
        progress(0.3, desc="正在生成基础模型图片...")
        base_start_time = time.time()
        
        base_image = generate_single_image(
            base_model, prompt, width, height, 
            num_steps=50, cfg_scale=4.0, seed=seed
        )
        
        base_end_time = time.time()
        base_generation_time = base_end_time - base_start_time
        
        # 生成LoRA模型图片
        progress(0.7, desc="正在生成LoRA模型图片...")
        lora_start_time = time.time()
        
        lora_image = generate_single_image(
            lora_model, prompt, width, height,
            num_steps=8, cfg_scale=1.0, seed=seed
        )
        
        lora_end_time = time.time()
        lora_generation_time = lora_end_time - lora_start_time
        
        progress(1.0, desc="生成完成!")
        
        # 保存图片
        timestamp = int(time.time())
        base_filename = f"base_{timestamp}_{seed}.png"
        lora_filename = f"lora_{timestamp}_{seed}.png"
        base_image.save(base_filename)
        lora_image.save(lora_filename)
        
        # 计算速度提升
        speed_improvement = base_generation_time / lora_generation_time if lora_generation_time > 0 else 0
        
        info = f"""生成完成！
种子值: {seed}
尺寸: {width}x{height}

📊 性能对比:
基础模型 (50步, CFG 4.0): {base_generation_time:.2f}秒
LoRA模型 (8步, CFG 1.0): {lora_generation_time:.2f}秒
速度提升: {speed_improvement:.1f}x

💾 保存文件:
基础模型: {base_filename}
LoRA模型: {lora_filename}
        """
        
        return base_image, lora_image, info
        
    except Exception as e:
        return None, None, f"生成失败: {str(e)}"

def randomize_seed():
    """随机种子生成"""
    return random.randint(0, MAX_SEED)

def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="Qwen-Image 对比生成器", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #2E86AB; font-size: 2.5em;">🎨 Qwen-Image 对比生成器</h1>
            <p style="font-size: 1.2em; color: #555;">基础模型 vs LoRA Lightning 模型对比</p>
            <p style="color: #777;">⏱️ 逐个生成图片，精确测量生成时间</p>
            <p style="color: #888; font-size: 0.9em;">左侧：基础模型 (50步, CFG 4.0) | 右侧：LoRA模型 (8步, CFG 1.0)</p>
            <p style="color: #666; font-size: 0.8em;">🚀 LoRA Lightning模型能显著提升生成速度</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="提示词 (Prompt)",
                    placeholder="描述你想要生成的图片...",
                    lines=4,
                    value=""
                )
                
                # 示例提示词
                with gr.Row():
                    gr.Markdown("**示例提示词:**")
                example_dropdown = gr.Dropdown(
                    choices=[prompt[:100] + "..." if len(prompt) > 100 else prompt for prompt in EXAMPLE_PROMPTS],
                    label="选择示例",
                    interactive=True
                )
                
                with gr.Row():
                    seed_input = gr.Number(
                        label="种子值 (Seed)",
                        value=42,
                        minimum=-1,
                        maximum=MAX_SEED
                    )
                    random_seed_btn = gr.Button("🎲 随机", size="sm")
                
                with gr.Row():
                    width_input = gr.Slider(
                        label="宽度",
                        minimum=512,
                        maximum=MAX_IMAGE_SIZE,
                        value=1328,
                        step=64
                    )
                    height_input = gr.Slider(
                        label="高度",
                        minimum=512,
                        maximum=MAX_IMAGE_SIZE,
                        value=1328,
                        step=64
                    )
                
                # 快速尺寸预设
                with gr.Row():
                    gr.Markdown("**快速尺寸预设:**")
                with gr.Row():
                    square_btn = gr.Button("⬜ 1:1", size="sm")
                    landscape_btn = gr.Button("📺 16:9", size="sm")
                    portrait_btn = gr.Button("📱 9:16", size="sm")
                    photo_btn = gr.Button("📷 4:3", size="sm")
                
                generate_btn = gr.Button("🎨 生成对比图片", variant="primary", size="lg")
                
                # 显示生成信息和性能对比
                output_info = gr.Textbox(
                    label="⏱️ 生成信息与性能对比", 
                    interactive=False, 
                    lines=10,
                    placeholder="生成信息将在这里显示，包括详细的时间对比..."
                )
        
        # 对比结果显示
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 基础模型 (50步, CFG 4.0)")
                base_output = gr.Image(label="基础模型结果", type="pil")
            
            with gr.Column(scale=1):
                gr.Markdown("### LoRA Lightning模型 (8步, CFG 1.0)")
                lora_output = gr.Image(label="LoRA模型结果", type="pil")
        
        # 事件绑定
        def load_example(choice):
            if choice:
                # 找到完整的提示词
                for prompt in EXAMPLE_PROMPTS:
                    if choice.startswith(prompt[:100]):
                        return prompt
            return ""
        
        example_dropdown.change(
            fn=load_example,
            inputs=[example_dropdown],
            outputs=[prompt_input]
        )
        
        random_seed_btn.click(fn=randomize_seed, outputs=seed_input)
        
        # 尺寸预设按钮
        square_btn.click(fn=lambda: (1328, 1328), outputs=[width_input, height_input])
        landscape_btn.click(fn=lambda: (1408, 928), outputs=[width_input, height_input])  # 修正为1408以符合限制
        portrait_btn.click(fn=lambda: (928, 1408), outputs=[width_input, height_input])  # 修正为1408以符合限制
        photo_btn.click(fn=lambda: (1408, 1056), outputs=[width_input, height_input])  # 修正尺寸
        
        generate_btn.click(
            fn=generate_comparison,
            inputs=[prompt_input, width_input, height_input, seed_input],
            outputs=[base_output, lora_output, output_info]
        )
        
        # 添加CSS样式
        demo.css = """
        .gradio-container {
            max-width: 1600px !important;
        }
        .gr-button {
            transition: all 0.3s ease;
        }
        .gr-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .gr-tab {
            font-weight: 600;
        }
        """
    
    return demo

if __name__ == "__main__":
    # 检查LoRA文件是否存在
    lora_path = "checkpoints/lora/Qwen-Image-Lightning-8steps-V1.0.safetensors"
    if not os.path.exists(lora_path):
        print(f"警告: LoRA文件 {lora_path} 不存在")
        print("请确保已下载LoRA权重文件，或者程序将使用基础模型进行对比")
    
    # 预加载模型
    print("正在预加载Qwen-Image模型...")
    load_base_model()
    load_lora_model()
    print("模型加载完成!")
    
    # 启动界面
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
