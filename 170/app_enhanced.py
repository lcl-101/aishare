# =============================================================================
# HYPIR 增强版 Gradio WebUI
# =============================================================================
# 
# 一个功能完整的 HYPIR 图像增强 Web 界面
# 
# 功能特性:
# - 内置配置（无需外部配置文件）
# - 示例图片和提示词加载
# - 增强的用户界面和参数控制
# - 本地模型支持
# 
# 使用方法:
#   python app_enhanced.py
# 
# 环境要求:
# - 模型文件在 checkpoints/ 目录中:
#   - checkpoints/stable-diffusion-2-1-base/
#   - checkpoints/HYPIR/HYPIR_sd2.pth
# - 示例文件在 examples/ 目录中（可选）:
#   - examples/lq/ (低质量图片)  
#   - examples/prompt/ (文本提示词)
# 
# =============================================================================

import random
import os
from pathlib import Path

import gradio as gr
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from dotenv import load_dotenv
from PIL import Image

from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import GPTCaptioner

print("🚀 HYPIR Enhanced WebUI - Starting...")
print("=" * 50)


# =============================================================================
# 配置 - 所有设置都在这里
# =============================================================================

# 模型配置
CONFIG = {
    "base_model_type": "sd2",
    "base_model_path": "checkpoints/stable-diffusion-2-1-base",
    "weight_path": "checkpoints/HYPIR/HYPIR_sd2.pth",
    "lora_modules": ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", 
                     "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"],
    "lora_rank": 256,
    "model_t": 200,
    "coeff_t": 200,
    "device": "cuda",
}

# WebUI 配置
WEBUI_CONFIG = {
    "port": 7860,
    "server_name": "0.0.0.0",  # 使用 "127.0.0.1" 只允许本地访问
    "share": False,
    "max_size": None,  # 设置为 (宽度, 高度) 元组来限制输出尺寸，例如 (2048, 2048)
    "enable_gpt_caption": False,  # 如果配置了 GPT API，设置为 True
}

# 默认参数
DEFAULT_PARAMS = {
    "upscale": 4,
    "patch_size": 512,
    "stride": 256,
    "seed": -1,
}

# =============================================================================
# 初始化
# =============================================================================

load_dotenv()
error_image = Image.open(os.path.join("assets", "gradio_error_img.png"))

# 检查模型文件是否存在
def check_model_files():
    """检查所有必需的模型文件是否存在"""
    base_model_path = Path(CONFIG["base_model_path"])
    weight_path = Path(CONFIG["weight_path"])
    
    print("📂 检查模型文件...")
    
    if not base_model_path.exists():
        print(f"❌ 基础模型未找到: {base_model_path}")
        print("请下载 Stable Diffusion 2.1 基础模型到 checkpoints/stable-diffusion-2-1-base/")
        raise FileNotFoundError(f"基础模型未找到: {base_model_path}")
    
    if not weight_path.exists():
        print(f"❌ HYPIR 权重未找到: {weight_path}")
        print("请下载 HYPIR 模型权重到 checkpoints/HYPIR/HYPIR_sd2.pth")
        raise FileNotFoundError(f"HYPIR 权重未找到: {weight_path}")
    
    print(f"✅ 基础模型已找到: {base_model_path}")
    print(f"✅ HYPIR 权重已找到: {weight_path}")
    
    # 检查示例
    examples_dir = Path("examples")
    if (examples_dir / "lq").exists() and (examples_dir / "prompt").exists():
        example_count = len(list((examples_dir / "lq").glob("*.png")))
        print(f"📸 找到 {example_count} 个示例图片")
    else:
        print("⚠️  示例目录未找到 - 示例画廊将为空")

print("🔍 运行预检查...")
check_model_files()

# 设置最大尺寸限制
max_size = WEBUI_CONFIG["max_size"]
if max_size is not None:
    print(f"最大尺寸设置为 {max_size}，最大像素数: {max_size[0] * max_size[1]}")

# 如果启用则设置 GPT 字幕生成器
captioner = None
if WEBUI_CONFIG["enable_gpt_caption"]:
    if (
        "GPT_API_KEY" not in os.environ
        or "GPT_BASE_URL" not in os.environ
        or "GPT_MODEL" not in os.environ
    ):
        print("警告: GPT 字幕功能已启用但环境变量未设置。")
        print("请在 .env 文件中设置 GPT_API_KEY, GPT_BASE_URL 和 GPT_MODEL。")
        WEBUI_CONFIG["enable_gpt_caption"] = False
    else:
        captioner = GPTCaptioner(
            api_key=os.getenv("GPT_API_KEY"),
            base_url=os.getenv("GPT_BASE_URL"),
            model=os.getenv("GPT_MODEL"),
        )

to_tensor = transforms.ToTensor()

# 初始化模型
print("🚀 初始化 HYPIR 模型...")
print(f"📱 使用设备: {CONFIG['device']}")
print(f"🧠 模型配置: {CONFIG['base_model_type']}")
print(f"📂 基础模型: {CONFIG['base_model_path']}")
print(f"⚡ LoRA 秩: {CONFIG['lora_rank']}")

model = SD2Enhancer(
    base_model_path=CONFIG["base_model_path"],
    weight_path=CONFIG["weight_path"],
    lora_modules=CONFIG["lora_modules"],
    lora_rank=CONFIG["lora_rank"],
    model_t=CONFIG["model_t"],
    coeff_t=CONFIG["coeff_t"],
    device=CONFIG["device"],
)

print("📥 加载模型...")
model.init_models()
print("✅ 模型加载成功!")


def load_examples():
    """加载示例图片和对应的提示词"""
    examples_dir = Path("examples")
    lq_dir = examples_dir / "lq"
    prompt_dir = examples_dir / "prompt"
    
    examples = []
    if lq_dir.exists() and prompt_dir.exists():
        for img_file in sorted(lq_dir.glob("*.png")):
            prompt_file = prompt_dir / f"{img_file.stem}.txt"
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                examples.append({
                    "name": img_file.stem,
                    "image_path": str(img_file),
                    "prompt": prompt
                })
    return examples


def get_example_list():
    """获取示例名称列表用于下拉菜单"""
    examples = load_examples()
    return [ex["name"] for ex in examples]


def load_example(example_name):
    """加载选中的示例图片和提示词"""
    if not example_name:
        return None, ""
    
    examples = load_examples()
    for ex in examples:
        if ex["name"] == example_name:
            try:
                image = Image.open(ex["image_path"])
                return image, ex["prompt"]
            except Exception as e:
                print(f"加载示例 {example_name} 时出错: {e}")
                return None, ""
    return None, ""


def process(
    image,
    prompt,
    upscale,
    patch_size,
    stride,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    set_seed(seed)
    image = image.convert("RGB")
    # 检查图片尺寸
    if max_size is not None:
        out_w, out_h = tuple(int(x * upscale) for x in image.size)
        if out_w * out_h > max_size[0] * max_size[1]:
            return error_image, (
                "失败: 请求的分辨率超过最大像素限制。"
                f"您请求的分辨率是 ({out_h}, {out_w})。"
                f"最大允许的像素数是 {max_size[0]} x {max_size[1]} "
                f"= {max_size[0] * max_size[1]} :("
            )
    if prompt == "auto":
        if WEBUI_CONFIG["enable_gpt_caption"] and captioner is not None:
            prompt = captioner(image)
        else:
            return error_image, "失败: 此 Gradio 未启用 GPT 字幕支持 :("

    image_tensor = to_tensor(image).unsqueeze(0)
    try:
        pil_image = model.enhance(
            lq=image_tensor,
            prompt=prompt,
            upscale=upscale,
            patch_size=patch_size,
            stride=stride,
            return_type="pil",
        )[0]
    except Exception as e:
        return error_image, f"失败: {e} :("

    return pil_image, f"成功! :)\n使用的提示词: {prompt}\n使用的种子: {seed}"


MARKDOWN = """
## HYPIR: 利用扩散产生的分数先验进行图像恢复

[GitHub](https://github.com/XPixelGroup/HYPIR) | [论文](TODO) | [项目页面](TODO)

如果 HYPIR 对您有帮助，请帮助为 GitHub 仓库点星。谢谢！

### 使用说明:
1. **上传您自己的图片** 或 **从下面的示例中选择**
2. **输入描述图片的提示词** 或使用 "auto" 进行 GPT 生成的字幕
3. **根据需要调整参数**（放大倍数、补丁大小、步长）
4. **点击运行** 来增强您的图片
"""

# 自定义 CSS 样式
css = """
.example-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 10px;
    padding: 10px;
}
.example-item {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 8px;
    text-align: center;
}
.example-item img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
}
"""

block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown(MARKDOWN)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 输入")
            
            # 示例选择
            with gr.Group():
                gr.Markdown("**加载示例:**")
                example_dropdown = gr.Dropdown(
                    choices=get_example_list(),
                    label="选择示例",
                    value=None,
                    interactive=True
                )
                load_example_btn = gr.Button("加载选中的示例", variant="secondary")
            
            # 图片输入
            image = gr.Image(type="pil", label="输入图片")
            
            # 提示词输入
            prompt = gr.Textbox(
                label=(
                    "提示词 (输入 'auto' 使用 GPT 生成的字幕)"
                    if WEBUI_CONFIG["enable_gpt_caption"] else "提示词"
                ),
                placeholder="输入图片的描述...",
                lines=3
            )
            
            # 参数设置
            with gr.Group():
                gr.Markdown("**参数设置:**")
                upscale = gr.Slider(
                    minimum=1, 
                    maximum=8, 
                    value=DEFAULT_PARAMS["upscale"], 
                    label="放大倍数", 
                    step=1,
                    info="图片放大的倍数"
                )
                patch_size = gr.Slider(
                    minimum=512, 
                    maximum=1024, 
                    value=DEFAULT_PARAMS["patch_size"], 
                    label="补丁大小", 
                    step=128,
                    info="处理补丁的大小"
                )
                stride = gr.Slider(
                    minimum=256, 
                    maximum=1024, 
                    value=DEFAULT_PARAMS["stride"], 
                    label="补丁步长", 
                    step=128,
                    info="补丁之间的步长"
                )
                seed = gr.Number(
                    label="随机种子", 
                    value=DEFAULT_PARAMS["seed"],
                    info="随机种子 (-1 为随机)"
                )
            
            run = gr.Button(value="🚀 增强图片", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 输出")
            result = gr.Image(type="pil", format="png", label="增强后的图片")
            status = gr.Textbox(label="状态", interactive=False, lines=3)
    
    # 示例画廊
    with gr.Row():
        gr.Markdown("### 📸 示例画廊")
    
    # 创建示例画廊
    examples = load_examples()
    if examples:
        with gr.Row():
            for i in range(0, len(examples), 3):  # 每行显示3个示例
                with gr.Column():
                    for j in range(3):
                        if i + j < len(examples):
                            ex = examples[i + j]
                            with gr.Group():
                                gr.Image(
                                    value=ex["image_path"],
                                    label=ex["name"],
                                    show_label=True,
                                    interactive=False,
                                    height=150
                                )
                                gr.Textbox(
                                    value=ex["prompt"][:100] + "..." if len(ex["prompt"]) > 100 else ex["prompt"],
                                    label="提示词预览",
                                    interactive=False,
                                    lines=2,
                                    max_lines=2
                                )
    
    # 事件处理器
    def load_example_handler(example_name):
        return load_example(example_name)
    
    load_example_btn.click(
        fn=load_example_handler,
        inputs=[example_dropdown],
        outputs=[image, prompt]
    )
    
    run.click(
        fn=process,
        inputs=[image, prompt, upscale, patch_size, stride, seed],
        outputs=[result, status],
    )

if __name__ == "__main__":
    print("🌐 启动 Gradio 界面...")
    print(f"🔗 访问地址: http://{WEBUI_CONFIG['server_name']}:{WEBUI_CONFIG['port']}")
    print("📸 示例已从 examples/ 目录加载")
    print("🎉 准备增强图片!")
    
    block.launch(
        server_name=WEBUI_CONFIG["server_name"], 
        server_port=WEBUI_CONFIG["port"],
        share=WEBUI_CONFIG["share"]
    )
