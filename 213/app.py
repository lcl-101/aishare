import gradio as gr
import torch
import argparse
from ovi.ovi_fusion_engine import OviFusionEngine, DEFAULT_CONFIG
from diffusers import FluxPipeline
import tempfile
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import clean_text, scale_hw_to_area_divisible
import csv
import os

# ----------------------------
# 解析命令行参数
# ----------------------------
parser = argparse.ArgumentParser(description="Ovi 联合视频 + 音频 Gradio 演示")
parser.add_argument(
    "--use_image_gen",
    action="store_true",
    help="启用图像生成界面 (使用 FluxPipeline)"
)
parser.add_argument(
    "--cpu_offload",
    action="store_true",
    help="为 OviFusionEngine 和 FluxPipeline 启用 CPU 卸载"
)
parser.add_argument(
    "--fp8",
    action="store_true",
    help="启用融合模型的 8 位量化",
)
parser.add_argument(
    "--qint8",
    action="store_true",
    help="启用融合模型的 8 位量化。无需下载额外的模型。",
)
parser.add_argument("--server_name", type=str, default="0.0.0.0", help="IP 地址，局域网访问请改为 0.0.0.0")
parser.add_argument("--server_port", type=int, default=7860, help="使用端口")
parser.add_argument("--share", action="store_true", help="启用 gradio 分享")
parser.add_argument("--mcp_server", action="store_true", help="启用 MCP 服务")
args = parser.parse_args()


# 初始化 OviFusionEngine
enable_cpu_offload = args.cpu_offload or args.use_image_gen
use_image_gen = args.use_image_gen
fp8 = args.fp8
qint8 = args.qint8
print(f"正在加载模型... {enable_cpu_offload=}, {use_image_gen=}, {fp8=}, {qint8=} for gradio demo")
DEFAULT_CONFIG["cpu_offload"] = (
    enable_cpu_offload  # 如果启用图像生成，则始终使用 cpu 卸载
)
DEFAULT_CONFIG["mode"] = "t2v"  # 硬编码，因为它总是使用 cpu 卸载
DEFAULT_CONFIG["fp8"] = fp8
DEFAULT_CONFIG["qint8"] = qint8
ovi_engine = OviFusionEngine()
flux_model = None
if fp8 or qint8:
    assert not use_image_gen, "使用 FluxPipeline 的图像生成不支持 fp8 量化。这是因为如果您无法运行 bf16 模型，您可能也无法运行图像生成模型"
    
if use_image_gen:
    flux_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
    flux_model.enable_model_cpu_offload() # 通过将模型卸载到 CPU 来节省一些 VRAM。如果您有足够的 GPU VRAM，请删除此项
print("模型加载完成")


# 加载示例数据
def load_t2v_examples():
    """加载文本到视频示例"""
    examples = []
    csv_path = "example_prompts/gpt_examples_t2v.csv"
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 返回格式: [video_text_prompt, image, video_height, video_width, ...]
                examples.append([
                    row['text_prompt'],  # video_text_prompt
                    None,  # image (T2V 没有图像)
                    512,  # video_height
                    992,  # video_width
                    100,  # video_seed
                    "unipc",  # solver_name
                    50,  # sample_steps
                    5.0,  # shift
                    4.0,  # video_guidance_scale
                    3.0,  # audio_guidance_scale
                    11,  # slg_layer
                    "",  # video_negative_prompt
                    "",  # audio_negative_prompt
                ])
    return examples


def load_i2v_examples():
    """加载图像到视频示例"""
    examples = []
    csv_path = "example_prompts/gpt_examples_i2v.csv"
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 返回格式: [video_text_prompt, image, video_height, video_width, ...]
                examples.append([
                    row['text_prompt'],  # video_text_prompt
                    row['image_path'],  # image
                    512,  # video_height
                    992,  # video_width
                    100,  # video_seed
                    "unipc",  # solver_name
                    50,  # sample_steps
                    5.0,  # shift
                    4.0,  # video_guidance_scale
                    3.0,  # audio_guidance_scale
                    11,  # slg_layer
                    "",  # video_negative_prompt
                    "",  # audio_negative_prompt
                ])
    return examples


def generate_video(
    text_prompt,
    image,
    video_frame_height,
    video_frame_width,
    video_seed,
    solver_name,
    sample_steps,
    shift,
    video_guidance_scale,
    audio_guidance_scale,
    slg_layer,
    video_negative_prompt,
    audio_negative_prompt,
):
    try:
        image_path = None
        if image is not None:
            image_path = image

        generated_video, generated_audio, _ = ovi_engine.generate(
            text_prompt=text_prompt,
            image_path=image_path,
            video_frame_height_width=[video_frame_height, video_frame_width],
            seed=video_seed,
            solver_name=solver_name,
            sample_steps=sample_steps,
            shift=shift,
            video_guidance_scale=video_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            slg_layer=slg_layer,
            video_negative_prompt=video_negative_prompt,
            audio_negative_prompt=audio_negative_prompt,
        )

        tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = tmpfile.name
        save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)

        return output_path
    except Exception as e:
        print(f"视频生成过程中出错: {e}")
        return None


def generate_image(text_prompt, image_seed, image_height, image_width):
    if flux_model is None:
        return None
    text_prompt = clean_text(text_prompt)
    print(f"正在生成图像，提示词='{text_prompt}', 种子={image_seed}, 尺寸=({image_height},{image_width})")

    image_h, image_w = scale_hw_to_area_divisible(image_height, image_width, area=1024 * 1024)
    image = flux_model(
        text_prompt,
        height=image_h,
        width=image_w,
        guidance_scale=4.5,
        generator=torch.Generator().manual_seed(int(image_seed))
    ).images[0]

    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmpfile.name)
    return tmpfile.name


# 构建用户界面
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 Ovi 联合视频 + 音频生成演示")
    gr.Markdown(
        """
        ## 📘 使用说明

        请按顺序完成以下步骤：

        1️⃣ **输入文本提示词** — 描述您想要的视频内容。(如果启用了图像生成，此文本提示词将被共享用于图像生成。)  
        2️⃣ **上传或生成图像** — 上传一张图像或生成一张（如果启用了图像生成功能）。(如果您没有看到图像生成选项，请确保使用 `--use_image_gen` 参数运行脚本。)  
        3️⃣ **配置视频选项** — 设置分辨率、种子、求解器和其他参数。(它将自动使用上传/生成的图像作为第一帧，无论您在视频生成时屏幕上显示哪个。)  
        4️⃣ **生成视频** — 点击按钮生成您的最终视频和音频。  
        5️⃣ **查看结果** — 您生成的视频将显示在下方。  

        ---

        ### 💡 提示
        1. 为了获得最佳效果，请使用详细和具体的文本提示词。  
        2. 确保文本提示词格式正确，即要说的话应该用 `<S>...<E>` 包裹。可以在末尾提供可选的音频描述，用 `<AUDCAP> ... <ENDAUDCAP>` 包裹，请参考示例  
        3. 不要因为糟糕或奇怪的结果而气馁，检查提示词格式并尝试不同的种子、CFG 值和 SLG 层。
        """
    )


    with gr.Row():
        with gr.Column():
            # 图像部分
            image = gr.Image(type="filepath", label="首帧图像 (上传或生成)")

            if args.use_image_gen:
                with gr.Accordion("🖼️ 图像生成选项", visible=True):
                    image_text_prompt = gr.Textbox(label="图像提示词", placeholder="描述您想要生成的图像...")
                    image_seed = gr.Number(minimum=0, maximum=100000, value=42, label="图像种子")
                    image_height = gr.Number(minimum=128, maximum=1280, value=720, step=32, label="图像高度")
                    image_width = gr.Number(minimum=128, maximum=1280, value=1280, step=32, label="图像宽度")
                    gen_img_btn = gr.Button("生成图像 🎨")
            else:
                gen_img_btn = None

            with gr.Accordion("🎬 视频生成选项", open=True):
                video_text_prompt = gr.Textbox(label="视频提示词", placeholder="描述您想要的视频...")
                video_height = gr.Number(minimum=128, maximum=1280, value=512, step=32, label="视频高度")
                video_width = gr.Number(minimum=128, maximum=1280, value=992, step=32, label="视频宽度")

                video_seed = gr.Number(minimum=0, maximum=100000, value=100, label="视频种子")
                solver_name = gr.Dropdown(
                    choices=["unipc", "euler", "dpm++"], value="unipc", label="求解器名称"
                )
                sample_steps = gr.Number(
                    value=50,
                    label="采样步数",
                    precision=0,
                    minimum=20,
                    maximum=100
                )
                shift = gr.Slider(minimum=0.0, maximum=20.0, value=5.0, step=1.0, label="偏移量")
                video_guidance_scale = gr.Slider(minimum=0.0, maximum=10.0, value=4.0, step=0.5, label="视频引导比例")
                audio_guidance_scale = gr.Slider(minimum=0.0, maximum=10.0, value=3.0, step=0.5, label="音频引导比例")
                slg_layer = gr.Number(minimum=-1, maximum=30, value=11, step=1, label="SLG 层")
                video_negative_prompt = gr.Textbox(label="视频负面提示词", placeholder="视频中要避免的内容")
                audio_negative_prompt = gr.Textbox(label="音频负面提示词", placeholder="音频中要避免的内容")

                run_btn = gr.Button("生成视频 🚀")

        with gr.Column():
            output_path = gr.Video(label="生成的视频")

    # 添加示例部分
    with gr.Accordion("📝 示例", open=False):
        gr.Markdown("### 文本到视频 (T2AV) 示例")
        gr.Markdown("点击下面的示例自动填充参数（无需首帧图像）")
        
        t2v_examples = gr.Examples(
            examples=load_t2v_examples(),
            inputs=[
                video_text_prompt, image, video_height, video_width, video_seed, 
                solver_name, sample_steps, shift, video_guidance_scale, 
                audio_guidance_scale, slg_layer, video_negative_prompt, audio_negative_prompt
            ],
            label="T2AV 示例",
            examples_per_page=5
        )
        
        gr.Markdown("### 图像到视频 (I2AV) 示例")
        gr.Markdown("点击下面的示例自动填充参数（包含首帧图像）")
        
        i2v_examples = gr.Examples(
            examples=load_i2v_examples(),
            inputs=[
                video_text_prompt, image, video_height, video_width, video_seed, 
                solver_name, sample_steps, shift, video_guidance_scale, 
                audio_guidance_scale, slg_layer, video_negative_prompt, audio_negative_prompt
            ],
            label="I2AV 示例",
            examples_per_page=5
        )

    if args.use_image_gen and gen_img_btn is not None:
        gen_img_btn.click(
            fn=generate_image,
            inputs=[image_text_prompt, image_seed, image_height, image_width],
            outputs=[image],
        )

    # 连接视频生成
    run_btn.click(
        fn=generate_video,
        inputs=[
            video_text_prompt, image, video_height, video_width, video_seed, solver_name,
            sample_steps, shift, video_guidance_scale, audio_guidance_scale,
            slg_layer, video_negative_prompt, audio_negative_prompt,
        ],
        outputs=[output_path],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )
