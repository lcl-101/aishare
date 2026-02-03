"""
MOVA Gradio Web Application
基于 MOVA 模型的视频生成 Web 应用
"""

import os
import gc
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime

import gradio as gr
import torch
import torch.distributed as dist
from PIL import Image
from torch.distributed.device_mesh import DeviceMesh

# 保存原始的 tqdm 模块，用于避免 Gradio 的 tqdm 包装冲突
# 这是业界通用做法：Gradio 会 monkey-patch tqdm 以显示进度条，
# 但这可能与某些库（如 diffusers pipeline）的 tqdm 调用冲突。
# 解决方案是在调用这些库时临时恢复原始 tqdm。
import tqdm as _original_tqdm_module
_original_tqdm = _original_tqdm_module.tqdm
_original_tqdm_auto = None
try:
    from tqdm import auto as _tqdm_auto_module
    _original_tqdm_auto = _tqdm_auto_module.tqdm
except ImportError:
    pass

from mova.datasets.transforms.custom import crop_and_resize
from mova.diffusion.pipelines.pipeline_mova import MOVA
from mova.utils.data import save_video_with_audio


@contextmanager
def restore_original_tqdm():
    """
    上下文管理器：临时恢复原始的 tqdm 模块。
    
    Gradio 会通过 monkey-patching 替换 tqdm.tqdm 为自己的 TqdmProgress 类，
    以便在 UI 上显示进度条。但这会导致某些情况下出现 "list index out of range" 错误，
    因为 Gradio 的包装器依赖于特定的调用上下文。
    
    这个上下文管理器在执行 pipeline 时临时恢复原始的 tqdm，
    确保 pipeline 内部的 tqdm 调用不受 Gradio 包装影响。
    
    这是业界通用的解决方案，被 Hugging Face Spaces、ComfyUI 等广泛采用。
    """
    import tqdm
    import tqdm.auto
    
    # 保存 Gradio 包装后的 tqdm
    gradio_tqdm = tqdm.tqdm
    gradio_tqdm_auto = tqdm.auto.tqdm
    
    try:
        # 临时恢复原始 tqdm
        tqdm.tqdm = _original_tqdm
        tqdm.auto.tqdm = _original_tqdm_auto if _original_tqdm_auto else _original_tqdm
        yield
    finally:
        # 恢复 Gradio 的 tqdm 包装
        tqdm.tqdm = gradio_tqdm
        tqdm.auto.tqdm = gradio_tqdm_auto

# 默认负面提示词
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
)

# 模型路径
CKPT_PATH = "checkpoints/MOVA-720p/"

# 输出目录
OUTPUT_DIR = "data/gradio_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局变量
pipe = None
mesh = None


def init_distributed():
    """初始化分布式环境"""
    global mesh
    
    # 设置单 GPU 环境
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    
    local_rank = 0
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        torch.cuda.set_device(local_rank)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    cp_size = 1
    dp_size = world_size // cp_size
    mesh = DeviceMesh(
        "cuda",
        torch.arange(dist.get_world_size()).view(dp_size, cp_size),
        mesh_dim_names=("dp", "cp"),
    )
    
    return mesh


def load_model():
    """加载 MOVA 模型"""
    global pipe, mesh
    
    if pipe is not None:
        return pipe, mesh
    
    print("正在初始化分布式环境...")
    mesh = init_distributed()
    
    print(f"正在加载模型: {CKPT_PATH}")
    torch_dtype = torch.bfloat16
    pipe = MOVA.from_pretrained(CKPT_PATH, torch_dtype=torch_dtype)
    
    # 直接加载到 GPU，不使用 offload（H20 显卡有 141GB 显存）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    pipe.to(torch.device("cuda", local_rank))
    
    print("模型加载完成！")
    return pipe, mesh


def generate_video(
    prompt: str,
    ref_image,
    height: int,
    width: int,
    num_frames: int,
    fps: float,
    num_inference_steps: int,
    cfg_scale: float,
    sigma_shift: float,
    seed: int,
    negative_prompt: str,
):
    """生成视频"""
    global pipe, mesh
    
    if pipe is None:
        pipe, mesh = load_model()
    
    if ref_image is None:
        raise gr.Error("请上传参考图片！")
    
    if not prompt.strip():
        raise gr.Error("请输入提示词！")
    
    try:
        # 处理参考图片
        if isinstance(ref_image, str):
            img = Image.open(ref_image).convert("RGB")
        else:
            img = Image.fromarray(ref_image).convert("RGB")
        
        ref_img = crop_and_resize(img, height=height, width=width)
        
        # 设置随机种子
        torch.manual_seed(seed)
        
        print(f"开始生成视频...")
        print(f"提示词: {prompt[:100]}...")
        print(f"分辨率: {width}x{height}, 帧数: {num_frames}, FPS: {fps}")
        
        # 生成视频和音频
        # 使用 restore_original_tqdm 上下文管理器避免 Gradio tqdm 包装冲突
        with restore_original_tqdm():
            video, audio = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                image=ref_img,
                height=height,
                width=width,
                video_fps=fps,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                cfg_scale=cfg_scale,
                seed=seed,
                cp_mesh=mesh["cp"],
                remove_video_dit=False,
            )
        
        # 保存视频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        output_path = os.path.join(OUTPUT_DIR, f"mova_{timestamp}_{unique_id}.mp4")
        
        audio_save = audio[0].cpu().squeeze()
        
        save_video_with_audio(
            video[0],
            audio_save,
            output_path,
            fps=fps,
            sample_rate=pipe.audio_sample_rate,
            quality=9,
        )
        
        print(f"视频已保存到: {output_path}")
        
        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()
        
        return output_path
        
    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        raise gr.Error(f"生成失败: {str(e)}")


# 示例数据
SINGLE_PERSON_EXAMPLE = {
    "prompt": 'A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks. At one point, he says, "I would also say that this election in Germany wasn\'t surprising."',
    "ref_image": "./assets/single_person.jpg",
}

MULTI_PERSON_EXAMPLE = {
    "prompt": 'The scene shows a man and a child walking together through a park, surrounded by open greenery and a calm, everyday atmosphere. As they stroll side by side, the man turns his head toward the child and asks with mild curiosity, in English, "What do you want to do when you grow up?" The boy answers with clear confidence, saying, "A bond trader. That\'s what Don does, and he took me to his office." The man lets out a soft chuckle, then responds warmly, "It\'s a good profession." as their walk continues at an unhurried pace, the conversation settling into a quiet, reflective moment.',
    "ref_image": "./assets/multi_person.png",
}

NEWS_ANCHOR_EXAMPLE = {
    "prompt": 'A female news anchor sits at a modern broadcast desk with multiple screens behind her showing news graphics. The studio lighting is professional and bright. She looks directly at the camera and speaks clearly, saying "Breaking news tonight: scientists have made a remarkable discovery that could change how we understand climate change. Our correspondent is live at the research facility with more details."',
    "ref_image": "./assets/single_person.jpg",
}


def load_single_person_example():
    """加载单人示例"""
    return SINGLE_PERSON_EXAMPLE["prompt"], SINGLE_PERSON_EXAMPLE["ref_image"]


def load_multi_person_example():
    """加载多人示例"""
    return MULTI_PERSON_EXAMPLE["prompt"], MULTI_PERSON_EXAMPLE["ref_image"]


def load_news_anchor_example():
    """加载新闻主播示例"""
    return NEWS_ANCHOR_EXAMPLE["prompt"], NEWS_ANCHOR_EXAMPLE["ref_image"]


# 构建 Gradio 界面
def create_ui():
    with gr.Blocks(
        title="MOVA - 多模态视频音频生成",
        theme=gr.themes.Soft(),
        css="""
        .youtube-banner {
            background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .youtube-banner a {
            color: white;
            text-decoration: none;
            font-size: 18px;
            font-weight: bold;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            line-height: 1.1;
        }
        .youtube-banner a:hover {
            text-decoration: underline;
        }
        .youtube-icon {
            display: block;
        }
        """
    ) as demo:
        # YouTube 频道信息横幅
        gr.HTML("""
        <div class="youtube-banner">
            <a href="https://www.youtube.com/@rongyi-ai" target="_blank">
                <svg class="youtube-icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white">
                    <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/>
                </svg>
                🎬 AI 技术分享频道 - 欢迎订阅关注！点击访问 →
            </a>
        </div>
        """)
        
        gr.Markdown("""
        # 🎬 MOVA - 多模态视频音频生成系统
        
        **MOVA** 是一个强大的视频音频生成模型，可以根据文本描述和参考图片生成带有同步音频的视频。
        
        ### 使用说明：
        1. 上传一张参考图片（人物照片）
        2. 输入详细的场景和对话描述（英文效果最佳）
        3. 调整生成参数（可选）
        4. 点击"生成视频"按钮
        """)
        
        with gr.Tabs():
            with gr.TabItem("🎥 视频生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 输入区域
                        gr.Markdown("### 📝 输入设置")
                        
                        ref_image = gr.Image(
                            label="参考图片",
                            type="filepath",
                            height=300,
                        )
                        
                        prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入场景描述和对话内容（英文效果最佳）...",
                            lines=6,
                            max_lines=10,
                        )
                        
                        negative_prompt = gr.Textbox(
                            label="负面提示词",
                            value=NEGATIVE_PROMPT,
                            lines=3,
                            max_lines=5,
                        )
                        
                        # 示例按钮
                        gr.Markdown("### 📋 示例")
                        with gr.Row():
                            single_person_btn = gr.Button("👤 单人说话", size="sm")
                            multi_person_btn = gr.Button("👥 多人对话", size="sm")
                            news_anchor_btn = gr.Button("📺 新闻主播", size="sm")
                    
                    with gr.Column(scale=1):
                        # 参数设置
                        gr.Markdown("### ⚙️ 生成参数")
                        
                        with gr.Row():
                            height = gr.Slider(
                                label="视频高度",
                                minimum=480,
                                maximum=1080,
                                value=720,
                                step=16,
                            )
                            width = gr.Slider(
                                label="视频宽度",
                                minimum=640,
                                maximum=1920,
                                value=1280,
                                step=16,
                            )
                        
                        with gr.Row():
                            num_frames = gr.Slider(
                                label="视频帧数",
                                minimum=49,
                                maximum=289,
                                value=193,
                                step=8,
                                info="帧数越多，视频越长",
                            )
                            fps = gr.Slider(
                                label="帧率 (FPS)",
                                minimum=12,
                                maximum=30,
                                value=24,
                                step=1,
                            )
                        
                        with gr.Row():
                            num_inference_steps = gr.Slider(
                                label="推理步数",
                                minimum=20,
                                maximum=100,
                                value=50,
                                step=5,
                                info="步数越多，质量越高，但速度越慢",
                            )
                            cfg_scale = gr.Slider(
                                label="CFG 强度",
                                minimum=1.0,
                                maximum=15.0,
                                value=5.0,
                                step=0.5,
                                info="控制生成内容与提示词的匹配程度",
                            )
                        
                        with gr.Row():
                            sigma_shift = gr.Slider(
                                label="Sigma 偏移",
                                minimum=1.0,
                                maximum=10.0,
                                value=5.0,
                                step=0.5,
                            )
                            seed = gr.Number(
                                label="随机种子",
                                value=42,
                                precision=0,
                                info="相同种子可复现结果",
                            )
                        
                        # 生成按钮
                        generate_btn = gr.Button(
                            "🚀 生成视频",
                            variant="primary",
                            size="lg",
                        )
                        
                        # 输出区域
                        gr.Markdown("### 🎬 生成结果")
                        output_video = gr.Video(
                            label="生成的视频",
                            height=400,
                        )
            
            with gr.TabItem("📖 使用帮助"):
                gr.Markdown("""
                ## 📖 详细使用指南
                
                ### 1️⃣ 参考图片要求
                - 建议使用清晰的人物正面或侧面照片
                - 图片会自动裁剪和调整大小以匹配目标分辨率
                - 支持 JPG、PNG 等常见图片格式
                
                ### 2️⃣ 提示词编写技巧
                - **场景描述**：描述环境、光线、氛围等
                - **人物动作**：描述说话、表情、肢体动作
                - **对话内容**：使用引号包含具体对话，如 `"Hello, how are you?"`
                - **语言**：建议使用英文编写提示词，效果最佳
                
                ### 3️⃣ 参数说明
                | 参数 | 说明 | 建议值 |
                |------|------|--------|
                | 视频高度/宽度 | 输出视频分辨率 | 720x1280 (720p) |
                | 视频帧数 | 控制视频时长 | 193 帧 ≈ 8 秒 |
                | 帧率 | 视频播放速度 | 24 FPS |
                | 推理步数 | 影响生成质量 | 50 步 |
                | CFG 强度 | 提示词匹配度 | 5.0 |
                | 随机种子 | 可复现结果 | 任意整数 |
                
                ### 4️⃣ 常见问题
                - **生成速度慢**：减少帧数或推理步数可加快速度
                - **视频质量差**：增加推理步数，调整 CFG 强度
                - **口型不同步**：尝试调整提示词中的对话描述
                
                ### 5️⃣ 示例提示词
                
                **单人演讲场景：**
                ```
                A man in a blue blazer and glasses speaks in a formal indoor setting, 
                framed by wooden furniture and a filled bookshelf. At one point, he says, 
                "I would also say that this election in Germany wasn't surprising."
                ```
                
                **多人对话场景：**
                ```
                The scene shows a man and a child walking together through a park. 
                The man asks, "What do you want to do when you grow up?" 
                The boy answers, "A bond trader."
                ```
                """)
            
            with gr.TabItem("ℹ️ 关于"):
                gr.Markdown("""
                ## ℹ️ 关于 MOVA
                
                **MOVA** (Multimodal Omni Video-Audio) 是一个先进的多模态生成模型，
                能够同时生成高质量的视频和同步音频。
                
                ### 🔧 技术规格
                - **模型版本**: MOVA-720p
                - **支持分辨率**: 720p (1280x720)
                - **视频格式**: MP4 (H.264)
                - **音频格式**: AAC
                - **推理设备**: NVIDIA H20 (141GB VRAM)
                
                ### 📺 关注我的频道
                欢迎访问 [AI 技术分享频道](https://www.youtube.com/@rongyi-ai) 获取更多 AI 技术内容！
                
                ### 📄 许可证
                本项目遵循原始 MOVA 项目的许可证条款。
                """)
        
        # 绑定事件
        single_person_btn.click(
            fn=load_single_person_example,
            outputs=[prompt, ref_image],
        )
        
        multi_person_btn.click(
            fn=load_multi_person_example,
            outputs=[prompt, ref_image],
        )
        
        news_anchor_btn.click(
            fn=load_news_anchor_example,
            outputs=[prompt, ref_image],
        )
        
        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                ref_image,
                height,
                width,
                num_frames,
                fps,
                num_inference_steps,
                cfg_scale,
                sigma_shift,
                seed,
                negative_prompt,
            ],
            outputs=output_video,
        )
    
    return demo


if __name__ == "__main__":
    # 预加载模型
    print("=" * 60)
    print("MOVA Gradio Web Application")
    print("=" * 60)

    # 启动时加载模型（可通过环境变量 MOVA_PRELOAD=0 关闭）
    if os.environ.get("MOVA_PRELOAD", "1") != "0":
        try:
            load_model()
        except Exception as e:
            print(f"模型预加载失败: {e}", file=sys.stderr)
            raise
    
    # 创建并启动应用
    demo = create_ui()
    
    # 启动 Gradio 服务
    demo.queue(max_size=5)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
