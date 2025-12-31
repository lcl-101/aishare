"""
LongVie - 长视频生成 Web 应用
基于 Gradio 的交互式界面
"""

import os
import json
import torch
import gradio as gr
import decord
import tempfile
import shutil
from PIL import Image
from datetime import datetime

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new_longvie import LongViePipeline, ModelConfig

# ==================== 配置参数 ====================
TARGET_SIZE = (640, 352)  # 目标分辨率 (宽, 高)

# 模型路径配置（根据实际下载位置调整）
BASE_MODEL_PATH = "./checkpoints"
CONTROL_WEIGHT_PATH = "./checkpoints/LongVie2/control.safetensors"
DIT_WEIGHT_PATH = "./checkpoints/LongVie2/dit.safetensors"

# 默认负向提示词
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# 示例提示词（来自官方示例）
EXAMPLE_PROMPTS = {
    "ride_horse": "The video captures a serene journey through a snowy landscape. A lone rider, clad in a dark jacket and a wide-brimmed hat, is seen from behind, riding a dark-colored horse along a snow-covered trail. The path meanders through a winter wonderland, flanked by frosty bushes and evergreen trees dusted with snow. In the distance, majestic snow-capped mountains rise against a backdrop of a partly cloudy sky, with sunlight filtering through the clouds, casting a soft glow over the scene. The rider's steady pace and the tranquil surroundings evoke a sense of peaceful solitude and the beauty of nature in its winter guise.",
    "valley": "The video opens with an aerial view of a stunning autumnal valley. The camera begins at a high altitude, providing a broad overview of the landscape. It then gradually descends, moving closer to the ground, giving a more intimate look at the river that winds through the valley. The river is a vibrant green, contrasting beautifully with the surrounding dense forest of trees adorned with fiery hues of orange, red, and yellow."
}

# ==================== 全局变量 ====================
pipe = None


def load_pipeline():
    """加载模型管道"""
    global pipe
    
    if pipe is not None:
        return "✅ 模型已加载"
    
    try:
        import glob as glob_module
        
        # 设置 ModelScope 模型目录为本地 checkpoints
        # 模型 ID 映射到本地目录名
        model_id = "Wan2.1-I2V-14B-480P"
        model_base_path = f"{BASE_MODEL_PATH}/{model_id}"
        
        # 展开 glob 模式获取 DiT 权重文件列表
        dit_files = sorted(glob_module.glob(f"{model_base_path}/diffusion_pytorch_model*.safetensors"))
        
        pipe = LongViePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            use_usp=False,
            model_configs=[
                ModelConfig(
                    path=dit_files,  # 列表形式的多个文件
                    offload_device="cpu",
                    skip_download=True,
                ),
                ModelConfig(
                    path=f"{model_base_path}/models_t5_umt5-xxl-enc-bf16.pth",
                    offload_device="cpu",
                    skip_download=True,
                ),
                ModelConfig(
                    path=f"{model_base_path}/Wan2.1_VAE.pth",
                    offload_device="cpu",
                    skip_download=True,
                ),
                ModelConfig(
                    path=f"{model_base_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                    offload_device="cpu",
                    skip_download=True,
                ),
            ],
            redirect_common_files=False,
            control_weight_path=CONTROL_WEIGHT_PATH,
            dit_weight_path=DIT_WEIGHT_PATH,
        )
        pipe.enable_vram_management()
        return "✅ 模型加载成功！"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ 模型加载失败: {str(e)}"


def load_image(path):
    """加载并调整图片尺寸"""
    return Image.open(path).convert("RGB").resize(TARGET_SIZE)


def resize_video_frames(video_np):
    """调整视频帧尺寸"""
    return [Image.fromarray(frame).resize(TARGET_SIZE) for frame in video_np]


def load_json_file(json_path):
    """加载 JSON 配置文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_video_segments(
    first_image,
    json_path,
    seed,
    negative_prompt,
    progress=gr.Progress(track_tqdm=True)
):
    """
    生成视频片段
    
    Args:
        first_image: 首帧图片路径
        json_path: 条件 JSON 文件路径
        seed: 随机种子
        negative_prompt: 负向提示词
        progress: 进度条
    
    Returns:
        生成的视频列表, 状态信息
    """
    global pipe
    
    if pipe is None:
        return None, "❌ 请先加载模型"
    
    if first_image is None or first_image == "":
        return None, "❌ 请上传首帧图片或选择示例"
    
    if json_path is None or json_path == "":
        return None, "❌ 请输入条件配置文件路径 (JSON) 或选择示例"
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        return None, f"❌ JSON 配置文件不存在: {json_path}"
    
    if not os.path.exists(first_image):
        return None, f"❌ 首帧图片不存在: {first_image}"
    
    try:
        # 读取 JSON 配置
        samples = load_json_file(json_path)
        
        # 处理首帧图片
        image = load_image(first_image)
        
        # 创建临时输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(tempfile.gettempdir(), f"longvie_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        history = []
        noise = None
        video_paths = []
        
        progress(0, desc="开始生成视频...")
        
        for i, sample in enumerate(samples):
            progress((i + 0.1) / len(samples), desc=f"正在生成第 {i + 1}/{len(samples)} 个片段...")
            
            # 加载深度和轨迹视频
            dense_vr = decord.VideoReader(sample["depth"])
            sparse_vr = decord.VideoReader(sample["track"])
            
            dense_frames = resize_video_frames(dense_vr[:].asnumpy())
            sparse_frames = resize_video_frames(sparse_vr[:].asnumpy())
            
            # 生成视频
            video, noise = pipe(
                input_image=image,
                prompt=sample["text"],
                negative_prompt=negative_prompt,
                seed=seed,
                tiled=False,
                height=TARGET_SIZE[1],
                width=TARGET_SIZE[0],
                dense_video=dense_frames,
                sparse_video=sparse_frames,
                history=history,
                noise=noise,
            )
            
            # 更新图片和历史
            image = video[-1]
            history = video[-8:]
            
            # 保存视频
            save_path = os.path.join(output_dir, f"segment_{i:02d}.mp4")
            save_video(video, save_path, fps=16, quality=10)
            video_paths.append(save_path)
            
            progress((i + 1) / len(samples), desc=f"第 {i + 1}/{len(samples)} 个片段生成完成")
        
        # 合并所有视频片段
        if len(video_paths) > 1:
            merged_path = os.path.join(output_dir, "merged_video.mp4")
            merge_videos(video_paths, merged_path)
            return merged_path, f"✅ 成功生成 {len(video_paths)} 个视频片段并合并！保存至: {output_dir}"
        elif len(video_paths) == 1:
            return video_paths[0], f"✅ 视频生成成功！保存至: {video_paths[0]}"
        else:
            return None, "❌ 没有生成任何视频片段"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 生成失败: {str(e)}"


def merge_videos(video_paths, output_path):
    """合并多个视频片段"""
    import subprocess
    
    # 创建文件列表
    list_file = output_path.replace(".mp4", "_list.txt")
    with open(list_file, "w") as f:
        for path in video_paths:
            f.write(f"file '{path}'\n")
    
    # 使用 ffmpeg 合并
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_path
    ]
    subprocess.run(cmd, capture_output=True)
    
    # 清理临时文件
    os.remove(list_file)


def generate_single_segment(
    first_image,
    prompt,
    depth_video,
    track_video,
    seed,
    negative_prompt,
    progress=gr.Progress(track_tqdm=True)
):
    """
    生成单个视频片段（简易模式）
    
    Args:
        first_image: 首帧图片路径
        prompt: 正向提示词
        depth_video: 深度视频路径
        track_video: 轨迹视频路径
        seed: 随机种子
        negative_prompt: 负向提示词
        progress: 进度条
    
    Returns:
        生成的视频, 状态信息
    """
    global pipe
    
    if pipe is None:
        return None, "❌ 请先加载模型"
    
    if first_image is None or first_image == "":
        return None, "❌ 请上传首帧图片或输入图片路径"
    
    if depth_video is None or depth_video == "" or track_video is None or track_video == "":
        return None, "❌ 请输入深度视频和轨迹视频路径"
    
    if not prompt or prompt.strip() == "":
        return None, "❌ 请输入提示词"
    
    # 检查文件是否存在
    if not os.path.exists(first_image):
        return None, f"❌ 首帧图片不存在: {first_image}"
    
    if not os.path.exists(depth_video):
        return None, f"❌ 深度视频不存在: {depth_video}"
    
    if not os.path.exists(track_video):
        return None, f"❌ 轨迹视频不存在: {track_video}"
    
    try:
        # 处理首帧图片
        image = load_image(first_image)
        
        progress(0.1, desc="加载控制视频...")
        
        # 加载深度和轨迹视频
        dense_vr = decord.VideoReader(depth_video)
        sparse_vr = decord.VideoReader(track_video)
        
        dense_frames = resize_video_frames(dense_vr[:].asnumpy())
        sparse_frames = resize_video_frames(sparse_vr[:].asnumpy())
        
        progress(0.2, desc="正在生成视频...")
        
        # 生成视频
        video, _ = pipe(
            input_image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            tiled=False,
            height=TARGET_SIZE[1],
            width=TARGET_SIZE[0],
            dense_video=dense_frames,
            sparse_video=sparse_frames,
            history=[],
            noise=None,
        )
        
        progress(0.9, desc="保存视频...")
        
        # 保存视频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(tempfile.gettempdir(), f"longvie_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "output.mp4")
        save_video(video, save_path, fps=16, quality=10)
        
        progress(1.0, desc="完成！")
        
        return save_path, f"✅ 视频生成成功！保存至: {save_path}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 生成失败: {str(e)}"


def load_example(example_name):
    """加载示例数据"""
    examples_map = {
        "ride_horse": {
            "image": "./example/ride_horse/first.png",
            "json": "./example/ride_horse/cond.json"
        },
        "valley": {
            "image": "./example/valley/first.png",
            "json": "./example/valley/cond.json"
        }
    }
    
    if example_name not in examples_map:
        return None, None, ""
    
    example = examples_map[example_name]
    image_path = example["image"]
    json_path = example["json"]
    
    # 读取 JSON 获取第一个片段的提示词
    try:
        samples = load_json_file(json_path)
        first_prompt = samples[0]["text"] if samples else ""
    except:
        first_prompt = ""
    
    return image_path, json_path, first_prompt


# ==================== Gradio 界面 ====================
def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(
        title="LongVie - 长视频生成",
        theme=gr.themes.Soft(),
        css="""
        .title { text-align: center; margin-bottom: 20px; }
        .description { text-align: center; color: #666; margin-bottom: 30px; }
        .example-img { max-height: 150px; }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="title">
            <h1>🎬 LongVie - 长视频生成</h1>
        </div>
        <div class="description">
            <p>基于深度和轨迹控制的长视频生成工具</p>
        </div>
        """)
        
        with gr.Tabs():
            # ==================== 长视频生成模式 ====================
            with gr.TabItem("🎬 长视频生成（推荐）"):
                gr.Markdown("""
                ### 🎥 长视频生成模式
                
                此模式通过**分段连续生成**实现长视频：
                - 系统会按顺序生成多个视频片段
                - 每个片段使用前一片段的最后8帧作为历史，保证连续性
                - 最终自动合并为完整的长视频
                
                **工作原理：** `首帧图片 → 片段1 → 片段2 → ... → 片段N → 合并 → 长视频`
                
                **需要准备：**
                1. **首帧图片**：视频的起始画面
                2. **条件配置文件（JSON）**：包含每个片段的提示词、深度视频和轨迹视频路径
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # 输入区域
                        batch_image = gr.Image(
                            label="首帧图片",
                            type="filepath",
                            height=200
                        )
                        batch_json = gr.Textbox(
                            label="条件配置文件路径 (JSON)",
                            placeholder="例如: ./example/ride_horse/cond.json"
                        )
                        
                    with gr.Column(scale=1):
                        batch_seed = gr.Number(
                            label="随机种子",
                            value=0,
                            precision=0
                        )
                        batch_negative = gr.Textbox(
                            label="负向提示词",
                            value=DEFAULT_NEGATIVE_PROMPT,
                            lines=4
                        )
                        batch_generate_btn = gr.Button(
                            "🎬 开始生成长视频",
                            variant="primary",
                            size="lg"
                        )
                
                with gr.Row():
                    batch_output = gr.Video(label="生成的长视频")
                    batch_status = gr.Textbox(label="状态信息", lines=3)
                
                # 官方长视频示例
                gr.Markdown("""
                ### 🎯 官方示例（点击加载）
                
                | 示例 | 片段数 | 预计时长 | 描述 |
                |------|--------|----------|------|
                | 骑马雪景 | 10 个片段 | ~40秒 | 骑马穿越雪山的第三人称视角 |
                | 秋天山谷 | 12 个片段 | ~48秒 | 秋天山谷河流的航拍视角 |
                """)
                gr.Examples(
                    examples=[
                        ["./example/ride_horse/first.png", "./example/ride_horse/cond.json"],
                        ["./example/valley/first.png", "./example/valley/cond.json"],
                    ],
                    inputs=[batch_image, batch_json],
                    label="选择示例",
                    examples_per_page=2,
                )
            
            # ==================== 单片段测试模式 ====================
            with gr.TabItem("🧪 单片段测试"):
                gr.Markdown("""
                ### 🧪 单片段测试模式
                
                此模式用于**快速测试单个片段**的生成效果，适合：
                - 调试参数和提示词
                - 预览控制视频的效果
                - 快速验证想法
                
                **注意：** 如需生成完整长视频，请使用「长视频生成」模式。
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        single_image = gr.Image(
                            label="首帧图片",
                            type="filepath",
                            height=200
                        )
                        single_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入描述视频内容的提示词...",
                            lines=4,
                            value=EXAMPLE_PROMPTS["ride_horse"]
                        )
                        
                    with gr.Column(scale=1):
                        single_depth = gr.Video(
                            label="深度视频 (MP4)",
                            height=150
                        )
                        single_track = gr.Video(
                            label="轨迹视频 (MP4)",
                            height=150
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        single_seed = gr.Number(
                            label="随机种子",
                            value=0,
                            precision=0
                        )
                    with gr.Column(scale=2):
                        single_negative = gr.Textbox(
                            label="负向提示词",
                            value=DEFAULT_NEGATIVE_PROMPT,
                            lines=2
                        )
                
                single_generate_btn = gr.Button(
                    "🎬 开始生成",
                    variant="primary",
                    size="lg"
                )
                
                with gr.Row():
                    single_output = gr.Video(label="生成结果")
                    single_status = gr.Textbox(label="状态信息", lines=3)
                
                # 官方示例
                gr.Markdown("### 🎯 官方示例（点击加载）")
                gr.Examples(
                    examples=[
                        [
                            "./example/ride_horse/first.png",
                            EXAMPLE_PROMPTS["ride_horse"],
                            "./example/ride_horse/depth_00.mp4",
                            "./example/ride_horse/track_00.mp4"
                        ],
                        [
                            "./example/valley/first.png",
                            EXAMPLE_PROMPTS["valley"],
                            "./example/valley/depth_00.mp4",
                            "./example/valley/track_00.mp4"
                        ],
                    ],
                    inputs=[single_image, single_prompt, single_depth, single_track],
                    label="选择示例",
                    examples_per_page=2,
                )
            
            # ==================== 使用帮助 ====================
            with gr.TabItem("❓ 使用帮助"):
                gr.Markdown("""
                ## LongVie 长视频生成使用指南
                
                ### 什么是 LongVie?
                LongVie 是一个基于深度和轨迹控制的长视频生成模型，可以根据首帧图片和控制信号生成高质量的连续视频。
                
                ### 准备工作
                
                #### 1. 首帧图片
                - 格式：PNG 或 JPG
                - 建议尺寸：640x352 或类似比例
                - 图片将被自动调整为 640x352
                
                #### 2. 深度视频 (Depth Video)
                - 用于控制视频的空间结构和景深
                - 可以使用深度估计模型（如 MiDaS）从原视频生成
                - 格式：MP4
                
                #### 3. 轨迹视频 (Track Video)
                - 用于控制物体的运动轨迹
                - 可以使用光流或点追踪算法从原视频生成
                - 格式：MP4
                
                #### 4. JSON 配置文件（批量模式）
                ```json
                [
                    {
                        "text": "描述第一个片段的提示词",
                        "depth": "./path/to/depth_00.mp4",
                        "track": "./path/to/track_00.mp4"
                    },
                    {
                        "text": "描述第二个片段的提示词",
                        "depth": "./path/to/depth_01.mp4",
                        "track": "./path/to/track_01.mp4"
                    }
                ]
                ```
                
                ### 提示词技巧
                - 使用详细的场景描述
                - 包含相机运动方向（如：相机前进、向左转等）
                - 描述光照和氛围
                - 英文提示词效果通常更好
                
                ### 常见问题
                
                **Q: 为什么生成速度很慢？**
                A: LongVie 使用 14B 参数的大模型，需要较强的 GPU。建议使用至少 24GB 显存的显卡。
                
                **Q: 如何获取深度和轨迹视频？**
                A: 可以使用项目中的 `utils/get_depth.py` 和 `utils/get_track.py` 脚本生成。
                
                **Q: 生成的视频不连续怎么办？**
                A: 使用批量模式，确保 JSON 中的片段顺序正确，系统会自动使用历史帧保持连续性。
                """)
        
        # ==================== 事件绑定 ====================
        
        # 批量生成
        batch_generate_btn.click(
            fn=generate_video_segments,
            inputs=[batch_image, batch_json, batch_seed, batch_negative],
            outputs=[batch_output, batch_status]
        )
        
        # 单片段生成
        single_generate_btn.click(
            fn=generate_single_segment,
            inputs=[single_image, single_prompt, single_depth, single_track, single_seed, single_negative],
            outputs=[single_output, single_status]
        )
    
    return demo


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 设置环境变量
    os.environ["MODELSCOPE_CACHE"] = "./checkpoints"
    os.environ["HF_HOME"] = "./checkpoints"
    
    # 启动时自动加载模型
    print("正在加载模型，请稍候...")
    status = load_pipeline()
    print(status)
    
    if pipe is None:
        print("模型加载失败，程序退出")
        exit(1)
    
    # 创建并启动 UI
    demo = create_ui()
    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
