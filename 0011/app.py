#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACE-Step 1.5 简化版推理界面
- 使用 transformers 后端（不使用 vllm）
- 本地模型路径: checkpoints/Ace-Step1.5
- 中文界面
"""

import os
import sys
import random
import tempfile
import traceback
from typing import Optional, List, Dict, Any

import torch
import gradio as gr

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入 ACE-Step 模块
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.constants import VALID_LANGUAGES, DEFAULT_DIT_INSTRUCTION

# ==================== 全局配置 ====================
# 模型在 checkpoints/Ace-Step1.5/ 下，需要创建符号链接或直接指定路径
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "Ace-Step1.5")
DIT_MODEL_PATH = "acestep-v15-turbo"
LM_MODEL_PATH = "acestep-5Hz-lm-1.7B"

# 确保 handler 能找到模型：创建符号链接
def setup_model_paths():
    """设置模型路径，创建符号链接让 handler 能找到模型"""
    src_dir = os.path.join(PROJECT_ROOT, "checkpoints", "Ace-Step1.5")
    dst_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    
    # 需要链接的目录
    dirs_to_link = ["acestep-v15-turbo", "acestep-5Hz-lm-1.7B", "vae", "Qwen3-Embedding-0.6B"]
    
    for dir_name in dirs_to_link:
        src = os.path.join(src_dir, dir_name)
        dst = os.path.join(dst_dir, dir_name)
        
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                os.symlink(src, dst)
                print(f"创建符号链接: {dst} -> {src}")
            except Exception as e:
                print(f"创建符号链接失败 {dir_name}: {e}")

# 启动时设置路径
setup_model_paths()

# 全局 handler 实例
dit_handler: Optional[AceStepHandler] = None
llm_handler: Optional[LLMHandler] = None

# ==================== 初始化函数 ====================

def initialize_service(
    device: str = "auto",
    use_flash_attention: bool = True,
    offload_to_cpu: bool = False,
    init_llm: bool = True,
) -> str:
    """初始化模型服务"""
    global dit_handler, llm_handler
    
    try:
        # 设备检测
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        status_msgs = []
        
        # 初始化 DiT Handler
        dit_handler = AceStepHandler()
        dit_status, success = dit_handler.initialize_service(
            project_root=PROJECT_ROOT,
            config_path=DIT_MODEL_PATH,
            device=device,
            use_flash_attention=use_flash_attention,
            compile_model=False,
            offload_to_cpu=offload_to_cpu,
            offload_dit_to_cpu=offload_to_cpu,
        )
        status_msgs.append(dit_status)
        
        if not success:
            return f"❌ DiT 模型初始化失败:\n{dit_status}"
        
        # 初始化 LLM Handler（使用 PyTorch 后端）
        if init_llm:
            llm_handler = LLMHandler()
            llm_status, llm_success = llm_handler.initialize(
                checkpoint_dir=CHECKPOINT_DIR,
                lm_model_path=LM_MODEL_PATH,
                backend="pt",  # 使用 PyTorch 后端，不使用 vllm
                device=device,
                offload_to_cpu=offload_to_cpu,
            )
            status_msgs.append(llm_status)
            
            if not llm_success:
                return f"❌ LLM 模型初始化失败:\n{llm_status}"
        else:
            llm_handler = LLMHandler()
        
        return "\n\n".join(status_msgs)
        
    except Exception as e:
        error_msg = f"❌ 初始化失败: {str(e)}\n{traceback.format_exc()}"
        return error_msg


def generate_music(
    caption: str,
    lyrics: str,
    vocal_language: str = "unknown",
    instrumental: bool = False,
    bpm: Optional[float] = None,
    key_scale: str = "",
    time_signature: str = "",
    audio_duration: float = -1,
    batch_size: int = 1,
    inference_steps: int = 8,
    guidance_scale: float = 3.5,
    seed: int = -1,
    use_thinking: bool = True,
    reference_audio: Optional[str] = None,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    """生成音乐"""
    global dit_handler, llm_handler
    
    if dit_handler is None or dit_handler.model is None:
        return None, "❌ 请先初始化模型服务"
    
    try:
        # 处理歌词
        if instrumental:
            lyrics = "[Instrumental]"
        
        # 处理种子
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        
        # 处理时长
        if audio_duration <= 0:
            audio_duration = 30.0
        
        # 处理 BPM
        bpm_value = int(bpm) if bpm and bpm > 0 else None
        
        # 处理参考音频
        processed_refer_audio = None
        if reference_audio:
            processed_refer_audio = dit_handler.process_reference_audio(reference_audio)
        
        # 准备参考音频列表
        if processed_refer_audio is not None:
            refer_audios = [[processed_refer_audio]]
        else:
            refer_audios = [[torch.zeros(2, 30 * 48000)]]
        
        # LLM Chain-of-Thought 生成（如果启用）
        audio_code_string = ""
        lm_metadata = None
        
        if use_thinking and llm_handler and llm_handler.llm_initialized:
            # 使用 LLM 生成元数据和音频代码
            try:
                result = llm_handler.generate_with_stop_condition(
                    caption=caption or "",
                    lyrics=lyrics or "",
                    infer_type="llm_dit",
                    temperature=0.85,
                    cfg_scale=2.0,
                    negative_prompt="NO USER INPUT",
                    top_k=None,
                    top_p=0.9,
                    target_duration=audio_duration,
                    user_metadata={"bpm": bpm_value} if bpm_value else None,
                    use_cot_caption=True,
                    use_cot_language=True,
                    use_cot_metas=True,
                    use_constrained_decoding=True,
                    batch_size=batch_size,
                    seeds=[seed + i for i in range(batch_size)],
                )
                
                if result.get("success", False):
                    lm_metadata = result.get("metadata", {})
                    audio_codes = result.get("audio_codes", [])
                    if audio_codes:
                        audio_code_string = audio_codes if isinstance(audio_codes, list) else [audio_codes]
                    
                    # 更新元数据
                    if lm_metadata:
                        if not bpm_value and lm_metadata.get("bpm"):
                            try:
                                bpm_value = int(lm_metadata["bpm"])
                            except:
                                pass
                        if not key_scale and lm_metadata.get("keyscale"):
                            key_scale = lm_metadata["keyscale"]
                        if not time_signature and lm_metadata.get("timesignature"):
                            time_signature = lm_metadata["timesignature"]
                        if lm_metadata.get("caption"):
                            caption = lm_metadata["caption"]
                        if lm_metadata.get("vocal_language"):
                            vocal_language = lm_metadata["vocal_language"]
                            
            except Exception as e:
                print(f"LLM 生成警告: {e}")
        
        # 创建目标音频
        target_wavs = dit_handler.create_target_wavs(audio_duration)
        target_wavs = target_wavs.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 调用 DiT 生成
        result = dit_handler.generate_music(
            captions=caption,
            lyrics=lyrics,
            bpm=bpm_value,
            key_scale=key_scale,
            time_signature=time_signature,
            vocal_language=vocal_language,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            use_random_seed=False,
            seed=str(seed),
            reference_audio=reference_audio,
            audio_duration=audio_duration,
            batch_size=batch_size,
            src_audio=None,
            audio_code_string=audio_code_string if audio_code_string else "",
            repainting_start=None,
            repainting_end=None,
            instruction=DEFAULT_DIT_INSTRUCTION,
            audio_cover_strength=1.0,
            task_type="text2music",
            use_adg=False,
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            shift=3.0,
        )
        
        if not result.get("success", False):
            return None, f"❌ 生成失败: {result.get('error', '未知错误')}"
        
        # 获取生成的音频
        audios = result.get("audios", [])
        if not audios:
            return None, "❌ 未生成任何音频"
        
        # 保存音频文件
        output_files = []
        for i, audio_info in enumerate(audios):
            audio_tensor = audio_info.get("tensor")
            if audio_tensor is not None:
                # 保存到临时文件
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                import torchaudio
                torchaudio.save(temp_file.name, audio_tensor.cpu(), 48000)
                output_files.append(temp_file.name)
        
        status_msg = f"✅ 成功生成 {len(output_files)} 个音频\n"
        status_msg += f"种子: {seed}\n"
        status_msg += f"时长: {audio_duration}秒\n"
        if bpm_value:
            status_msg += f"BPM: {bpm_value}\n"
        if key_scale:
            status_msg += f"调性: {key_scale}\n"
        
        # 返回第一个音频
        if output_files:
            return output_files[0], status_msg
        else:
            return None, "❌ 音频保存失败"
            
    except Exception as e:
        error_msg = f"❌ 生成失败: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


def random_caption() -> str:
    """生成随机示例描述"""
    examples = [
        "A gentle acoustic guitar melody with warm fingerpicking patterns, soft ambient pads, and a dreamy atmosphere",
        "Upbeat electronic dance music with punchy bass, catchy synth hooks, and energetic drums",
        "Classical piano piece with emotional arpeggios, dynamic crescendos, and romantic harmonies",
        "Lo-fi hip hop beat with dusty vinyl textures, mellow keys, and relaxed drums",
        "Epic orchestral soundtrack with dramatic strings, powerful brass, and cinematic percussion",
        "Jazz fusion with smooth saxophone, walking bass, and sophisticated chord progressions",
        "Ambient soundscape with ethereal pads, nature sounds, and meditative textures",
        "Rock anthem with distorted guitars, driving drums, and powerful vocals",
    ]
    return random.choice(examples)


def load_example(language: str = "zh") -> tuple:
    """加载示例数据
    
    Args:
        language: 语言代码 "zh" 或 "en"
    
    Returns:
        (caption, lyrics, vocal_language, bpm, key_scale, time_signature, duration)
    """
    import json
    import glob
    import re
    
    examples_dir = os.path.join(PROJECT_ROOT, "examples", "text2music")
    
    # 获取所有示例文件
    all_files = glob.glob(os.path.join(examples_dir, "example_*.json"))
    
    # 筛选指定语言的示例
    matching_files = []
    for f in all_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                if data.get("language", "") == language:
                    lyrics = data.get("lyrics", "")
                    # 对于中文，检查是否包含真正的中文字符（排除拼音格式）
                    if language == "zh":
                        # 检查是否包含中文字符
                        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', lyrics))
                        # 排除拼音格式 [zh] xxx
                        has_pinyin = bool(re.search(r'\[zh\]\s*[a-z]', lyrics))
                        if has_chinese and not has_pinyin:
                            matching_files.append(f)
                    else:
                        matching_files.append(f)
        except:
            continue
    
    if not matching_files:
        # 如果没找到，返回默认值
        return "", "", "unknown", None, "", "", 30
    
    # 随机选择一个
    chosen_file = random.choice(matching_files)
    
    try:
        with open(chosen_file, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        caption = data.get("caption", "")
        lyrics = data.get("lyrics", "")
        vocal_language = data.get("language", "unknown")
        bpm = data.get("bpm", None)
        key_scale = data.get("keyscale", "")
        time_signature = str(data.get("timesignature", "")) if data.get("timesignature") else ""
        duration = data.get("duration", 30)
        
        return caption, lyrics, vocal_language, bpm, key_scale, time_signature, duration
    except Exception as e:
        print(f"加载示例失败: {e}")
        return "", "", "unknown", None, "", "", 30


def load_chinese_example() -> tuple:
    """加载中文示例"""
    return load_example("zh")


def load_english_example() -> tuple:
    """加载英文示例"""
    return load_example("en")


# ==================== Gradio 界面 ====================

def create_ui():
    """创建 Gradio 界面"""
    
    # 自定义 CSS
    custom_css = """
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .header-banner h1 {
        color: white;
        margin: 0 0 5px 0;
        font-size: 28px;
    }
    .header-banner p {
        color: rgba(255,255,255,0.9);
        margin: 0;
        font-size: 14px;
    }
    .youtube-link {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,0,0,0.9);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        text-decoration: none;
        font-size: 13px;
        margin-top: 10px;
        transition: transform 0.2s;
    }
    .youtube-link:hover {
        transform: scale(1.05);
        color: white;
    }
    .example-buttons {
        display: flex;
        gap: 8px;
        margin-bottom: 10px;
    }
    .section-header {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #374151;
    }
    """
    
    # 顶部横幅
    header_html = """
    <div class="header-banner">
        <h1>🎵 ACE-Step 1.5 音乐生成</h1>
        <p>基于 ACE-Step 1.5 模型的文本到音乐生成系统</p>
        <a href="https://www.youtube.com/@rongyi-ai" target="_blank" class="youtube-link">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="white">
                <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/>
            </svg>
            关注 YouTube 频道 @rongyi-ai
        </a>
    </div>
    """
    
    with gr.Blocks(
        title="ACE-Step 1.5 音乐生成",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:
        
        # 顶部横幅
        gr.HTML(header_html)
        
        # ============ 第一行：快捷操作 ============
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("**💡 快速开始：加载示例或随机生成描述**")
            with gr.Column(scale=1):
                with gr.Row():
                    load_zh_btn = gr.Button("📖 中文示例", variant="secondary", size="sm")
                    load_en_btn = gr.Button("📖 英文示例", variant="secondary", size="sm")
                    random_btn = gr.Button("🎲 随机描述", variant="secondary", size="sm")
        
        gr.Markdown("---")
        
        # ============ 主内容区 ============
        with gr.Row():
            # ========== 左列：输入区 ==========
            with gr.Column(scale=3):
                # 音乐描述
                caption_input = gr.Textbox(
                    label="📝 音乐描述",
                    placeholder="描述你想要生成的音乐风格、乐器、情绪等...\n例如: A cheerful pop song with acoustic guitar and piano...",
                    lines=4,
                )
                
                # 歌词
                lyrics_input = gr.Textbox(
                    label="🎤 歌词",
                    placeholder="输入歌词，使用 [Verse], [Chorus] 等标记段落结构...\n如果是纯音乐，请勾选右侧「纯音乐」选项",
                    lines=8,
                )
                
                # 歌词选项行
                with gr.Row():
                    instrumental_checkbox = gr.Checkbox(
                        label="🎹 纯音乐（无人声）",
                        value=False,
                    )
                    vocal_language = gr.Dropdown(
                        choices=VALID_LANGUAGES,
                        value="unknown",
                        label="人声语言",
                        scale=2,
                    )
                
                # 参考音频（折叠）
                with gr.Accordion("🎧 参考音频（可选）", open=False):
                    reference_audio = gr.Audio(
                        label="上传参考音频",
                        type="filepath",
                    )
            
            # ========== 右列：参数区 ==========
            with gr.Column(scale=2):
                # 基本音乐参数
                gr.Markdown("**🎛️ 音乐参数**")
                with gr.Row():
                    bpm_input = gr.Number(
                        label="BPM",
                        value=None,
                        step=1,
                        info="节拍速度",
                    )
                    duration_input = gr.Number(
                        label="时长（秒）",
                        value=30,
                        minimum=10,
                        maximum=600,
                        step=1,
                    )
                with gr.Row():
                    key_scale_input = gr.Textbox(
                        label="调性",
                        placeholder="如: C major",
                        value="",
                    )
                    time_sig_input = gr.Dropdown(
                        choices=["", "2", "3", "4", "6"],
                        value="",
                        label="拍号",
                    )
                
                gr.Markdown("---")
                
                # 生成参数
                gr.Markdown("**⚡ 生成参数**")
                with gr.Row():
                    steps_input = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=8,
                        step=1,
                        label="推理步数",
                    )
                    guidance_input = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.5,
                        step=0.5,
                        label="引导强度",
                    )
                
                with gr.Row():
                    batch_size_input = gr.Number(
                        label="批量数",
                        value=1,
                        minimum=1,
                        maximum=8,
                        step=1,
                    )
                    seed_input = gr.Number(
                        label="种子 (-1随机)",
                        value=-1,
                        step=1,
                    )
                
                thinking_checkbox = gr.Checkbox(
                    label="🧠 启用 LLM 思维链（智能生成元数据）",
                    value=True,
                )
                
                # 生成按钮
                gr.Markdown("")
                generate_btn = gr.Button(
                    "🎵 生成音乐",
                    variant="primary",
                    size="lg",
                )
        
        gr.Markdown("---")
        
        # ============ 输出区 ============
        gr.Markdown("### 🎶 生成结果")
        with gr.Row():
            with gr.Column(scale=3):
                audio_output = gr.Audio(
                    label="生成的音乐",
                    type="filepath",
                )
            with gr.Column(scale=2):
                status_output = gr.Textbox(
                    label="生成状态",
                    lines=6,
                    interactive=False,
                )
        
        # 事件绑定
        random_btn.click(
            fn=random_caption,
            inputs=[],
            outputs=[caption_input]
        )
        
        # 加载中文示例
        load_zh_btn.click(
            fn=load_chinese_example,
            inputs=[],
            outputs=[caption_input, lyrics_input, vocal_language, bpm_input, key_scale_input, time_sig_input, duration_input]
        )
        
        # 加载英文示例
        load_en_btn.click(
            fn=load_english_example,
            inputs=[],
            outputs=[caption_input, lyrics_input, vocal_language, bpm_input, key_scale_input, time_sig_input, duration_input]
        )
        
        generate_btn.click(
            fn=generate_music,
            inputs=[
                caption_input,
                lyrics_input,
                vocal_language,
                instrumental_checkbox,
                bpm_input,
                key_scale_input,
                time_sig_input,
                duration_input,
                batch_size_input,
                steps_input,
                guidance_input,
                seed_input,
                thinking_checkbox,
                reference_audio,
            ],
            outputs=[audio_output, status_output]
        )
    
    return demo


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ACE-Step 1.5 音乐生成")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="服务地址")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    
    args = parser.parse_args()
    
    # 启动时自动加载模型
    print("\n" + "="*60)
    print("🚀 正在初始化 ACE-Step 1.5 模型...")
    print("="*60 + "\n")
    
    status = initialize_service(
        device="auto",
        use_flash_attention=False,
        offload_to_cpu=False,
        init_llm=True,
    )
    print(status)
    print("\n" + "="*60 + "\n")
    
    demo = create_ui()
    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
