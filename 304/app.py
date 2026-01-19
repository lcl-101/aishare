# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
PersonaPlex Gradio Web Interface
基于 Gradio 的 PersonaPlex 语音对话模型 Web 界面
"""

import os
import sys
import tempfile
import tarfile
from pathlib import Path
from typing import Optional, List, Tuple
import json

import numpy as np
import torch
import sentencepiece
import sphn
import gradio as gr

# 添加 moshi 模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "moshi"))

from moshi.client_utils import make_log
from moshi.models import loaders, LMGen, MimiModel
from moshi.models.lm import load_audio as lm_load_audio
from moshi.models.lm import _iterate_audio as lm_iterate_audio
from moshi.models.lm import encode_from_sphn as lm_encode_from_sphn

# ==================== 全局配置 ====================
CHECKPOINT_DIR = "checkpoints/personaplex-7b-v1"
VOICES_TGZ = os.path.join(CHECKPOINT_DIR, "voices.tgz")
VOICES_DIR = os.path.join(CHECKPOINT_DIR, "voices")
MOSHI_WEIGHT = os.path.join(CHECKPOINT_DIR, "model.safetensors")
MIMI_WEIGHT = os.path.join(CHECKPOINT_DIR, "tokenizer-e351c8d8-checkpoint125.safetensors")
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer_spm_32k_3.model")

# 声音选项
VOICE_OPTIONS = {
    "Natural Female 0 (NATF0)": "NATF0.pt",
    "Natural Female 1 (NATF1)": "NATF1.pt",
    "Natural Female 2 (NATF2)": "NATF2.pt",
    "Natural Female 3 (NATF3)": "NATF3.pt",
    "Natural Male 0 (NATM0)": "NATM0.pt",
    "Natural Male 1 (NATM1)": "NATM1.pt",
    "Natural Male 2 (NATM2)": "NATM2.pt",
    "Natural Male 3 (NATM3)": "NATM3.pt",
    "Variety Female 0 (VARF0)": "VARF0.pt",
    "Variety Female 1 (VARF1)": "VARF1.pt",
    "Variety Female 2 (VARF2)": "VARF2.pt",
    "Variety Female 3 (VARF3)": "VARF3.pt",
    "Variety Female 4 (VARF4)": "VARF4.pt",
    "Variety Male 0 (VARM0)": "VARM0.pt",
    "Variety Male 1 (VARM1)": "VARM1.pt",
    "Variety Male 2 (VARM2)": "VARM2.pt",
    "Variety Male 3 (VARM3)": "VARM3.pt",
    "Variety Male 4 (VARM4)": "VARM4.pt",
}

# 官方示例配置
OFFICIAL_EXAMPLES = {
    "不使用官方示例": {
        "input_wav": None,
        "voice": "Natural Male 1 (NATM1)",
        "text_prompt": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    },
    "官方示例 1: Assistant (助手角色)": {
        "input_wav": "assets/test/input_assistant.wav",
        "voice": "Natural Female 2 (NATF2)",
        "text_prompt": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    },
    "官方示例 2: Service (客服角色)": {
        "input_wav": "assets/test/input_service.wav",
        "voice": "Natural Male 1 (NATM1)",
        "text_prompt": "You work for SwiftPlex Appliances which is a appliance repair company and your name is Farhod Toshmatov. Information: The dishwasher model is out of stock for replacement parts; we can use an alternative part with a 3-day delay. Labor cost remains $60 per hour.",
    },
}

# 示例提示词（保持英文原样）
EXAMPLE_PROMPTS = {
    "助手角色 (Assistant)": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    "客服角色 - 废物管理 (Waste Management)": "You work for CitySan Services which is a waste management and your name is Ayelen Lucero. Information: Verify customer name Omar Torres. Current schedule: every other week. Upcoming pickup: April 12th. Compost bin service available for $8/month add-on.",
    "客服角色 - 餐厅 (Restaurant)": "You work for Jerusalem Shakshuka which is a restaurant and your name is Owen Foster. Information: There are two shakshuka options: Classic (poached eggs, $9.50) and Spicy (scrambled eggs with jalapenos, $10.25). Sides include warm pita ($2.50) and Israeli salad ($3). No combo offers. Available for drive-through until 9 PM.",
    "客服角色 - 无人机租赁 (Drone Rental)": "You work for AeroRentals Pro which is a drone rental company and your name is Tomaz Novak. Information: AeroRentals Pro has the following availability: PhoenixDrone X ($65/4 hours, $110/8 hours), and the premium SpectraDrone 9 ($95/4 hours, $160/8 hours). Deposit required: $150 for standard models, $300 for premium.",
    "客服角色 - 家电维修 (Appliance Repair)": "You work for SwiftPlex Appliances which is a appliance repair company and your name is Farhod Toshmatov. Information: The dishwasher model is out of stock for replacement parts; we can use an alternative part with a 3-day delay. Labor cost remains $60 per hour.",
    "休闲对话 - 基础 (Casual Basic)": "You enjoy having a good conversation.",
    "休闲对话 - 饮食话题 (Dining Topic)": "You enjoy having a good conversation. Have a casual discussion about eating at home versus dining out.",
    "休闲对话 - 家庭话题 (Family Topic)": "You enjoy having a good conversation. Have an empathetic discussion about the meaning of family amid uncertainty.",
    "休闲对话 - 职业话题 (Career Topic)": "You enjoy having a good conversation. Have a reflective conversation about career changes and feeling of home. You have lived in California for 21 years and consider San Francisco your home. You work as a teacher and have traveled a lot. You dislike meetings.",
    "休闲对话 - 美食话题 (Food Topic)": "You enjoy having a good conversation. Have a casual conversation about favorite foods and cooking experiences. You are David Green, a former baker now living in Boston. You enjoy cooking diverse international dishes and appreciate many ethnic restaurants.",
}

# 全局模型状态
model_state = {
    "loaded": False,
    "mimi": None,
    "other_mimi": None,
    "lm_gen": None,
    "text_tokenizer": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "frame_size": None,
}


def log(level: str, msg: str):
    """日志输出函数"""
    print(make_log(level, msg))


def seed_all(seed: int):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """添加系统标签"""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def extract_voices():
    """解压语音嵌入文件"""
    if not os.path.exists(VOICES_DIR):
        if os.path.exists(VOICES_TGZ):
            log("info", f"正在解压语音文件: {VOICES_TGZ}")
            with tarfile.open(VOICES_TGZ, "r:gz") as tar:
                tar.extractall(path=CHECKPOINT_DIR)
            log("info", "语音文件解压完成")
        else:
            raise FileNotFoundError(f"找不到语音文件: {VOICES_TGZ}")


def warmup(mimi: MimiModel, other_mimi: MimiModel, lm_gen: LMGen, device: str, frame_size: int):
    """模型预热"""
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c : c + 1])
            if tokens is None:
                continue
            _ = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def decode_tokens_to_pcm(mimi: MimiModel, other_mimi: MimiModel, lm_gen: LMGen, tokens: torch.Tensor) -> np.ndarray:
    """将模型输出的 tokens 解码为 PCM 音频"""
    pcm = mimi.decode(tokens[:, 1:9])
    _ = other_mimi.decode(tokens[:, 1:9])
    pcm = pcm.detach().cpu().numpy()[0, 0]
    return pcm


def load_models():
    """加载模型（启动时调用）"""
    global model_state
    
    if model_state["loaded"]:
        log("info", "模型已加载完成！")
        return
    
    try:
        device = model_state["device"]
        
        # 解压语音文件
        log("info", "解压语音文件...")
        extract_voices()
        
        # 加载 Mimi 编码器/解码器
        log("info", "正在加载 Mimi...")
        mimi = loaders.get_mimi(MIMI_WEIGHT, device)
        other_mimi = loaders.get_mimi(MIMI_WEIGHT, device)
        log("info", "Mimi 加载完成")
        
        # 加载分词器
        log("info", "加载分词器...")
        text_tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)
        
        # 加载 Moshi LM
        log("info", "正在加载 Moshi LM...")
        lm = loaders.get_moshi_lm(MOSHI_WEIGHT, device=device)
        lm.eval()
        log("info", "Moshi LM 加载完成")
        
        # 构建 LMGen
        log("info", "初始化推理引擎...")
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
            sample_rate=mimi.sample_rate,
            device=device,
            frame_rate=mimi.frame_rate,
            save_voice_prompt_embeddings=False,
            use_sampling=True,
            temp=0.8,
            temp_text=0.7,
            top_k=250,
            top_k_text=25,
        )
        
        # 设置流式模式
        mimi.streaming_forever(1)
        other_mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)
        
        # 预热（需要在 no_grad 上下文中执行）
        log("info", "正在预热模型...")
        with torch.no_grad():
            warmup(mimi, other_mimi, lm_gen, device, frame_size)
        
        # 保存状态
        model_state["mimi"] = mimi
        model_state["other_mimi"] = other_mimi
        model_state["lm_gen"] = lm_gen
        model_state["text_tokenizer"] = text_tokenizer
        model_state["frame_size"] = frame_size
        model_state["loaded"] = True
        
        log("info", "✅ 模型加载成功！可以开始对话了。")
        
    except Exception as e:
        log("error", f"模型加载失败: {str(e)}")
        raise e


def run_inference(
    input_audio: Optional[str],
    voice_name: str,
    text_prompt: str,
    seed: int,
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    use_sampling: bool,
    progress=gr.Progress(),
) -> Tuple[Optional[str], str, str]:
    """运行推理"""
    global model_state
    
    if not model_state["loaded"]:
        return None, "", "❌ 请先加载模型！"
    
    if input_audio is None:
        return None, "", "❌ 请上传或录制音频！"
    
    try:
        mimi = model_state["mimi"]
        other_mimi = model_state["other_mimi"]
        lm_gen = model_state["lm_gen"]
        text_tokenizer = model_state["text_tokenizer"]
        device = model_state["device"]
        
        # 设置随机种子
        if seed != -1:
            seed_all(seed)
        
        # 更新采样参数
        progress(0.1, desc="配置参数...")
        lm_gen._use_sampling = use_sampling
        lm_gen._temp = temp_audio
        lm_gen._temp_text = temp_text
        lm_gen._top_k = topk_audio
        lm_gen._top_k_text = topk_text
        
        # 加载语音嵌入
        progress(0.2, desc="加载语音嵌入...")
        voice_file = VOICE_OPTIONS.get(voice_name, "NATM1.pt")
        voice_prompt_path = os.path.join(VOICES_DIR, voice_file)
        
        if not os.path.exists(voice_prompt_path):
            return None, "", f"❌ 找不到语音文件: {voice_prompt_path}"
        
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
        
        # 设置文本提示
        lm_gen.text_prompt_tokens = (
            text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
        )
        
        # 重置流式状态
        progress(0.3, desc="初始化推理状态...")
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()
        
        # 加载用户音频
        progress(0.4, desc="处理输入音频...")
        sample_rate = mimi.sample_rate
        user_audio = lm_load_audio(input_audio, sample_rate)
        total_target_samples = user_audio.shape[-1]
        
        # 推理
        progress(0.5, desc="生成回复中...")
        generated_frames: List[np.ndarray] = []
        generated_text_tokens: List[str] = []
        
        audio_iterator = lm_encode_from_sphn(
            mimi,
            lm_iterate_audio(
                user_audio, sample_interval_size=lm_gen._frame_size, pad=True
            ),
            max_batch=1,
        )
        
        total_steps = int(np.ceil(total_target_samples / lm_gen._frame_size))
        current_step = 0
        
        for user_encoded in audio_iterator:
            steps = user_encoded.shape[-1]
            for c in range(steps):
                step_in = user_encoded[:, :, c : c + 1]
                tokens = lm_gen.step(step_in)
                if tokens is None:
                    continue
                
                pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
                generated_frames.append(pcm)
                
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    _text = text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")
                    generated_text_tokens.append(_text)
                else:
                    text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
                    generated_text_tokens.append(text_token_map[text_token])
                
                current_step += 1
                if current_step % 10 == 0:
                    prog = 0.5 + 0.4 * (current_step / max(total_steps, 1))
                    progress(prog, desc=f"生成中... ({current_step}/{total_steps})")
        
        if len(generated_frames) == 0:
            return None, "", "❌ 没有生成任何音频帧，请检查输入。"
        
        # 处理输出
        progress(0.95, desc="处理输出...")
        output_pcm = np.concatenate(generated_frames, axis=-1)
        if output_pcm.shape[-1] > total_target_samples:
            output_pcm = output_pcm[:total_target_samples]
        elif output_pcm.shape[-1] < total_target_samples:
            pad_len = total_target_samples - output_pcm.shape[-1]
            output_pcm = np.concatenate(
                [output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1
            )
        
        # 保存输出音频
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        sphn.write_wav(output_path, output_pcm, int(sample_rate))
        
        # 处理文本输出
        filtered_tokens = [t for t in generated_text_tokens if t not in ['EPAD', 'BOS', 'EOS', 'PAD']]
        output_text = "".join(filtered_tokens).strip()
        
        progress(1.0, desc="完成！")
        return output_path, output_text, "✅ 推理完成！"
        
    except Exception as e:
        log("error", f"推理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, "", f"❌ 推理失败: {str(e)}"


def update_prompt(example_name: str) -> str:
    """更新提示词"""
    return EXAMPLE_PROMPTS.get(example_name, "")


def update_official_example(example_name: str):
    """更新官方示例选择"""
    example = OFFICIAL_EXAMPLES.get(example_name, OFFICIAL_EXAMPLES["不使用官方示例"])
    input_wav = example["input_wav"]
    voice = example["voice"]
    text_prompt = example["text_prompt"]
    
    # 查找对应的示例提示词键
    prompt_key = None
    for key, value in EXAMPLE_PROMPTS.items():
        if value == text_prompt:
            prompt_key = key
            break
    if prompt_key is None:
        prompt_key = "助手角色 (Assistant)"
    
    return input_wav, voice, prompt_key, text_prompt


# ==================== Gradio 界面 ====================
def create_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(
        title="PersonaPlex - 全双工语音对话",
        theme=gr.themes.Soft(),
        css="""
        .youtube-banner {
            background: linear-gradient(90deg, #FF0000, #CC0000);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        .youtube-banner a {
            color: white !important;
            text-decoration: none;
            font-weight: bold;
        }
        .youtube-banner a:hover {
            text-decoration: underline;
        }
        """
    ) as demo:
        
        # YouTube 频道信息
        gr.HTML("""
        <div class="youtube-banner">
            <h3 style="margin: 0;">🎬 欢迎访问 AI 技术分享频道</h3>
            <p style="margin: 5px 0 0 0;">
                <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank">
                    📺 YouTube: @rongyikanshijie-ai
                </a>
                &nbsp;|&nbsp;
                更多 AI 技术教程和分享，敬请订阅！
            </p>
        </div>
        """)
        
        gr.Markdown("""
        # 🎙️ PersonaPlex - 全双工语音对话系统
        
        PersonaPlex 是一个支持全双工语音对话的 AI 模型，可以实现自然流畅的语音交互。
        """)
        
        # 官方示例选择
        gr.Markdown("### 🎯 官方示例（快速体验）")
        with gr.Row():
            official_example_dropdown = gr.Dropdown(
                choices=list(OFFICIAL_EXAMPLES.keys()),
                value="不使用官方示例",
                label="选择官方示例",
                info="选择官方提供的示例，将自动填充音频、声音和提示词",
                scale=2
            )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 输入设置")
                
                input_audio = gr.Audio(
                    label="输入音频（上传或录制）",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                voice_dropdown = gr.Dropdown(
                    choices=list(VOICE_OPTIONS.keys()),
                    value="Natural Male 1 (NATM1)",
                    label="选择声音",
                    info="选择模型使用的声音类型"
                )
                
                example_dropdown = gr.Dropdown(
                    choices=list(EXAMPLE_PROMPTS.keys()),
                    value="助手角色 (Assistant)",
                    label="示例提示词",
                    info="选择预设的提示词示例"
                )
                
                text_prompt = gr.Textbox(
                    label="文本提示词",
                    value=EXAMPLE_PROMPTS["助手角色 (Assistant)"],
                    lines=4,
                    placeholder="输入角色设定和对话背景...",
                    info="定义 AI 的角色和行为"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 高级参数")
                
                with gr.Row():
                    seed = gr.Number(
                        label="随机种子",
                        value=42424242,
                        precision=0,
                        info="-1 表示随机"
                    )
                    use_sampling = gr.Checkbox(
                        label="启用采样",
                        value=True,
                        info="关闭则使用贪婪解码"
                    )
                
                with gr.Row():
                    temp_audio = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="音频温度",
                        info="控制音频生成的随机性"
                    )
                    temp_text = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="文本温度",
                        info="控制文本生成的随机性"
                    )
                
                with gr.Row():
                    topk_audio = gr.Slider(
                        minimum=1,
                        maximum=500,
                        value=250,
                        step=1,
                        label="音频 Top-K",
                        info="音频采样的 Top-K 值"
                    )
                    topk_text = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=25,
                        step=1,
                        label="文本 Top-K",
                        info="文本采样的 Top-K 值"
                    )
        
        gr.Markdown("---")
        
        with gr.Row():
            run_btn = gr.Button("🎯 开始推理", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔊 输出音频")
                output_audio = gr.Audio(
                    label="生成的音频",
                    type="filepath"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📝 输出文本")
                output_text = gr.Textbox(
                    label="生成的文本",
                    lines=5,
                    interactive=False
                )
        
        inference_status = gr.Textbox(
            label="推理状态",
            value="",
            interactive=False
        )
        
        gr.Markdown("---")
        
        gr.Markdown("""
        ### 📖 使用说明
        
        1. **快速体验**：选择「官方示例」可以快速体验模型效果
        2. **上传音频**：上传一段音频文件或使用麦克风录制
        3. **选择声音**：从预设的声音中选择一个
        4. **设置提示词**：选择示例提示词或自定义角色设定
        5. **调整参数**：根据需要调整采样参数
        6. **开始推理**：点击「开始推理」按钮，等待生成结果
        
        ### 🎯 官方示例说明
        
        - **Assistant (助手角色)**：模拟一个友好的老师，回答问题或提供建议
        - **Service (客服角色)**：模拟家电维修公司的客服人员
        
        ### 🎭 声音类型说明
        
        - **Natural (NAT)**: 更自然、更对话化的声音
        - **Variety (VAR)**: 更多样化的声音风格
        - **Female (F)**: 女性声音
        - **Male (M)**: 男性声音
        
        ### 💡 提示词建议
        
        - **助手角色**: 适用于问答和建议类对话
        - **客服角色**: 适用于模拟客户服务场景
        - **休闲对话**: 适用于开放式的日常对话
        """)
        
        # 事件绑定
        official_example_dropdown.change(
            fn=update_official_example,
            inputs=[official_example_dropdown],
            outputs=[input_audio, voice_dropdown, example_dropdown, text_prompt]
        )
        
        example_dropdown.change(
            fn=update_prompt,
            inputs=[example_dropdown],
            outputs=[text_prompt]
        )
        
        run_btn.click(
            fn=run_inference,
            inputs=[
                input_audio,
                voice_dropdown,
                text_prompt,
                seed,
                temp_audio,
                temp_text,
                topk_audio,
                topk_text,
                use_sampling
            ],
            outputs=[output_audio, output_text, inference_status]
        )
    
    return demo


if __name__ == "__main__":
    # 启动时加载模型
    print("正在加载模型，请稍候...")
    load_models()
    print("模型加载完成，启动 Web 服务...")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
