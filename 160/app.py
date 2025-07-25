"""
Gradio WebUI for Higgs Audio Generation
Supports zero-shot voice cloning, single-speaker generation, and multi-speaker dialog
"""

import gradio as gr
import os
import torch
import torchaudio
import tempfile
import soundfile as sf
from pathlib import Path
import re
from typing import List, Optional, Tuple, Dict

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

# Configuration
MODEL_PATH = "checkpoints/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "checkpoints/higgs-audio-v2-tokenizer"
VOICE_PROMPTS_DIR = "examples/voice_prompts"
TRANSCRIPT_DIR = "examples/transcript"

# Initialize the serve engine
device = "cuda" if torch.cuda.is_available() else "cpu"
serve_engine = None

def initialize_model():
    """Initialize the HiggsAudio model"""
    global serve_engine
    if serve_engine is None:
        serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)
    return serve_engine

def get_available_voices():
    """Get list of available voice references"""
    voice_dir = Path(VOICE_PROMPTS_DIR)
    if not voice_dir.exists():
        return []
    
    voices = []
    for wav_file in voice_dir.glob("*.wav"):
        voice_name = wav_file.stem
        voices.append(voice_name)
    
    return sorted(voices)

def get_available_transcripts():
    """Get list of available multi-speaker transcripts"""
    transcript_dir = Path(TRANSCRIPT_DIR) / "multi_speaker"
    if not transcript_dir.exists():
        return []
    
    transcripts = []
    for txt_file in transcript_dir.glob("*.txt"):
        transcripts.append(txt_file.name)
    
    return sorted(transcripts)

def load_transcript_content(transcript_name):
    """Load content from a transcript file"""
    if not transcript_name:
        return ""
    
    transcript_path = Path(TRANSCRIPT_DIR) / "multi_speaker" / transcript_name
    if transcript_path.exists():
        return transcript_path.read_text().strip()
    return ""

def create_system_message(scene_desc="Audio is recorded from a quiet room."):
    """Create system message with scene description"""
    return f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"

def generate_single_speaker_audio(
    transcript: str,
    ref_audio: str,
    scene_desc: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: int
):
    """Generate single-speaker audio with optional voice cloning"""
    
    if not transcript.strip():
        return None, "请提供转录文本"
    
    try:
        # Initialize model if needed
        engine = initialize_model()
        
        # Set seed for reproducibility
        if seed > 0:
            torch.manual_seed(seed)
        
        # Create system message
        system_prompt = create_system_message(scene_desc)
        
        messages = [
            Message(role="system", content=system_prompt)
        ]
        
        # Add reference audio if provided (as conversation history for voice cloning)
        ref_audio_loaded = False
        if ref_audio and ref_audio != "None":
            ref_audio_path = Path(VOICE_PROMPTS_DIR) / f"{ref_audio}.wav"
            ref_text_path = Path(VOICE_PROMPTS_DIR) / f"{ref_audio}.txt"
            
            if ref_audio_path.exists() and ref_text_path.exists():
                # Read the reference text
                with open(ref_text_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
                
                # Add reference audio as user-assistant conversation history for voice cloning
                # This teaches the model what voice to use
                messages.extend([
                    Message(role="user", content=ref_text),
                    Message(role="assistant", content=AudioContent(audio_url=str(ref_audio_path)))
                ])
                ref_audio_loaded = True
            else:
                missing_files = []
                if not ref_audio_path.exists():
                    missing_files.append(f"{ref_audio}.wav")
                if not ref_text_path.exists():
                    missing_files.append(f"{ref_audio}.txt")
                return None, f"缺少参考文件：{', '.join(missing_files)}"
        
        # Add the actual user transcript that we want to generate
        messages.append(Message(role="user", content=transcript.strip()))
        
        # Generate audio
        output: HiggsAudioResponse = engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            torchaudio.save(
                tmp_file.name, 
                torch.from_numpy(output.audio)[None, :], 
                output.sampling_rate
            )
            return tmp_file.name, f"音频生成成功！{' （使用参考语音：' + ref_audio + '）' if ref_audio_loaded else ''}"
            
    except Exception as e:
        return None, f"生成音频时出错：{str(e)}"

def generate_multi_speaker_audio(
    transcript: str,
    speaker1_voice: str,
    speaker2_voice: str,
    scene_desc: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: int,
    use_smart_voice: bool
):
    """Generate multi-speaker dialog audio"""
    
    if not transcript.strip():
        return None, "请提供转录文本"
    
    try:
        # Initialize model if needed
        engine = initialize_model()
        
        # Set seed for reproducibility
        if seed > 0:
            torch.manual_seed(seed)
        
        # Create system message for multi-speaker
        system_prompt = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""
        
        if scene_desc.strip():
            system_prompt += f"\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"
        
        messages = [
            Message(role="system", content=system_prompt)
        ]
        
        # Add reference audio if provided and not using smart voice
        ref_voices_loaded = []
        if not use_smart_voice:
            ref_voices = []
            if speaker1_voice and speaker1_voice != "None":
                ref_voices.append(speaker1_voice)
            if speaker2_voice and speaker2_voice != "None":
                ref_voices.append(speaker2_voice)
            
            for spk_id, voice in enumerate(ref_voices):
                ref_audio_path = Path(VOICE_PROMPTS_DIR) / f"{voice}.wav"
                ref_text_path = Path(VOICE_PROMPTS_DIR) / f"{voice}.txt"
                
                if ref_audio_path.exists() and ref_text_path.exists():
                    # Read the reference text
                    with open(ref_text_path, "r", encoding="utf-8") as f:
                        ref_text = f.read().strip()
                    
                    # Add reference audio as user-assistant conversation history
                    # Use speaker tags for multi-speaker
                    speaker_text = f"[SPEAKER{spk_id}] {ref_text}" if len(ref_voices) > 1 else ref_text
                    messages.extend([
                        Message(role="user", content=speaker_text),
                        Message(role="assistant", content=AudioContent(audio_url=str(ref_audio_path)))
                    ])
                    ref_voices_loaded.append(voice)
                else:
                    missing_files = []
                    if not ref_audio_path.exists():
                        missing_files.append(f"{voice}.wav")
                    if not ref_text_path.exists():
                        missing_files.append(f"{voice}.txt")
                    return None, f"语音 '{voice}' 缺少参考文件：{', '.join(missing_files)}"
        
        # Add the actual user transcript that we want to generate
        messages.append(Message(role="user", content=transcript.strip()))
        
        # Generate audio
        output: HiggsAudioResponse = engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            torchaudio.save(
                tmp_file.name, 
                torch.from_numpy(output.audio)[None, :], 
                output.sampling_rate
            )
            status_msg = "多说话人音频生成成功！"
            if ref_voices_loaded:
                status_msg += f" （使用参考语音：{', '.join(ref_voices_loaded)}）"
            elif use_smart_voice:
                status_msg += " （使用智能语音选择）"
            return tmp_file.name, status_msg
            
    except Exception as e:
        return None, f"生成音频时出错：{str(e)}"

def load_preset_transcript(transcript_name):
    """Load preset transcript content"""
    return load_transcript_content(transcript_name)

def load_reference_audio_preview(voice_name):
    """Load reference audio and text for preview"""
    if not voice_name or voice_name == "None":
        return None, "", gr.update(visible=False)
    
    ref_audio_path = Path(VOICE_PROMPTS_DIR) / f"{voice_name}.wav"
    ref_text_path = Path(VOICE_PROMPTS_DIR) / f"{voice_name}.txt"
    
    audio_file = None
    text_content = ""
    
    if ref_audio_path.exists():
        audio_file = str(ref_audio_path)
    
    if ref_text_path.exists():
        with open(ref_text_path, "r", encoding="utf-8") as f:
            text_content = f.read().strip()
    
    # Show the preview section if we have either audio or text
    show_preview = audio_file is not None or text_content != ""
    
    return audio_file, text_content, gr.update(visible=show_preview)

def load_multi_speaker_reference_preview(speaker1_voice, speaker2_voice, use_smart_voice):
    """Load reference audios for both speakers with preview"""
    if use_smart_voice:
        # Hide all preview sections when using smart voice
        return (
            gr.update(visible=False),  # speaker1_preview
            None, "",                  # speaker1 audio and text
            gr.update(visible=False),  # speaker2_preview  
            None, "",                  # speaker2 audio and text
            ""                         # ref_audio_list for generation
        )
    
    # Process Speaker 1
    speaker1_audio = None
    speaker1_text = ""
    show_speaker1 = False
    
    if speaker1_voice and speaker1_voice != "None":
        ref_audio_path = Path(VOICE_PROMPTS_DIR) / f"{speaker1_voice}.wav"
        ref_text_path = Path(VOICE_PROMPTS_DIR) / f"{speaker1_voice}.txt"
        
        if ref_audio_path.exists():
            speaker1_audio = str(ref_audio_path)
        
        if ref_text_path.exists():
            with open(ref_text_path, "r", encoding="utf-8") as f:
                speaker1_text = f.read().strip()
        
        show_speaker1 = speaker1_audio is not None or speaker1_text != ""
    
    # Process Speaker 2
    speaker2_audio = None
    speaker2_text = ""
    show_speaker2 = False
    
    if speaker2_voice and speaker2_voice != "None":
        ref_audio_path = Path(VOICE_PROMPTS_DIR) / f"{speaker2_voice}.wav"
        ref_text_path = Path(VOICE_PROMPTS_DIR) / f"{speaker2_voice}.txt"
        
        if ref_audio_path.exists():
            speaker2_audio = str(ref_audio_path)
        
        if ref_text_path.exists():
            with open(ref_text_path, "r", encoding="utf-8") as f:
                speaker2_text = f.read().strip()
        
        show_speaker2 = speaker2_audio is not None or speaker2_text != ""
    
    # Create reference audio list for generation function
    ref_voices = []
    if speaker1_voice and speaker1_voice != "None":
        ref_voices.append(speaker1_voice)
    if speaker2_voice and speaker2_voice != "None":
        ref_voices.append(speaker2_voice)
    
    ref_audio_list = ",".join(ref_voices)
    
    return (
        gr.update(visible=show_speaker1),  # speaker1_preview
        speaker1_audio, speaker1_text,     # speaker1 audio and text
        gr.update(visible=show_speaker2),  # speaker2_preview
        speaker2_audio, speaker2_text,     # speaker2 audio and text
        ref_audio_list                     # for generation
    )

def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Get available options
    available_voices = get_available_voices()
    available_transcripts = get_available_transcripts()
    
    with gr.Blocks(title="Higgs Audio Generation WebUI", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# 🎵 Higgs Audio 音频生成 WebUI")
        gr.Markdown("使用零样本语音克隆和多说话人对话功能生成高质量音频。")
        
        with gr.Tabs():
            
            # Single Speaker Tab
            with gr.TabItem("🎤 单说话人生成"):
                gr.Markdown("### 生成带可选语音克隆的单说话人音频")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        single_transcript = gr.Textbox(
                            label="转录文本",
                            placeholder="输入您想要转换为语音的文本...",
                            lines=5,
                            value="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."
                        )
                        
                        single_ref_audio = gr.Dropdown(
                            label="参考语音（可选）",
                            choices=["None"] + available_voices,
                            value="None",
                            info="选择用于克隆的参考语音，或选择 'None' 进行智能语音选择"
                        )
                        
                        # Reference Audio Preview Section
                        with gr.Group(visible=False) as single_ref_preview:
                            gr.Markdown("#### 🎧 参考音频预览")
                            gr.Markdown("*此参考音频将用于克隆语音特征，但您上面输入的转录文本将是实际生成的内容。*")
                            single_ref_audio_player = gr.Audio(
                                label="参考音频", 
                                type="filepath",
                                interactive=False
                            )
                            single_ref_text_display = gr.Textbox(
                                label="参考文本（仅用于语音克隆）",
                                interactive=False,
                                lines=3,
                                info="此文本对应参考音频，仅用于语音学习"
                            )
                        
                        single_scene_desc = gr.Textbox(
                            label="场景描述",
                            value="Audio is recorded from a quiet room.",
                            lines=2,
                            info="描述声学环境"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 生成参数")
                        single_temperature = gr.Slider(0.1, 1.0, value=0.3, label="温度", info="创造性水平")
                        single_top_p = gr.Slider(0.1, 1.0, value=0.95, label="Top P", info="核采样")
                        single_top_k = gr.Slider(1, 100, value=50, label="Top K", info="Top-k采样")
                        single_max_tokens = gr.Slider(128, 2048, value=1024, label="最大令牌数", info="最大生成长度")
                        single_seed = gr.Number(label="随机种子（0为随机）", value=0, precision=0)
                
                single_generate_btn = gr.Button("🎵 生成音频", variant="primary", size="lg")
                
                with gr.Row():
                    single_audio_output = gr.Audio(label="生成的音频", type="filepath")
                    single_status = gr.Textbox(label="状态", interactive=False)
            
            # Multi-Speaker Tab
            with gr.TabItem("👥 多说话人生成"):
                gr.Markdown("### 生成多说话人对话音频")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            multi_preset_transcript = gr.Dropdown(
                                label="预设转录文本",
                                choices=[""] + available_transcripts,
                                value="",
                                info="选择预设转录文本或在下方编写自己的文本"
                            )
                            load_preset_btn = gr.Button("📁 加载预设", size="sm")
                        
                        multi_transcript = gr.Textbox(
                            label="多说话人转录文本",
                            placeholder="使用 [SPEAKER0]、[SPEAKER1] 标签表示不同说话人...\n示例：\n[SPEAKER0] Hello, how are you?\n[SPEAKER1] I'm doing great, thanks!",
                            lines=8,
                            info="使用 [SPEAKER0]、[SPEAKER1] 等标签表示不同说话人"
                        )
                        
                        multi_smart_voice = gr.Checkbox(
                            label="使用智能语音选择",
                            value=False,  # 默认关闭，让用户可以选择声音
                            info="让模型自动选择合适的语音"
                        )
                        
                        # Speaker Voice Selection Section
                        with gr.Group():
                            gr.Markdown("#### 🎭 说话人语音选择")
                            
                            with gr.Row():
                                with gr.Column():
                                    speaker1_voice = gr.Dropdown(
                                        label="说话人0语音",
                                        choices=["None"] + available_voices,
                                        value="None",
                                        info="为 SPEAKER0 选择语音",
                                        interactive=True  # 默认启用
                                    )
                                    
                                    # Speaker 1 Preview
                                    with gr.Group(visible=False) as speaker1_preview:
                                        gr.Markdown("**🎧 说话人0预览**")
                                        speaker1_audio_player = gr.Audio(
                                            label="说话人0参考音频", 
                                            type="filepath",
                                            interactive=False
                                        )
                                        speaker1_text_display = gr.Textbox(
                                            label="说话人0参考文本",
                                            interactive=False,
                                            lines=2
                                        )
                                
                                with gr.Column():
                                    speaker2_voice = gr.Dropdown(
                                        label="说话人1语音",
                                        choices=["None"] + available_voices,
                                        value="None", 
                                        info="为 SPEAKER1 选择语音",
                                        interactive=True  # 默认启用
                                    )
                                    
                                    # Speaker 2 Preview
                                    with gr.Group(visible=False) as speaker2_preview:
                                        gr.Markdown("**🎧 说话人1预览**")
                                        speaker2_audio_player = gr.Audio(
                                            label="说话人1参考音频",
                                            type="filepath", 
                                            interactive=False
                                        )
                                        speaker2_text_display = gr.Textbox(
                                            label="说话人1参考文本",
                                            interactive=False,
                                            lines=2
                                        )
                        
                        # Hidden field to store the combined reference audio list for the generation function
                        multi_ref_audio_combined = gr.Textbox(visible=False, value="")
                        
                        multi_scene_desc = gr.Textbox(
                            label="场景描述",
                            value="Audio is recorded from a quiet room.",
                            lines=2
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 生成参数")
                        multi_temperature = gr.Slider(0.1, 1.0, value=0.3, label="温度")
                        multi_top_p = gr.Slider(0.1, 1.0, value=0.95, label="Top P")
                        multi_top_k = gr.Slider(1, 100, value=50, label="Top K")
                        multi_max_tokens = gr.Slider(128, 2048, value=1024, label="最大令牌数")
                        multi_seed = gr.Number(label="随机种子（0为随机）", value=12345, precision=0)
                
                multi_generate_btn = gr.Button("🎭 生成多说话人音频", variant="primary", size="lg")
                
                with gr.Row():
                    multi_audio_output = gr.Audio(label="生成的音频", type="filepath")
                    multi_status = gr.Textbox(label="状态", interactive=False)
            
            # Help Tab
            with gr.TabItem("ℹ️ 帮助"):
                gr.Markdown("""
                ## 如何使用 Higgs Audio 音频生成 WebUI
                
                ### 单说话人生成
                1. **输入转录文本** - 您想要转换为语音的文本
                2. **选择参考语音**（可选）- 从可用语音中选择用于语音克隆的参考语音
                   - 参考音频提供语音特征
                   - 您的转录文本将用于生成，而不是参考文本
                3. **设置场景描述** - 描述声学环境
                4. **调整参数** - 温度控制创造性，其他参数影响生成质量
                5. **点击生成** - 等待音频生成
                
                ### 多说话人生成
                1. **格式化转录文本** - 使用 `[SPEAKER0]`、`[SPEAKER1]` 标签表示不同说话人
                2. **选择语音** - 使用智能语音选择或指定参考语音
                3. **设置参数** - 与单说话人生成类似
                4. **生成** - 模型将创建多说话人对话
                
                ### 使用技巧
                - **智能语音选择**：让模型根据内容选择合适的语音
                - **语音克隆**：使用参考语音获得一致的角色语音
                - **随机种子**：使用相同种子获得可重现的结果
                - **温度**：较低值（0.1-0.3）获得更一致的语音，较高值（0.5-0.8）获得更多变化
                
                ### 多说话人格式示例
                ```
                [SPEAKER0] I can't believe you did that without even asking me first!
                [SPEAKER1] Oh, come on! It wasn't a big deal, and I knew you would overreact like this.
                [SPEAKER0] Overreact? You made a decision that affects both of us!
                ```
                """)
        
        # Event handlers
        single_generate_btn.click(
            fn=generate_single_speaker_audio,
            inputs=[
                single_transcript, single_ref_audio, single_scene_desc,
                single_temperature, single_top_p, single_top_k, 
                single_max_tokens, single_seed
            ],
            outputs=[single_audio_output, single_status]
        )
        
        multi_generate_btn.click(
            fn=generate_multi_speaker_audio,
            inputs=[
                multi_transcript, speaker1_voice, speaker2_voice, multi_scene_desc,
                multi_temperature, multi_top_p, multi_top_k,
                multi_max_tokens, multi_seed, multi_smart_voice
            ],
            outputs=[multi_audio_output, multi_status]
        )
        
        load_preset_btn.click(
            fn=load_preset_transcript,
            inputs=[multi_preset_transcript],
            outputs=[multi_transcript]
        )
        
        # Reference audio preview for single speaker
        single_ref_audio.change(
            fn=load_reference_audio_preview,
            inputs=[single_ref_audio],
            outputs=[single_ref_audio_player, single_ref_text_display, single_ref_preview]
        )
        
        # Reference audio preview for multi-speaker
        def update_multi_speaker_previews(speaker1, speaker2, smart_voice):
            return load_multi_speaker_reference_preview(speaker1, speaker2, smart_voice)
        
        # Update previews when speaker voices change
        speaker1_voice.change(
            fn=update_multi_speaker_previews,
            inputs=[speaker1_voice, speaker2_voice, multi_smart_voice],
            outputs=[
                speaker1_preview, speaker1_audio_player, speaker1_text_display,
                speaker2_preview, speaker2_audio_player, speaker2_text_display,
                multi_ref_audio_combined
            ]
        )
        
        speaker2_voice.change(
            fn=update_multi_speaker_previews,
            inputs=[speaker1_voice, speaker2_voice, multi_smart_voice],
            outputs=[
                speaker1_preview, speaker1_audio_player, speaker1_text_display,
                speaker2_preview, speaker2_audio_player, speaker2_text_display,
                multi_ref_audio_combined
            ]
        )
        
        # Update previews and enable/disable dropdowns when smart voice changes
        multi_smart_voice.change(
            fn=update_multi_speaker_previews,
            inputs=[speaker1_voice, speaker2_voice, multi_smart_voice],
            outputs=[
                speaker1_preview, speaker1_audio_player, speaker1_text_display,
                speaker2_preview, speaker2_audio_player, speaker2_text_display,
                multi_ref_audio_combined
            ]
        )
        
        # Enable/disable speaker voice dropdowns based on smart voice checkbox
        multi_smart_voice.change(
            fn=lambda smart: (gr.update(interactive=not smart), gr.update(interactive=not smart)),
            inputs=[multi_smart_voice],
            outputs=[speaker1_voice, speaker2_voice]
        )
    
    return demo

if __name__ == "__main__":
    # Check if model files exist
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please ensure the model checkpoints are in the correct location.")
        exit(1)
    
    if not Path(AUDIO_TOKENIZER_PATH).exists():
        print(f"❌ Audio tokenizer not found at {AUDIO_TOKENIZER_PATH}")
        print("Please ensure the tokenizer checkpoints are in the correct location.")
        exit(1)
    
    print(f"🚀 Starting Higgs Audio Generation WebUI...")
    print(f"📱 Device: {device}")
    print(f"🎭 Available voices: {len(get_available_voices())}")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True
    )
