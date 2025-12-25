# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fun-Audio-Chat Gradio Web Demo
A simple Gradio-based web application for Fun-Audio-Chat model inference.
Supports both Speech-to-Text (S2T) and Speech-to-Speech (S2S) modes.
"""

import os
import sys
import uuid
import json
import torch
import librosa
import torchaudio
import gradio as gr
import numpy as np
from loguru import logger

# Register Fun-Audio-Chat model
from funaudiochat.register import register_funaudiochat
register_funaudiochat()

from transformers import AutoConfig, AutoProcessor, AutoModelForSeq2SeqLM

from utils.cosyvoice_detokenizer import get_audio_detokenizer, token2wav
from utils.constant import (
    DEFAULT_S2M_GEN_KWARGS,
    DEFAULT_SP_GEN_KWARGS,
    DEFAULT_S2T_PROMPT,
    SPOKEN_S2M_PROMPT,
    AUDIO_TEMPLATE,
)

# ============= Configuration =============
MODEL_PATH = "checkpoints/Fun-Audio-Chat-8B"
TTS_MODEL_PATH = "checkpoints/Fun-CosyVoice3-0.5B-2512"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000  # CosyVoice output sample rate

# ============= Example Audio Files =============
# You can add more example audio files here
EXAMPLE_AUDIO_DIR = "examples"
EXAMPLE_AUDIOS = [
    "examples/ck7vv9ag.wav",  # Default example from the repo
]

# ============= Preset System Prompts =============
PRESET_PROMPTS = {
    "default": ("默认对话", DEFAULT_S2T_PROMPT),
    "transcribe": ("语音转写 (ASR)", "Please transcribe the audio content accurately."),
    "translate_en": ("翻译成英文", "Please translate the audio content into English."),
    "translate_zh": ("翻译成中文", "Please translate the audio content into Chinese."),
    "summarize": ("内容总结", "Please summarize the main points of the audio content."),
    "qa": ("问答助手", "You are a helpful assistant. Please answer the question in the audio."),
}

# ============= Knowledge Base Configuration =============
# Simple in-memory knowledge base (can be replaced with vector DB like FAISS, Milvus, etc.)
KNOWLEDGE_BASE = {}

def load_knowledge_base(file_path: str) -> dict:
    """Load knowledge base from a text file. Each line is a knowledge entry."""
    kb = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    kb[f"doc_{i}"] = line
    return kb

def simple_search(query: str, knowledge_base: dict, top_k: int = 3) -> list:
    """
    Simple search in knowledge base with Chinese support.
    For production, use vector similarity search (FAISS, Milvus, etc.)
    """
    if not knowledge_base:
        return []
    
    results = []
    query_lower = query.lower()
    
    # For Chinese: use character-level matching
    # Remove common punctuation and whitespace
    import re
    query_chars = set(re.sub(r'[，。？！、\s\?\!\.\,]', '', query_lower))
    
    for doc_id, content in knowledge_base.items():
        content_lower = content.lower()
        
        # Score 1: Check if any query character appears in content
        char_score = sum(1 for char in query_chars if char in content_lower)
        
        # Score 2: Check for substring match (important for Chinese)
        # Extract key phrases (2-4 character combinations)
        substring_score = 0
        for i in range(len(query_lower)):
            for length in [2, 3, 4]:
                if i + length <= len(query_lower):
                    phrase = query_lower[i:i+length]
                    # Skip if phrase is all punctuation
                    if re.match(r'^[，。？！、\s\?\!\.\,]+$', phrase):
                        continue
                    if phrase in content_lower:
                        substring_score += length  # Longer matches score higher
        
        total_score = char_score + substring_score * 2  # Weight substring matches higher
        
        if total_score > 0:
            results.append((total_score, content))
    
    # Sort by score and return top_k
    results.sort(key=lambda x: x[0], reverse=True)
    return [content for score, content in results[:top_k]]

RAG_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question based on the following knowledge base content.

## Knowledge Base:
{knowledge_content}

## Instructions:
- Answer based on the knowledge provided above
- If the knowledge doesn't contain relevant information, say you don't know
- Be concise and accurate
- Respond in the same language as the user's question"""

# ============= Global Model Variables =============
model = None
processor = None
cosyvoice_model = None

def load_models():
    """Load Fun-Audio-Chat model and CosyVoice TTS model"""
    global model, processor, cosyvoice_model
    
    logger.info(f"Loading Fun-Audio-Chat model from {MODEL_PATH}...")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH, 
        config=config, 
        torch_dtype=torch.bfloat16, 
        device_map=DEVICE
    )
    logger.info("Fun-Audio-Chat model loaded successfully!")
    
    # Modify CosyVoice model path in cosyvoice_detokenizer
    logger.info(f"Loading CosyVoice TTS model from {TTS_MODEL_PATH}...")
    
    # Import CosyVoice modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    submodule_path = os.path.join(current_dir, 'third_party/CosyVoice')
    sys.path.insert(0, submodule_path)
    matcha_tts_path = os.path.join(current_dir, 'third_party/CosyVoice/third_party/Matcha-TTS')
    sys.path.insert(0, matcha_tts_path)
    
    from cosyvoice.cli.cosyvoice import CosyVoice3
    
    cosyvoice_model = CosyVoice3(
        TTS_MODEL_PATH,
        load_trt=False, 
        load_vllm=False, 
        fp16=False
    )
    cosyvoice_model.model.flow.decoder.estimator.static_chunk_size = 2 * 25 * 30
    logger.info("CosyVoice TTS model loaded successfully!")
    
    return "Models loaded successfully!"

def process_audio_input(audio_path):
    """Process audio input and return audio array"""
    if audio_path is None:
        return None
    
    # Load audio with librosa (resample to 16kHz)
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    return audio

def speech_to_text(audio_path, system_prompt=None):
    """
    Speech-to-Text inference: Generate text response from audio input.
    
    Args:
        audio_path: Path to input audio file
        system_prompt: Optional custom system prompt
    
    Returns:
        Generated text response
    """
    global model, processor
    
    if model is None or processor is None:
        return "Error: Models not loaded. Please click 'Load Models' first."
    
    if audio_path is None:
        return "Error: Please upload or record an audio file."
    
    try:
        # Load audio
        audio = [process_audio_input(audio_path)]
        
        # Set generation parameters for text-only mode
        model.sp_gen_kwargs.update({
            'text_greedy': True, 
            'disable_speech': True,
        })
        
        # Build conversation
        prompt = system_prompt if system_prompt else DEFAULT_S2T_PROMPT
        conversation = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": AUDIO_TEMPLATE},
        ]
        
        # Process input
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
        
        # Generate
        generate_ids, _ = model.generate(**inputs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        generate_text = processor.decode(generate_ids[0], skip_special_tokens=True)
        
        return generate_text
        
    except Exception as e:
        logger.error(f"Error in speech_to_text: {e}")
        return f"Error: {str(e)}"

def speech_to_speech(audio_path, system_prompt=None):
    """
    Speech-to-Speech inference: Generate both text and audio response from audio input.
    
    Args:
        audio_path: Path to input audio file
        system_prompt: Optional custom system prompt
    
    Returns:
        Tuple of (generated_text, audio_output_path)
    """
    global model, processor, cosyvoice_model
    
    if model is None or processor is None:
        return "Error: Models not loaded. Please click 'Load Models' first.", None
    
    if cosyvoice_model is None:
        return "Error: TTS model not loaded.", None
    
    if audio_path is None:
        return "Error: Please upload or record an audio file.", None
    
    try:
        # Load audio
        audio = [process_audio_input(audio_path)]
        
        # Set generation parameters for speech+text mode
        sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
        sp_gen_kwargs['text_greedy'] = True
        gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
        gen_kwargs['max_new_tokens'] = 2048
        model.sp_gen_kwargs.update(sp_gen_kwargs)
        
        # Build conversation
        prompt = system_prompt if system_prompt else SPOKEN_S2M_PROMPT
        conversation = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": AUDIO_TEMPLATE},
        ]
        
        # Process input
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
        
        # Generate text and audio tokens
        generate_ids, audio_ids = model.generate(**inputs, **gen_kwargs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        generate_text = processor.decode(generate_ids[0], skip_special_tokens=True)
        
        # Filter valid audio tokens (0-6560)
        token_for_cosyvoice = list(filter(lambda x: 0 <= x < 6561, audio_ids[0].tolist()))
        
        if len(token_for_cosyvoice) == 0:
            return generate_text, None
        
        # Load speaker embedding (default Chinese female voice)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        spk_emb_path = os.path.join(current_dir, "utils/new_spk2info.pt")
        embedding = torch.load(spk_emb_path)["中文女"]["embedding"]
        
        # Convert audio tokens to waveform
        logger.info(f"Converting {len(token_for_cosyvoice)} audio tokens to waveform...")
        speech = token2wav(
            cosyvoice_model, 
            token_for_cosyvoice, 
            embedding=embedding, 
            token_hop_len=25 * 30, 
            pre_lookahead_len=3
        )
        
        # Save output audio
        output_uuid = str(uuid.uuid4())[:8]
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"output_{output_uuid}.wav")
        
        torchaudio.save(output_path, speech.cpu(), cosyvoice_model.sample_rate)
        logger.info(f"Audio saved to: {output_path}")
        
        return generate_text, output_path
        
    except Exception as e:
        logger.error(f"Error in speech_to_speech: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None

def update_prompt_from_preset(preset_key):
    """Update system prompt based on preset selection"""
    if preset_key and preset_key in PRESET_PROMPTS:
        return PRESET_PROMPTS[preset_key][1]
    return ""

def rag_speech_to_text(audio_path, knowledge_text=None):
    """
    RAG-based Speech-to-Text: First transcribe, then search knowledge base, then answer.
    
    Args:
        audio_path: Path to input audio file
        knowledge_text: Knowledge base content (one entry per line)
    
    Returns:
        Tuple of (transcribed_question, retrieved_knowledge, generated_answer)
    """
    global model, processor
    
    if model is None or processor is None:
        return "Error: Models not loaded.", "", ""
    
    if audio_path is None:
        return "Error: Please upload or record an audio file.", "", ""
    
    try:
        # Step 1: Transcribe audio to get the question
        audio = [process_audio_input(audio_path)]
        
        model.sp_gen_kwargs.update({
            'text_greedy': True, 
            'disable_speech': True,
        })
        
        # First pass: transcribe the question
        transcribe_prompt = "Please transcribe the audio content accurately."
        conversation = [
            {"role": "system", "content": transcribe_prompt},
            {"role": "user", "content": AUDIO_TEMPLATE},
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
        generate_ids, _ = model.generate(**inputs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        question = processor.decode(generate_ids[0], skip_special_tokens=True)
        
        logger.info(f"Transcribed question: {question}")
        
        # Step 2: Search knowledge base
        if knowledge_text and knowledge_text.strip():
            # Build knowledge base from input text
            kb = {}
            for i, line in enumerate(knowledge_text.strip().split('\n')):
                line = line.strip()
                if line:
                    kb[f"doc_{i}"] = line
            
            # Search for relevant knowledge
            retrieved_docs = simple_search(question, kb, top_k=3)
            knowledge_content = "\n".join([f"- {doc}" for doc in retrieved_docs]) if retrieved_docs else "No relevant knowledge found."
        else:
            knowledge_content = "No knowledge base provided."
            retrieved_docs = []
        
        logger.info(f"Retrieved knowledge: {knowledge_content}")
        
        # Step 3: Generate answer based on knowledge
        rag_prompt = RAG_SYSTEM_PROMPT_TEMPLATE.format(knowledge_content=knowledge_content)
        
        conversation = [
            {"role": "system", "content": rag_prompt},
            {"role": "user", "content": AUDIO_TEMPLATE},
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
        generate_ids, _ = model.generate(**inputs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        answer = processor.decode(generate_ids[0], skip_special_tokens=True)
        
        return question, knowledge_content, answer
        
    except Exception as e:
        logger.error(f"Error in rag_speech_to_text: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", "", ""

def rag_speech_to_speech(audio_path, knowledge_text=None):
    """
    RAG-based Speech-to-Speech: Answer with both text and audio based on knowledge base.
    
    Args:
        audio_path: Path to input audio file
        knowledge_text: Knowledge base content (one entry per line)
    
    Returns:
        Tuple of (transcribed_question, retrieved_knowledge, generated_answer, audio_output_path)
    """
    global model, processor, cosyvoice_model
    
    if model is None or processor is None:
        return "Error: Models not loaded.", "", "", None
    
    if cosyvoice_model is None:
        return "Error: TTS model not loaded.", "", "", None
    
    if audio_path is None:
        return "Error: Please upload or record an audio file.", "", "", None
    
    try:
        # Step 1: Transcribe audio to get the question
        audio = [process_audio_input(audio_path)]
        
        model.sp_gen_kwargs.update({
            'text_greedy': True, 
            'disable_speech': True,
        })
        
        transcribe_prompt = "Please transcribe the audio content accurately."
        conversation = [
            {"role": "system", "content": transcribe_prompt},
            {"role": "user", "content": AUDIO_TEMPLATE},
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
        generate_ids, _ = model.generate(**inputs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        question = processor.decode(generate_ids[0], skip_special_tokens=True)
        
        logger.info(f"Transcribed question: {question}")
        
        # Step 2: Search knowledge base
        if knowledge_text and knowledge_text.strip():
            kb = {}
            for i, line in enumerate(knowledge_text.strip().split('\n')):
                line = line.strip()
                if line:
                    kb[f"doc_{i}"] = line
            
            retrieved_docs = simple_search(question, kb, top_k=3)
            knowledge_content = "\n".join([f"- {doc}" for doc in retrieved_docs]) if retrieved_docs else "No relevant knowledge found."
        else:
            knowledge_content = "No knowledge base provided."
            retrieved_docs = []
        
        logger.info(f"Retrieved knowledge: {knowledge_content}")
        
        # Step 3: Generate answer with speech
        rag_prompt = RAG_SYSTEM_PROMPT_TEMPLATE.format(knowledge_content=knowledge_content)
        # Add speech generation instruction
        rag_prompt_with_speech = rag_prompt + "\n\nYou are asked to generate both text and speech tokens at the same time."
        
        sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
        sp_gen_kwargs['text_greedy'] = True
        gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
        gen_kwargs['max_new_tokens'] = 2048
        model.sp_gen_kwargs.update(sp_gen_kwargs)
        
        conversation = [
            {"role": "system", "content": rag_prompt_with_speech},
            {"role": "user", "content": AUDIO_TEMPLATE},
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
        generate_ids, audio_ids = model.generate(**inputs, **gen_kwargs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        answer = processor.decode(generate_ids[0], skip_special_tokens=True)
        
        # Convert audio tokens to speech
        token_for_cosyvoice = list(filter(lambda x: 0 <= x < 6561, audio_ids[0].tolist()))
        
        output_path = None
        if len(token_for_cosyvoice) > 0:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            spk_emb_path = os.path.join(current_dir, "utils/new_spk2info.pt")
            embedding = torch.load(spk_emb_path)["中文女"]["embedding"]
            
            logger.info(f"Converting {len(token_for_cosyvoice)} audio tokens to waveform...")
            speech = token2wav(
                cosyvoice_model, 
                token_for_cosyvoice, 
                embedding=embedding, 
                token_hop_len=25 * 30, 
                pre_lookahead_len=3
            )
            
            output_uuid = str(uuid.uuid4())[:8]
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"rag_output_{output_uuid}.wav")
            
            torchaudio.save(output_path, speech.cpu(), cosyvoice_model.sample_rate)
            logger.info(f"Audio saved to: {output_path}")
        
        return question, knowledge_content, answer, output_path
        
    except Exception as e:
        logger.error(f"Error in rag_speech_to_speech: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", "", "", None

def create_gradio_interface():
    """Create and return the Gradio interface"""
    
    with gr.Blocks(
        title="Fun-Audio-Chat Demo"
    ) as demo:
        
        gr.Markdown(
            """
            # 🎙️ Fun-Audio-Chat Demo
            
            **Fun-Audio-Chat** 是一个大型音频语言模型，支持自然、低延迟的语音交互。
            
            | 功能 | 说明 |
            |------|------|
            | 📝 **Speech-to-Text** | 语音转文字 - 上传音频获取文字回复 |
            | 🔊 **Speech-to-Speech** | 语音转语音 - 上传音频获取语音+文字回复 |
            | � **Knowledge Base QA** | 知识库问答 - 基于知识库的语音问答 |
            
            ---
            """
        )
        
        # Model status display
        model_status = "✅ Models loaded / 模型已加载" if model is not None else "⏳ Loading... / 加载中..."
        gr.Markdown(f"**模型状态:** {model_status}")
        
        gr.Markdown("---")
        
        # Main interface with tabs
        with gr.Tabs():
            # Speech-to-Text Tab
            with gr.TabItem("📝 Speech-to-Text (语音转文字)"):
                gr.Markdown("上传或录制音频，获取文字回复。支持多种任务：对话、转写、翻译、总结等。")
                
                with gr.Row():
                    with gr.Column():
                        s2t_audio_input = gr.Audio(
                            label="Input Audio / 输入音频",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        s2t_preset = gr.Dropdown(
                            label="Preset Prompt / 预设提示词",
                            choices=[(v[0], k) for k, v in PRESET_PROMPTS.items()],
                            value="default",
                            interactive=True
                        )
                        s2t_system_prompt = gr.Textbox(
                            label="System Prompt / 系统提示词 (可自定义)",
                            placeholder="选择预设或自定义提示词...",
                            value=DEFAULT_S2T_PROMPT,
                            lines=3
                        )
                        s2t_btn = gr.Button("🎯 Generate / 生成", variant="primary")
                    
                    with gr.Column():
                        s2t_output = gr.Textbox(
                            label="Generated Text / 生成文本",
                            lines=10
                        )
                
                # Connect preset dropdown to prompt textbox
                s2t_preset.change(
                    fn=update_prompt_from_preset,
                    inputs=[s2t_preset],
                    outputs=[s2t_system_prompt]
                )
                
                # Example section
                gr.Markdown("**📂 示例音频 (点击加载):**")
                gr.Examples(
                    examples=[
                        ["examples/ck7vv9ag.wav", "default", DEFAULT_S2T_PROMPT],
                    ],
                    inputs=[s2t_audio_input, s2t_preset, s2t_system_prompt],
                    label="示例"
                )
                
                s2t_btn.click(
                    fn=speech_to_text,
                    inputs=[s2t_audio_input, s2t_system_prompt],
                    outputs=s2t_output
                )
            
            # Speech-to-Speech Tab
            with gr.TabItem("🔊 Speech-to-Speech (语音转语音)"):
                gr.Markdown("上传或录制音频，同时获取语音和文字回复。默认使用中文女声 (小云)。")
                
                with gr.Row():
                    with gr.Column():
                        s2s_audio_input = gr.Audio(
                            label="Input Audio / 输入音频",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        s2s_system_prompt = gr.Textbox(
                            label="System Prompt / 系统提示词 (可选)",
                            placeholder="留空使用默认人设 (小云: 来自杭州的温柔女孩)",
                            value="",
                            lines=3
                        )
                        s2s_btn = gr.Button("🎯 Generate / 生成", variant="primary")
                    
                    with gr.Column():
                        s2s_text_output = gr.Textbox(
                            label="Generated Text / 生成文本",
                            lines=5
                        )
                        s2s_audio_output = gr.Audio(
                            label="Generated Audio / 生成音频",
                            type="filepath"
                        )
                
                # Example section
                gr.Markdown("**📂 示例音频 (点击加载):**")
                gr.Examples(
                    examples=[
                        ["examples/ck7vv9ag.wav", ""],
                    ],
                    inputs=[s2s_audio_input, s2s_system_prompt],
                    label="示例"
                )
                
                s2s_btn.click(
                    fn=speech_to_speech,
                    inputs=[s2s_audio_input, s2s_system_prompt],
                    outputs=[s2s_text_output, s2s_audio_output]
                )
            
            # RAG Knowledge Base Tab
            with gr.TabItem("📚 Knowledge Base QA (知识库问答)"):
                gr.Markdown(
                    """
                    基于知识库的语音问答。上传你的知识库内容，然后用语音提问，模型会基于知识库内容回答。
                    
                    **使用方法:**
                    1. 在左侧文本框输入你的知识库内容（每行一条知识）
                    2. 录制或上传你的问题音频
                    3. 选择仅文字回答或语音+文字回答
                    
                    **示例知识库:**
                    ```
                    Fun-Audio-Chat是阿里云开发的大型音频语言模型，支持语音对话。
                    Fun-Audio-Chat使用双分辨率语音表示技术，帧率为5Hz，比其他模型更高效。
                    Fun-Audio-Chat支持语音问答、音频理解、语音函数调用等功能。
                    CosyVoice是用于语音合成的模型，可以将文本转换为自然语音。
                    ```
                    """
                )
                
                with gr.Row():
                    with gr.Column():
                        rag_knowledge = gr.Textbox(
                            label="Knowledge Base / 知识库内容 (每行一条)",
                            placeholder="输入你的知识库内容，每行一条知识...\n例如:\n公司成立于2020年，总部位于杭州。\n公司主要业务是人工智能研发。\n公司有500名员工。",
                            lines=10,
                            value="""Fun-Audio-Chat是阿里云通义实验室开发的大型音频语言模型，支持自然、低延迟的语音交互。
Fun-Audio-Chat使用双分辨率语音表示技术（5Hz骨干网络+25Hz精细头部），计算效率比其他模型高50%。
Fun-Audio-Chat支持语音问答、音频理解、语音函数调用、语音指令跟随等功能。
Fun-Audio-Chat在OpenAudioBench、VoiceBench等多个评测中取得领先成绩。
CosyVoice是阿里云开发的语音合成模型，可以将文本转换为自然流畅的语音。
小云是Fun-Audio-Chat的默认语音人设，是一位来自杭州的温柔友善的女孩。"""
                        )
                        rag_audio_input = gr.Audio(
                            label="Question Audio / 问题音频",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        with gr.Row():
                            rag_text_btn = gr.Button("📝 文字回答", variant="primary")
                            rag_speech_btn = gr.Button("🔊 语音回答", variant="secondary")
                    
                    with gr.Column():
                        rag_question = gr.Textbox(
                            label="Transcribed Question / 识别的问题",
                            lines=2
                        )
                        rag_retrieved = gr.Textbox(
                            label="Retrieved Knowledge / 检索到的知识",
                            lines=4
                        )
                        rag_answer = gr.Textbox(
                            label="Generated Answer / 生成的回答",
                            lines=5
                        )
                        rag_audio_output = gr.Audio(
                            label="Audio Response / 语音回答",
                            type="filepath"
                        )
                
                # Example section
                gr.Markdown("**📂 示例 (点击加载):**")
                gr.Examples(
                    examples=[
                        ["examples/ck7vv9ag.wav", """Fun-Audio-Chat是阿里云通义实验室开发的大型音频语言模型。
Fun-Audio-Chat使用双分辨率语音表示技术，帧率为5Hz。
小云是默认语音人设，是一位来自杭州的温柔女孩。"""],
                    ],
                    inputs=[rag_audio_input, rag_knowledge],
                    label="示例"
                )
                
                rag_text_btn.click(
                    fn=rag_speech_to_text,
                    inputs=[rag_audio_input, rag_knowledge],
                    outputs=[rag_question, rag_retrieved, rag_answer]
                )
                
                rag_speech_btn.click(
                    fn=rag_speech_to_speech,
                    inputs=[rag_audio_input, rag_knowledge],
                    outputs=[rag_question, rag_retrieved, rag_answer, rag_audio_output]
                )
        
        gr.Markdown(
            """
            ---
            ### 📌 使用说明
            1. 模型已自动加载，等待加载完成即可使用
            2. 选择功能 Tab，上传或录制音频
            3. 点击生成按钮获取结果
            
            ### ⚙️ 系统要求
            - GPU 显存: ~24GB
            - 支持格式: WAV, MP3, FLAC 等
            - 默认语音: 中文女声 (小云)
            
            ### 🔗 相关链接
            - [GitHub](https://github.com/FunAudioLLM/Fun-Audio-Chat) | [HuggingFace](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B) | [Technical Report](https://github.com/FunAudioLLM/Fun-Audio-Chat/blob/main/Fun-Audio-Chat-Technical-Report.pdf)
            """
        )
    
    return demo

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fun-Audio-Chat Gradio Demo")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()
    
    # Auto-load models on startup
    logger.info("Auto-loading models on startup...")
    load_models()
    
    # Create and launch the demo
    demo = create_gradio_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
