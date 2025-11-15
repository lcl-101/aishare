###!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf
import numpy as np
import gradio as gr
import tempfile
import os

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009


def build_prompt(tokenizer, description: str, text: str) -> str:
    """Build formatted prompt for Maya1."""
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token
    
    formatted_text = f'<description="{description}"> {text}'
    
    prompt = (
        soh_token + bos_token + formatted_text + eot_token +
        eoh_token + soa_token + sos_token
    )
    
    return prompt


def extract_snac_codes(token_ids: list) -> list:
    """Extract SNAC codes from generated tokens."""
    try:
        eos_idx = token_ids.index(CODE_END_TOKEN_ID)
    except ValueError:
        eos_idx = len(token_ids)
    
    snac_codes = [
        token_id for token_id in token_ids[:eos_idx]
        if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
    ]
    
    return snac_codes


def unpack_snac_from_7(snac_tokens: list) -> list:
    """Unpack 7-token SNAC frames to 3 hierarchical levels."""
    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]
    
    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[:frames * SNAC_TOKENS_PER_FRAME]
    
    if frames == 0:
        return [[], [], []]
    
    l1, l2, l3 = [], [], []
    
    for i in range(frames):
        slots = snac_tokens[i*7:(i+1)*7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([
            (slots[1] - CODE_TOKEN_OFFSET) % 4096,
            (slots[4] - CODE_TOKEN_OFFSET) % 4096,
        ])
        l3.extend([
            (slots[2] - CODE_TOKEN_OFFSET) % 4096,
            (slots[3] - CODE_TOKEN_OFFSET) % 4096,
            (slots[5] - CODE_TOKEN_OFFSET) % 4096,
            (slots[6] - CODE_TOKEN_OFFSET) % 4096,
        ])
    
    return [l1, l2, l3]


# Global variables to store loaded models
model = None
tokenizer = None
snac_model = None


def load_models():
    """Load models once at startup."""
    global model, tokenizer, snac_model
    
    if model is not None:
        return
    
    print("\n[1/2] Loading Maya1 model...")
    model = AutoModelForCausalLM.from_pretrained(
        "checkpoints/maya1", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "checkpoints/maya1",
        trust_remote_code=True
    )
    print(f"Model loaded: {len(tokenizer)} tokens in vocabulary")
    
    print("\n[2/2] Loading SNAC audio decoder...")
    snac_model = SNAC.from_pretrained("checkpoints/snac_24khz").eval()
    if torch.cuda.is_available():
        snac_model = snac_model.to("cuda")
    print("SNAC decoder loaded")


def generate_speech(description: str, text: str, progress=gr.Progress()):
    """Generate speech from description and text."""
    global model, tokenizer, snac_model
    
    try:
        # Load models if not loaded
        progress(0.1, desc="Loading models...")
        load_models()
        
        progress(0.2, desc="Building prompt...")
        print(f"\nGenerating speech...")
        print(f"Description: {description}")
        print(f"Text: {text}")
        
        # Create prompt with proper formatting
        prompt = build_prompt(tokenizer, description, text)
        
        # Generate emotional speech
        progress(0.3, desc="Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"Input token count: {inputs['input_ids'].shape[1]} tokens")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        progress(0.4, desc="Generating audio tokens...")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=2048,
                min_new_tokens=28,
                temperature=0.4, 
                top_p=0.9, 
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Extract generated tokens
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
        print(f"Generated {len(generated_ids)} tokens")
        
        # Extract SNAC audio tokens
        progress(0.6, desc="Extracting audio codes...")
        snac_tokens = extract_snac_codes(generated_ids)
        print(f"Extracted {len(snac_tokens)} SNAC tokens")
        
        if len(snac_tokens) < 7:
            return None, "错误: 生成的 SNAC 令牌不足，请重试。"
        
        # Unpack SNAC tokens
        progress(0.7, desc="Unpacking audio frames...")
        levels = unpack_snac_from_7(snac_tokens)
        frames = len(levels[0])
        print(f"Unpacked to {frames} frames")
        
        # Convert to tensors
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codes_tensor = [
            torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
            for level in levels
        ]
        
        # Generate final audio
        progress(0.8, desc="Decoding audio...")
        with torch.inference_mode():
            z_q = snac_model.quantizer.from_codes(codes_tensor)
            audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()
        
        # Trim warmup samples
        if len(audio) > 2048:
            audio = audio[2048:]
        
        duration_sec = len(audio) / 24000
        print(f"Audio generated: {len(audio)} samples ({duration_sec:.2f}s)")
        
        # Save to temporary file
        progress(0.9, desc="Saving audio...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, audio, 24000)
            output_path = f.name
        
        progress(1.0, desc="Complete!")
        status_msg = f"✓ 已生成 {duration_sec:.2f}s 的音频 ({frames} 帧)"
        return output_path, status_msg
        
    except Exception as e:
        import traceback
        error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def create_ui():
    """Create Gradio interface."""
    with gr.Blocks(title="Maya1 语音生成", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎤 Maya1 语音生成
            使用自然语言描述生成带有情感的语音。
            
            **支持的情感标签**: `<laugh>`, `<laugh_harder>`, `<sigh>`, `<gasp>` 等
            """
        )
        
        with gr.Row():
            with gr.Column():
                description = gr.Textbox(
                    label="声音描述",
                    placeholder="例如: Realistic female voice in the 20s age with british accent. Normal pitch, warm timbre, conversational pacing.",
                    value="Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.",
                    lines=3,
                )
                
                text = gr.Textbox(
                    label="要朗读的文本",
                    placeholder="输入想要转换为语音的文本...",
                    value="Hello! This is Maya1 <laugh_harder> the best open source voice AI model with emotions.",
                    lines=4,
                )
                
                generate_btn = gr.Button("🎵 生成语音", variant="primary", size="lg")
                
            with gr.Column():
                audio_output = gr.Audio(
                    label="生成的音频",
                    type="filepath",
                )
                
                status = gr.Textbox(
                    label="状态",
                    interactive=False,
                    lines=2,
                )
        
        gr.Markdown(
            """
            ### 提示:
            - 描述声音特征：年龄、性别、口音、音高、音色、语速
            - 使用标签添加情感，如 `<laugh>`, `<sigh>`, `<gasp>`
            - 保持文本简洁以获得更好的质量
            - 生成可能需要 10-30 秒，具体取决于文本长度
            """
        )
        
        gr.Examples(
            examples=[
                [
                    "Young female voice with british accent. High pitch, bright timbre, energetic pacing.",
                    "Welcome to the future of AI voice technology! <laugh>"
                ],
                [
                    "Deep male voice with authoritative tone. Low pitch, rich timbre, slow pacing.",
                    "This is an important announcement. <pause> Please pay attention."
                ],
                [
                    "Cheerful female voice in the 20s. Normal pitch, warm and friendly tone.",
                    "Hi there! How are you doing today? <laugh_harder>"
                ],
            ],
            inputs=[description, text],
            label="示例提示词"
        )
        
        generate_btn.click(
            fn=generate_speech,
            inputs=[description, text],
            outputs=[audio_output, status],
        )
    
    return demo


def main():
    """Main entry point."""
    print("启动 Maya1 语音生成 Web 界面...")
    demo = create_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
