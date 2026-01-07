import gradio as gr
import torch
import torchaudio
import numpy as np
from transformers import AutoModelForCausalLM
from megatron.tokenizer import build_tokenizer
from mucodec.generate_1rvq import Tango
import os


class Args:
    def __init__(self):
        pass


class SongPrepModel:
    def __init__(self, model_path, codec_path, vocal_file, tokenizer="Qwen2Tokenizer", extra_vocab_size=16384):
        print("正在加载音频编解码器...")
        self.tango = Tango(model_path=codec_path)
        
        print("正在加载语言模型...")
        args = Args()
        args.vocab_file = vocal_file
        args.load = model_path
        args.extra_vocab_size = extra_vocab_size
        args.patch_tokenizer_type = tokenizer

        self.tokenizer = build_tokenizer(args)
        self.text_offset = len(self.tokenizer.tokenizer.get_vocab())
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        print("模型加载完成！")

    def process_audio(self, audio_path):
        """处理音频文件并返回歌词和结构分析结果"""
        if audio_path is None:
            return "请先上传音频文件"
        
        try:
            # 加载音频
            src_wave, fs = torchaudio.load(audio_path)
            
            # 重采样到 48kHz
            if fs != 48000:
                src_wave = torchaudio.functional.resample(src_wave, fs, 48000)
            
            # 音频编码
            code = self.tango.sound2code(src_wave)
            
            # 模型推理
            audio = np.array(code[0][0].to("cpu")).astype(np.int32) + self.text_offset
            sentence_ids = [self.tokenizer.sep_token_id] + audio.tolist() + [self.tokenizer.tokenizer.sep_token_id]
            
            prompt = torch.LongTensor(sentence_ids).to("cuda").unsqueeze(0)
            generate_ids = self.model.generate(
                prompt, 
                do_sample=True, 
                top_p=0.1,
                temperature=0.1, 
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id, 
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=8192,
            ).squeeze(0).cpu().numpy()

            # 解析输出
            indices = (generate_ids == self.tokenizer.sep_token_id).nonzero()[0]
            assert len(indices) >= 2, indices
            start = indices[1] + 1
            if len(indices) == 2:
                end = -1
            else:
                end = indices[2] - 1
            
            result = self.tokenizer.detokenize(generate_ids[start:end])
            return result
            
        except Exception as e:
            return f"处理出错: {str(e)}"


# 全局模型实例
model = None


def load_model():
    """加载模型"""
    global model
    if model is None:
        model_path = "checkpoints/SongPrep-7B/"
        codec_path = "checkpoints/SongPrep-7B/mucodec.safetensors"
        vocal_file = "conf/vocab_type.yaml"
        model = SongPrepModel(model_path, codec_path, vocal_file)
    return model


def process_audio(audio_path):
    """处理上传的音频"""
    if audio_path is None:
        return "请先上传音频文件"
    
    m = load_model()
    result = m.process_audio(audio_path)
    return result


def format_result(result):
    """格式化输出结果，使其更易读"""
    if result.startswith("请先") or result.startswith("处理出错"):
        return result
    
    # 将分号分隔的片段换行显示
    segments = result.split(";")
    formatted = []
    for seg in segments:
        seg = seg.strip()
        if seg:
            formatted.append(seg)
    
    return "\n\n".join(formatted)


def analyze_audio(audio_path):
    """分析音频并返回格式化结果"""
    raw_result = process_audio(audio_path)
    formatted_result = format_result(raw_result)
    return raw_result, formatted_result


# 创建 Gradio 界面
with gr.Blocks(title="SongPrep - 歌曲结构分析与歌词识别", theme=gr.themes.Soft()) as demo:
    # YouTube 频道信息
    gr.HTML("""
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">
            📺 欢迎访问我的 YouTube 频道: 
            <a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank" style="color: #ffeb3b; text-decoration: none;">
                AI 技术分享频道
            </a>
        </h3>
        <p style="color: #e0e0e0; margin: 5px 0 0 0; font-size: 14px;">
            点击上方链接订阅频道，获取更多 AI 技术教程！
        </p>
    </div>
    """)
    
    # 标题
    gr.Markdown("""
    # 🎵 SongPrep - 歌曲结构分析与歌词识别
    
    **SongPrep** 是一个端到端的歌曲预处理模型，能够分析完整歌曲的结构并识别歌词，同时提供精确的时间戳，无需额外的音源分离。
    
    ### 使用说明
    1. 上传音频文件（支持 WAV、MP3 等格式）
    2. 点击「开始分析」按钮
    3. 等待模型处理（首次运行需要加载模型，请耐心等待）
    4. 查看分析结果
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # 音频上传
            audio_input = gr.Audio(
                label="上传音频文件",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            # 分析按钮
            analyze_btn = gr.Button("🎯 开始分析", variant="primary", size="lg")
            
            # 示例说明
            gr.Markdown("""
            ### 输出格式说明
            ```
            [structure][start:end]lyric ; [structure][start:end]lyric
            ```
            - **structure**: 歌曲结构标签
            - **start:end**: 片段的起止时间
            - **lyric**: 识别的歌词，句子用 `.` 分隔
            
            ### 结构标签含义
            | 标签 | 中文含义 |
            |------|----------|
            | intro | 前奏 |
            | verse | 主歌 |
            | chorus | 副歌 |
            | bridge | 桥段 |
            | inst | 间奏（纯音乐）|
            | outro | 尾奏 |
            | pre-chorus | 预副歌 |
            | interlude | 插曲 |
            """)
        
        with gr.Column(scale=1):
            # 格式化结果
            formatted_output = gr.Textbox(
                label="分析结果（格式化）",
                lines=15,
                placeholder="分析结果将显示在这里..."
            )
            
            # 原始结果
            raw_output = gr.Textbox(
                label="原始输出",
                lines=5,
                placeholder="原始输出将显示在这里..."
            )
    
    # 绑定事件
    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input],
        outputs=[raw_output, formatted_output]
    )
    
    # 底部信息
    gr.Markdown("""
    ---
    ### 关于模型
    - **模型**: SongPrep-7B
    - **参数量**: 7B
    - **支持语言**: 中文、英文
    - **论文**: [arXiv:2509.17404](https://arxiv.org/abs/2509.17404)
    - **模型权重**: [Hugging Face](https://huggingface.co/waytan22/SongPrep-7B)
    """)


if __name__ == "__main__":
    # 预加载模型
    print("正在预加载模型，请稍候...")
    load_model()
    print("模型加载完成，启动 Web 服务...")
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
