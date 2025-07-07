"""
M3-3: 编码器和语言检测 (从本地加载模型)

本脚本演示 Whisper 工作流的第三步：使用编码器处理梅尔频谱图。
我们将从本地文件加载预先下载好的模型，进行语言检测并观察其输出。
特别注意：本脚本处理了 large-v3 模型需要 128 个梅尔频率通道的特殊情况。

依赖安装:
- pip install -U openai-whisper
- [可选] PyTorch with CUDA for GPU acceleration
"""

import os
import whisper
import torch

def demonstrate_encoder_and_language_detection(audio_path, model_path):
    """
    加载本地 Whisper 模型文件，并演示其编码器在语言检测和特征提取中的作用。

    Args:
        audio_path (str): 输入的音频文件路径。
        model_path (str): 本地 Whisper 模型文件 (.pt) 的路径。
    """
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件未找到 -> {os.path.abspath(audio_path)}")
        return

    if not os.path.exists(model_path):
        print(f"错误: 本地模型文件未找到 -> {os.path.abspath(model_path)}")
        return

    # 1. 从本地文件加载 Whisper 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在从本地路径 '{os.path.basename(model_path)}' 加载模型到 {device}...")
    model = whisper.load_model(model_path, device=device)
    
    # 2. 预处理音频
    audio_array = whisper.load_audio(audio_path)
    audio_chunk = whisper.pad_or_trim(audio_array)

    # 【核心修复】: 从加载的模型中动态获取所需的梅尔通道数
    # -----------------------------------------------------------------
    n_mels_required = model.dims.n_mels
    print(f"'{os.path.basename(model_path)}' 模型需要 {n_mels_required} 个梅尔频率通道。")
    
    # 使用获取到的 n_mels 值来生成频谱图
    mel_spectrogram = whisper.log_mel_spectrogram(audio_chunk, n_mels=n_mels_required).to(model.device)
    # -----------------------------------------------------------------
    
    mel_batch = mel_spectrogram.unsqueeze(0)
    
    # --- 演示 1: 语言检测 ---
    print("\n--- 1. 演示语言检测 ---")
    # `detect_language` 函数内部只使用编码器来预测语言
    _, probs = model.detect_language(mel_batch)
    detected_language = max(probs[0], key=probs[0].get)
    probability = probs[0][detected_language]
    
    print(f"模型检测到的语言是: '{detected_language}' (概率: {probability:.4f})")
    
    # --- 演示 2: 获取编码器的原始输出 ---
    print("\n--- 2. 查看编码器的最终输出 ---")
    encoder_output = model.encoder(mel_batch)
    
    print(f"编码器输出的张量 (Tensor) 形状为: {encoder_output.shape}")
    print("解释这个形状 (batch_size, sequence_length, embedding_dim):")
    print(f"- {encoder_output.shape[0]}: 批处理大小 (我们一次处理 1 个音频)")
    print(f"- {encoder_output.shape[1]}: 序列长度 (音频在时间上被压缩成了 {encoder_output.shape[1]} 帧)")
    print(f"- {encoder_output.shape[2]}: 特征维度 (large-v3 模型每一帧被编码成一个包含 {encoder_output.shape[2]} 个数字的向量)")
    print("\n这个高维张量就是编码器对音频的'深度理解'，准备交给解码器。")


def main():
    """
    脚本主入口函数。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_audio = os.path.join(script_dir, '..', 'audio', 'M2-1-cn.wav')
    local_model_path = os.path.join(script_dir, '..', 'models', 'large-v3.pt')

    demonstrate_encoder_and_language_detection(
        audio_path=input_audio, 
        model_path=local_model_path
    )


if __name__ == "__main__":
    main()