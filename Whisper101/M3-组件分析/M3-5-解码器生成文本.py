"""
M3-5: 解码器生成文本 (完整流程演示)

本脚本将前面所有组件串联起来，演示 Whisper 的完整工作流程：
1. 加载音频和模型。
2. 预处理音频，生成梅尔频谱图。
3. 使用编码器提取声学特征。
4. 使用解码器根据声学特征生成 Token ID 序列。
5. 使用分词器将 Token ID 序列转换为最终文本。

依赖安装:
- pip install -U openai-whisper
- [可选] PyTorch with CUDA for GPU acceleration
"""

import os
import whisper
import torch

def demonstrate_full_transcription_process(audio_path, model_path):
    """
    演示 Whisper 从音频到文本的完整转录流程。

    Args:
        audio_path (str): 输入的音频文件路径。
        model_path (str): 本地 Whisper 模型文件 (.pt) 的路径。
    """
    if not os.path.exists(audio_path) or not os.path.exists(model_path):
        print("错误: 音频或模型文件未找到。")
        return

    # --- 第 1 步: 加载模型 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"1. 正在从本地路径 '{os.path.basename(model_path)}' 加载模型到 {device}...")
    model = whisper.load_model(model_path, device=device)

    # --- 第 2 步: 音频预处理 & 生成梅尔频谱图 ---
    print("\n2. 正在预处理音频...")
    audio_array = whisper.load_audio(audio_path)
    audio_chunk = whisper.pad_or_trim(audio_array)
    n_mels_required = model.dims.n_mels
    mel_spectrogram = whisper.log_mel_spectrogram(audio_chunk, n_mels=n_mels_required).to(model.device)
    mel_batch = mel_spectrogram.unsqueeze(0)
    print(f"   - 已生成 ({n_mels_required}, 3000) 形状的梅尔频谱图。")

    # --- 第 3 步: 编码器提取特征 ---
    print("\n3. 编码器正在提取声学特征...")
    with torch.no_grad(): # 在推理时使用 no_grad 可以节省显存并加速
        encoder_output = model.encoder(mel_batch)
    print(f"   - 已生成 {encoder_output.shape} 形状的声学特征张量。")

    # --- 第 4 步: 解码器生成 Token IDs ---
    print("\n4. 解码器正在生成 Token IDs...")
    # 配置解码选项。我们可以指定语言，以获得更好的效果。
    # 如果不指定，模型会先进行语言检测。
    decode_options = whisper.DecodingOptions(language="zh", without_timestamps=True)
    
    with torch.no_grad():
        decoding_result = model.decode(encoder_output, decode_options)

    # `decoding_result` 是一个列表，我们取第一个结果
    generated_tokens = decoding_result[0].tokens
    print("   - 解码器已生成 Token ID 序列:")
    print(f"     {generated_tokens}")

    # --- 第 5 步: 分词器解码为文本 ---
    print("\n5. 分词器正在将 Token IDs 转换为文本...")
    # 获取与模型匹配的分词器
    # 这里我们不再需要手动指定语言，因为模型对象已经包含了分词器实例
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="zh", task=decode_options.task)

    # 【修复】这里不再需要访问 vocab_size，直接进行解码
    final_text = tokenizer.decode(generated_tokens)
    
    print("\n" + "="*50)
    print("         最终识别结果")
    print("="*50)
    print(final_text)
    print("="*50)

def main():
    """
    脚本主入口函数。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_audio = os.path.join(script_dir, '..', 'audio', 'M2-1-cn.wav')
    local_model_path = os.path.join(script_dir, '..', 'models', 'large-v3.pt')

    demonstrate_full_transcription_process(
        audio_path=input_audio, 
        model_path=local_model_path
    )

if __name__ == "__main__":
    main()