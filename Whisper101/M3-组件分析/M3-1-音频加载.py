"""
M3-1: 音频加载与波形图可视化

本脚本演示如何使用 OpenAI Whisper 的 `load_audio` 函数加载音频，
并利用 librosa 和 matplotlib 将其可视化为波形图。

依赖安装:
- pip install librosa matplotlib
- [Ubuntu/Debian] sudo apt-get update && sudo apt-get install -y fonts-wqy-zenhei
"""

import os
import whisper
import librosa
import matplotlib.pyplot as plt

def setup_matplotlib_for_chinese():
    """
    配置 Matplotlib 以正确显示中文字符。
    
    Returns:
        bool: 是否成功配置中文字体支持
    """
    import matplotlib.font_manager as fm
    
    # 重新初始化字体管理器以获取最新字体列表
    try:
        fm.fontManager.__init__()
    except:
        pass
    
    # 按优先级尝试不同的中文字体
    chinese_fonts = [
        'WenQuanYi Zen Hei',
        'SimHei', 
        'Microsoft YaHei',
        'Arial Unicode MS'
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找并配置第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"已配置中文字体: {font}")
            return True
    
    # 如果未找到已知字体，尝试使用系统默认的 WenQuanYi Zen Hei
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("使用默认中文字体配置")
    return True

def create_waveform_visualization(audio_path, output_path, use_chinese=True):
    """
    加载音频文件，生成其波形图，并保存为图片。

    Args:
        audio_path (str): 输入的音频文件路径。
        output_path (str): 输出的图片文件路径。
        use_chinese (bool): 是否使用中文标签。
    """
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件未找到 -> {os.path.abspath(audio_path)}")
        return

    print(f"正在加载音频: {os.path.basename(audio_path)}...")
    audio_array = whisper.load_audio(audio_path)
    print(f"音频被加载为 NumPy 数组，形状: {audio_array.shape}，采样率: {whisper.audio.SAMPLE_RATE} Hz")

    print(f"正在生成波形图并保存到: {os.path.basename(output_path)}...")
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(audio_array, sr=whisper.audio.SAMPLE_RATE, alpha=0.8)

    if use_chinese:
        plt.title("音频波形图 (Audio Waveform)")
        plt.xlabel("时间 (秒)")
        plt.ylabel("振幅 (Amplitude)")
    else:
        plt.title("Audio Waveform")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("波形图保存成功！")


def main():
    """
    脚本主入口函数。
    """
    # 配置中文字体支持
    chinese_support = setup_matplotlib_for_chinese()
    
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建输入音频文件的绝对路径
    input_audio = os.path.join(script_dir, '..', 'audio', 'M2-1-cn.wav')

    # 构建输出图片的绝对路径
    output_image = os.path.join(script_dir, 'M3-1-音频加载.png')
    
    # 执行音频波形图生成任务
    create_waveform_visualization(audio_path=input_audio, output_path=output_image, use_chinese=chinese_support)


if __name__ == "__main__":
    main()