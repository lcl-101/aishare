"""
M3-2: 梅尔频谱图可视化

本脚本演示 Whisper 工作流的第二步：将加载的音频数据转换为梅尔频谱图。
这是 Transformer 编码器的真正输入。

"""
import os
import whisper
import matplotlib.pyplot as plt
import torch  # Whisper 的频谱图计算返回的是 PyTorch 张量

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

def create_mel_spectrogram_visualization(audio_path, output_path, use_chinese=True):
    """
    加载音频，计算其对数梅尔频谱图，并将其可视化后保存。

    Args:
        audio_path (str): 输入的音频文件路径。
        output_path (str): 输出的图片文件路径。
        use_chinese (bool): 是否使用中文标签。
    """
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件未找到 -> {os.path.abspath(audio_path)}")
        return

    # 1. 加载音频 (与上一步相同)
    audio_array = whisper.load_audio(audio_path)

    # 2. 填充或裁剪音频至30秒
    # Whisper 的模型是基于 30 秒的音频片段进行训练的
    audio_chunk = whisper.pad_or_trim(audio_array)
    print(f"音频已填充/裁剪至 {len(audio_chunk) / whisper.audio.SAMPLE_RATE:.1f} 秒")

    # 3. 计算对数梅尔频谱图
    # 这是 Whisper 模型最关键的预处理步骤
    mel_spectrogram = whisper.log_mel_spectrogram(audio_chunk)
    print(f"已生成梅尔频谱图，形状为: {mel_spectrogram.shape}")
    print("这个 (80, 3000) 的张量就是模型的'视觉'输入。")

    # 4. 可视化频谱图
    print(f"正在生成频谱图并保存到: {os.path.basename(output_path)}...")
    plt.figure(figsize=(15, 7))
    
    # 使用 imshow 绘制二维图像 (频谱图)
    # origin='lower' 表示 (0,0) 坐标在左下角，符合频率从低到高的习惯
    # aspect='auto' 让图像自适应图形尺寸
    # cmap='viridis' 是一种美观且色觉友好的颜色映射
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')

    if use_chinese:
        plt.title("对数梅尔频谱图 (Log-Mel Spectrogram)")
        plt.xlabel("帧 (时间)")
        plt.ylabel("梅尔频率斌 (Mel-frequency bins)")
        
        # 添加一个颜色条，用于说明颜色的含义 (分贝)
        colorbar = plt.colorbar(format='%+2.0f dB')
        colorbar.set_label('能量 (dB)')
    else:
        plt.title("Log-Mel Spectrogram")
        plt.xlabel("Frames (Time)")
        plt.ylabel("Mel-frequency bins")
        
        # 添加一个颜色条，用于说明颜色的含义 (分贝)
        colorbar = plt.colorbar(format='%+2.0f dB')
        colorbar.set_label('Energy (dB)')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("频谱图保存成功！")


def main():
    """
    脚本主入口函数。
    """
    # 配置中文字体支持
    chinese_support = setup_matplotlib_for_chinese()
    
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_audio = os.path.join(script_dir, '..', 'audio', 'M2-1-cn.wav')
    output_image = os.path.join(script_dir, 'M3-2-频谱图可视化.png')
    
    # 执行频谱图生成任务
    create_mel_spectrogram_visualization(audio_path=input_audio, output_path=output_image, use_chinese=chinese_support)


if __name__ == "__main__":
    main()