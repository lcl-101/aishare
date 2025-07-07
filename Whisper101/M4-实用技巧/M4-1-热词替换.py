"""
M4-1: 实用技巧 - 热词替换与高级处理 (教学演示版)

本脚本为教学演示优化，清晰地展示了以下流程：
1. 使用 Whisper 进行基础转录。
2. 在控制台展示原始转录结果。
3. 对转录结果应用自定义的文本替换规则。
4. 在控制台展示处理后的转录结果，形成直观对比。
5. 将原始结果和处理后结果分别保存为 SRT 文件，以便后续分析。
"""
import os
import whisper
import datetime
import torch

# --- 1. 辅助函数：自定义 SRT 文件写入 ---
def generate_srt_file(segments, output_path):
    """根据字幕分段信息，生成标准 SRT 文件。"""
    with open(output_path, 'w', encoding='utf-8') as srt_file:
        for i, segment in enumerate(segments):
            # 使用 timedelta 进行精确的时间格式化
            start_time = str(datetime.timedelta(seconds=segment['start']))
            end_time = str(datetime.timedelta(seconds=segment['end']))
            # 确保毫秒部分是三位数
            start_time = start_time if '.' in start_time else start_time + '.000000'
            end_time = end_time if '.' in end_time else end_time + '.000000'
            start_ms = start_time.split('.')[-1][:3]
            end_ms = end_time.split('.')[-1][:3]
            start_time_str = start_time.split('.')[0] + f',{start_ms}'
            end_time_str = end_time.split('.')[0] + f',{end_ms}'

            text = segment['text'].strip()
            
            srt_file.write(f"{i + 1}\n")
            srt_file.write(f"{start_time_str} --> {end_time_str}\n")
            srt_file.write(f"{text}\n\n")
    print(f"✔️ SRT 文件已保存到: {os.path.abspath(output_path)}")

# --- 2. 核心函数：文本后处理 ---
def apply_text_replacements(original_segments, replacement_map):
    """对分段文本应用替换规则。"""
    import copy
    # 创建一个深拷贝，以避免修改原始数据
    processed_segments = copy.deepcopy(original_segments)
    
    print("\n>>> 步骤 3: 正在对文本应用自定义替换规则...")
    for segment in processed_segments:
        for old_word, new_word in replacement_map.items():
            segment['text'] = segment['text'].replace(old_word, new_word)
    print("✔️ 文本替换完成。")
    return processed_segments

# --- 3. 主流程函数 ---
def run_transcription_pipeline(model_name, media_path, language, replacement_map):
    """执行完整的转录和后处理流程。"""
    start_time = datetime.datetime.now()
    
    # --- 环境和路径准备 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir  # 保存到当前目录
    
    model_path = os.path.join(script_dir, '..', 'models', f"{model_name}.pt")
    if not os.path.exists(model_path):
        print(f"❌ 错误: 本地模型文件未找到 -> {model_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 步骤 1: 加载与转录 ---
    print(f">>> 步骤 1: 正在从 '{model_path}' 加载 '{model_name}' 模型到 {device}...")
    model = whisper.load_model(model_path, device=device)
    
    print(f"\n>>> 步骤 2: 正在转录音频文件 '{media_path}'...")
    result = model.transcribe(media_path, language=language, verbose=True)
    print("✔️ 原始转录完成。")

    # --- 演示点 1: 展示原始结果 ---
    print("\n" + "="*25 + " 原始转录文本 " + "="*25)
    print(result['text'])
    print("="*65)
    
    # --- 步骤 3: 后期处理 ---
    processed_segments = apply_text_replacements(result['segments'], replacement_map)
    
    # --- 演示点 2: 展示处理后结果 ---
    processed_full_text = " ".join(s['text'].strip() for s in processed_segments)
    print("\n" + "="*25 + " 处理后转录文本 " + "="*25)
    print(processed_full_text)
    print("="*65)
    
    # --- 步骤 4: 保存文件 ---
    print("\n>>> 步骤 4: 正在保存结果文件...")
    base_filename = os.path.splitext(os.path.basename(media_path))[0]
    
    # 保存原始文件以供对比
    original_srt_path = os.path.join(output_dir, f"{base_filename}_original.srt")
    generate_srt_file(result['segments'], original_srt_path)
    
    # 保存处理后的文件
    processed_srt_path = os.path.join(output_dir, f"{base_filename}_processed.srt")
    generate_srt_file(processed_segments, processed_srt_path)
    
    end_time = datetime.datetime.now()
    print(f'\n🎉 全部流程完成! 总耗时: {end_time - start_time}')

# --- “控制面板”：在这里修改配置 ---
if __name__ == "__main__":
    # 1. 定义你的热词和替换规则
    REPLACEMENT_MAP = {
        "乌班图": " Ubuntu",      # 中英文之间加空格
        "Moddy Talk": "MultiTalk",  # 修正错误识别并加空格
        "MultiTalk": " MultiTalk",   # 中英文之间加空格
        "呢": "",       # 删除口水词
    }

    # 2. 指定要使用的模型和音频文件
    MODEL_NAME = "large-v3"
    MEDIA_FILE = "../audio/139.wav"
    LANGUAGE = "zh" # 使用 ISO 639-1 语言代码

    # 3. 运行主流程
    script_dir = os.path.dirname(os.path.abspath(__file__))
    media_path_full = os.path.join(script_dir, MEDIA_FILE)
    
    run_transcription_pipeline(
        model_name=MODEL_NAME, 
        media_path=media_path_full, 
        language=LANGUAGE,
        replacement_map=REPLACEMENT_MAP
    )