"""
M4-2: 实用技巧 - 文本格式化增强 (迭代版)

本脚本在 M4-1 的基础上，增加了一步文本格式化处理：
使用正则表达式自动为中英文混合文本添加空格，以优化排版。
处理流程变为： 转录 -> 热词替换 -> 格式化 -> 保存。
"""
import os
import whisper
import datetime
import torch
import re  # <--- 新增：导入正则表达式模块

# --- 1. 辅助函数：自定义 SRT 文件写入 (保持不变) ---
def generate_srt_file(segments, output_path):
    """根据字幕分段信息，生成标准 SRT 文件。"""
    with open(output_path, 'w', encoding='utf-8') as srt_file:
        for i, segment in enumerate(segments):
            start_time = str(datetime.timedelta(seconds=segment['start']))
            end_time = str(datetime.timedelta(seconds=segment['end']))
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

# --- 2. 核心函数：文本后处理 (分步进行) ---
def apply_text_replacements(original_segments, replacement_map):
    """第一步处理：对分段文本应用热词替换规则。"""
    import copy
    processed_segments = copy.deepcopy(original_segments)
    
    print("\n>>> 步骤 3.1: 正在应用【热词替换】规则...")
    for segment in processed_segments:
        for old_word, new_word in replacement_map.items():
            segment['text'] = segment['text'].replace(old_word, new_word)
    print("✔️ 热词替换完成。")
    return processed_segments

# <--- 新增：格式化处理函数 ---
def apply_spacing_rules(segments_after_replacement):
    """第二步处理：为中英文混合文本添加空格。"""
    print("\n>>> 步骤 3.2: 正在应用【格式化】规则 (中英文加空格)...")
    
    # 模式1：中文 + 英文/数字 -> 中文 + 空格 + 英文/数字
    # 修改：匹配任意长度的英文单词（包括单个字母）
    pattern1 = re.compile(r'([\u4e00-\u9fa5])([a-zA-Z0-9]+)')
    
    # 模式2：英文/数字 + 中文 -> 英文/数字 + 空格 + 中文  
    # 修改：匹配任意长度的英文单词（包括单个字母）
    pattern2 = re.compile(r'([a-zA-Z0-9]+)([\u4e00-\u9fa5])')
    
    for segment in segments_after_replacement:
        # 先处理中文后面跟英文的情况
        segment['text'] = pattern1.sub(r'\1 \2', segment['text'])
        # 再处理英文后面跟中文的情况
        segment['text'] = pattern2.sub(r'\1 \2', segment['text'])
        
    print("✔️ 格式化完成。")
    return segments_after_replacement
# <--- 新增结束 ---

# --- 3. 主流程函数 ---
def run_transcription_pipeline(model_name, media_path, language, replacement_map):
    """执行完整的转录和后处理流程。"""
    start_time = datetime.datetime.now()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    model_path = os.path.join(script_dir, '..', 'models', f"{model_name}.pt")
    if not os.path.exists(model_path):
        print(f"❌ 错误: 本地模型文件未找到 -> {model_path}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 步骤 1 & 2: 加载与转录 ---
    print(f">>> 步骤 1: 正在从 '{model_path}' 加载 '{model_name}' 模型到 {device}...")
    model = whisper.load_model(model_path, device=device)
    print(f"\n>>> 步骤 2: 正在转录音频文件 '{media_path}'...")
    result = model.transcribe(media_path, language=language, verbose=True)
    print("✔️ 原始转录完成。")

    print("\n" + "="*25 + " 原始转录文本 " + "="*25)
    print(result['text'])
    print("="*65)
    
    # --- 步骤 3: 多步后期处理 ---
    # <--- 修改：将处理流程串联起来 ---
    segments_after_replacement = apply_text_replacements(result['segments'], replacement_map)
    final_processed_segments = apply_spacing_rules(segments_after_replacement)
    # <--- 修改结束 ---
    
    # --- 演示点：展示最终处理后结果 ---
    processed_full_text = " ".join(s['text'].strip() for s in final_processed_segments)
    print("\n" + "="*25 + " 最终处理后文本 " + "="*25)
    print(processed_full_text)
    print("="*65)
    
    # --- 步骤 4: 保存文件 ---
    print("\n>>> 步骤 4: 正在保存结果文件...")
    base_filename = os.path.splitext(os.path.basename(media_path))[0]
    
    # 保存原始文件以供对比
    original_srt_path = os.path.join(output_dir, f"{base_filename}_original.srt")
    generate_srt_file(result['segments'], original_srt_path)
    
    # 保存最终处理后的文件
    processed_srt_path = os.path.join(output_dir, f"{base_filename}_processed.srt")
    generate_srt_file(final_processed_segments, processed_srt_path)
    
    end_time = datetime.datetime.now()
    print(f'\n🎉 全部流程完成! 总耗时: {end_time - start_time}')

# --- “控制面板” ---
if __name__ == "__main__":
    REPLACEMENT_MAP = {
        "乌班图": "Ubuntu",
        "吉特哈布": "GitHub", # <--- 新增一个例子用于测试格式化
        "Moddy Talk": "MultiTalk",
        "呢": "",
    }

    MODEL_NAME = "large-v3"
    MEDIA_FILE = "../audio/139.wav"
    LANGUAGE = "zh"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    media_path_full = os.path.join(script_dir, MEDIA_FILE)
    
    run_transcription_pipeline(
        model_name=MODEL_NAME, 
        media_path=media_path_full, 
        language=LANGUAGE,
        replacement_map=REPLACEMENT_MAP
    )