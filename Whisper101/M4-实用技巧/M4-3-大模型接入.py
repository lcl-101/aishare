"""
M4-4: 智能后处理 - 停顿感知增强版 (终极版)

本脚本是 M4-3 的重大升级，引入了“停顿感知”能力：
1.  **利用 Whisper 分段信息**: 将 Whisper 的 segments 间的自然停顿作为强信号 ([P]) 提供给 LLM。
2.  **全新 Prompt**: 设计了专门的 Prompt，指导 LLM 理解并利用 [P] 标记，实现更精准、更自然的断句。
3.  保留了之前版本的所有优点：详细的耗时统计、清晰的流程、支持多种模型加载方式等。

依赖安装：
pip install ollama

确保Ollama服务已启动：
ollama serve

下载模型（如需要）：
ollama pull qwen2.5:72b
"""
import os
import whisper
import datetime
import torch
import re
import ollama

# --- 1. 文本处理函数 ---
def apply_replacements(text, replacement_map):
    """对单个文本片段应用热词替换规则。"""
    for old_word, new_word in replacement_map.items():
        text = text.replace(old_word, new_word)
    return text

def add_spaces_around_english(text):
    """为中英文混排文本自动添加空格。"""
    pattern1 = re.compile(r'([\u4e00-\u9fa5])([a-zA-Z0-9]+)')
    pattern2 = re.compile(r'([a-zA-Z0-9]+)([\u4e00-\u9fa5])')
    text = pattern1.sub(r'\1 \2', text)
    text = pattern2.sub(r'\1 \2', text)
    return text

# <--- 重点修改：此函数现在接收 segments，并使用带 [P] 的 Prompt ---
def enhance_punctuation_with_llm(segments, llm_model_name):
    """
    使用 LLM 为文本智能添加标点，利用 Whisper 的分段作为停顿信号。
    """
    # 1. 在 segment 之间插入 [P] 标记作为停顿提示
    # ["你好", "今天天气不错"] -> "你好[P]今天天气不错"
    text_with_pauses = "[P]".join([s['text'].strip() for s in segments if s['text'].strip()])

    if not text_with_pauses.strip():
        print("⚠️ 文本内容为空，跳过 LLM 处理。")
        return "" # 返回空字符串

    # 2. 构建新的、更智能的 Prompt
    prompt = f"""# 角色
你是一位顶级的中文文案编辑，能深刻理解语音节奏和书面表达的转换。

# 任务
为下面的原始文本添加标点。这段文本由多个语音片段拼接而成，我使用了特殊标记 `[P]` 来表示原始语音中的自然停顿点。

# 指导原则
1.  **关键指令**: `[P]` 标记处是强烈的断句提示。你应该在这些位置优先考虑使用逗号（，）或句号（。）。
2.  **句内微调**: 即使在没有 `[P]` 标记的片段内部，如果句子过长，你也可以根据逻辑添加逗号。
3.  **忠实内容**: 绝对不允许增删或改动原文的任何汉字或英文单词。
4.  **最终输出**: 在最终结果中，必须移除所有的 `[P]` 标记。

# 待处理文本
---
{text_with_pauses}
---

# 输出要求
直接输出经过你精细编辑和标点优化后的最终文本，确保不含 `[P]` 标记。"""

    # 3. 调用 Ollama API
    try:
        print("   🔄 正在发送请求到 Ollama...")
        response = ollama.chat(
            model=llm_model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.2} # 较低的温度让输出更稳定
        )
        punctuated_text = response['message']['content'].strip()
        print("   ✅ Ollama 处理完成。")
        return punctuated_text
    except Exception as e:
        print(f"❌ 调用 Ollama API 时出错: {e}")
        # 出错时，返回一个不带标点的、拼接好的文本作为后备
        fallback_text = " ".join([s['text'].strip() for s in segments])
        print("   ⚠️ 将使用未处理的原始拼接文本继续。")
        return fallback_text

def save_text_file(text, output_path):
    """将文本保存为 TXT 文件。"""
    with open(output_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)
    print(f"📁 文件已保存: {os.path.abspath(output_path)}")

# --- 2. 主流程函数 ---
def run_transcription_pipeline(model_name, media_path, language, replacement_map, llm_model_name):
    """执行完整的转录和后处理流程，每一步都有详细的耗时统计。"""
    print("🚀 开始 Whisper 语音转录与智能后处理流程 (v4.0 - 停顿感知版)")
    print("=" * 70)
    
    total_start_time = datetime.datetime.now()
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    output_dir = script_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 步骤 1: 加载模型 (代码不变) ---
    # ... (与您提供的代码完全相同)
    step1_start = datetime.datetime.now()
    print(f"🔧 步骤 1: 正在加载 Whisper 模型 '{model_name}' 到 {device}...")
    if os.path.exists(model_name) or '/' in model_name or model_name.endswith('.pt'):
        model = whisper.load_model(model_name, device=device)
        print(f"   📂 使用本地模型文件: {model_name}")
    else:
        model = whisper.load_model(model_name, device=device)
        print(f"   🌐 使用在线模型: {model_name}")
    step1_end = datetime.datetime.now()
    step1_duration = step1_end - step1_start
    print(f"✅ 模型加载完成。耗时: {step1_duration}\n")

    # --- 步骤 2: 转录音频 (代码不变) ---
    step2_start = datetime.datetime.now()
    print(f"🎙️ 步骤 2: 正在转录音频文件...")
    print(f"   📄 文件路径: {media_path}")
    print(f"   🌍 语言设置: {language}")
    
    result = model.transcribe(media_path, language=language, verbose=False)
    # <--- 修改：现在我们保留 segments，原始文本仅用于展示 ---
    original_segments = result['segments']
    original_text_display = result['text'].strip()
    
    step2_end = datetime.datetime.now()
    step2_duration = step2_end - step2_start
    print(f"✅ 音频转录完成。耗时: {step2_duration}")
    print(f"📊 原始文本长度: {len(original_text_display)} 字符, 分段数: {len(original_segments)}")
    
    print("\n📝 原始转录文本 (无标点拼接):")
    print("-" * 50)
    print(original_text_display)
    print("-" * 50 + "\n")
    
    # <--- 修改：后续处理现在基于 segments ---
    # --- 步骤 3 & 4: 逐段进行热词替换和空格处理 ---
    step3_4_start = datetime.datetime.now()
    print("🔄 步骤 3 & 4: 正在进行热词替换与中英文空格处理 (逐段进行)...")
    
    processed_segments = []
    for segment in original_segments:
        # 复制 segment 以免修改原始数据
        new_segment = segment.copy()
        
        # 步骤 3: 热词替换
        text = apply_replacements(new_segment['text'], replacement_map)
        
        # 步骤 4: 中英文空格
        text = add_spaces_around_english(text)
        
        new_segment['text'] = text
        processed_segments.append(new_segment)
    
    step3_4_end = datetime.datetime.now()
    step3_4_duration = step3_4_end - step3_4_start
    print(f"✅ 热词与空格处理完成。耗时: {step3_4_duration}\n")

    # --- 步骤 5: LLM 智能标点 (利用停顿信号) ---
    step5_start = datetime.datetime.now()
    print(f"🤖 步骤 5: 正在调用 LLM '{llm_model_name}' 进行停顿感知智能标点...")
    
    # <--- 修改：将处理过的 segments 传递给 LLM 函数 ---
    final_text = enhance_punctuation_with_llm(processed_segments, llm_model_name)
    
    step5_end = datetime.datetime.now()
    step5_duration = step5_end - step5_start
    print(f"✅ LLM 标点处理完成。耗时: {step5_duration}")
    print(f"📊 最终文本长度: {len(final_text)} 字符\n")
    
    print("✨ LLM 处理后最终文本:")
    print("=" * 50)
    print(final_text)
    print("=" * 50 + "\n")
    
    # --- 步骤 6: 保存文件 (逻辑简化) ---
    step6_start = datetime.datetime.now()
    print("💾 步骤 6: 正在保存最终文本...")
    
    base_name = os.path.splitext(os.path.basename(media_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}_whisper_final.txt")
    save_text_file(final_text, txt_path)
    
    original_txt_path = os.path.join(output_dir, f"{base_name}_whisper_original.txt")
    save_text_file(original_text_display, original_txt_path)
    print(f"📁 原始文本已保存: {os.path.abspath(original_txt_path)}")
    
    step6_end = datetime.datetime.now()
    step6_duration = step6_end - step6_start
    print(f"✅ 文件保存完成。耗时: {step6_duration}\n")
    
    # --- 总结统计 ---
    total_end_time = datetime.datetime.now()
    total_duration = total_end_time - total_start_time
    
    print("🎉 全部流程完成!")
    print("=" * 70)
    print("⏱️ 各步骤耗时统计:")
    print(f"   步骤1 (模型加载): {step1_duration}")
    print(f"   步骤2 (音频转录): {step2_duration}")
    print(f"   步骤3&4 (预处理): {step3_4_duration}")
    print(f"   步骤5 (LLM标点):   {step5_duration}")
    print(f"   步骤6 (文件保存): {step6_duration}")
    print(f"   ⏰ 总计耗时:      {total_duration}")
    print("\n📁 输出文件:")
    print(f"   📄 最终文本: {txt_path}")
    print(f"   📄 原始文本: {original_txt_path}")
    print("=" * 70)

# --- 3. 配置与执行 (代码不变) ---
if __name__ == "__main__":
    # ... (您的配置部分完全不变)
    REPLACEMENT_MAP = {
        "乌班图": "Ubuntu", "吉特哈布": "GitHub", "Moddy Talk": "MultiTalk",
        "马克道": "Markdown", "VS 扣德": "VS Code", "派森": "Python",
        "甲瓦": "Java", "JavaScript": "JavaScript",
    }
    
    WHISPER_MODEL_NAME = "large-v3"
    MEDIA_FILE = "../audio/139.wav"
    LANGUAGE = "zh"
    
    LLM_MODEL_NAME = "qwen2.5:72b"
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    media_path_full = os.path.join(script_dir, MEDIA_FILE)
    
    if not os.path.exists(media_path_full):
        print(f"❌ 错误: 音频文件不存在 -> {media_path_full}")
        print("请检查 MEDIA_FILE 路径设置。")
    else:
        run_transcription_pipeline(
            model_name=WHISPER_MODEL_NAME,
            media_path=media_path_full,
            language=LANGUAGE,
            replacement_map=REPLACEMENT_MAP,
            llm_model_name=LLM_MODEL_NAME
        )