"""
M4-7: 校对增强版 (最终完全体)

本脚本是整个 Module 的最终形态，采用了“代码预处理 + LLM校对”的终极方案：
1.  **代码预处理**: 确定性地在每个语音片段后添加逗号，生成一份“标点草稿”。
2.  **LLM校对**: LLM 的任务被简化为“校对和修正”这份草稿，将部分逗号根据语境智能地升级为句号、问号等。
这是一种极其高效和可靠的 AI 工程实践。

最终流程:
1. 转录 -> 2. 预处理 -> 3. 代码生成标点草稿 -> 4. LLM 校对修正 -> 5. 摘要提取 -> 6. 保存
"""
import os
import whisper
import datetime
import torch
import re
import ollama

# --- 1. 文本处理函数 ---
# (apply_replacements 和 add_spaces_around_english 无变化)
def apply_replacements(text, replacement_map):
    for old_word, new_word in replacement_map.items():
        text = text.replace(old_word, new_word)
    return text

def add_spaces_around_english(text):
    pattern1 = re.compile(r'([\u4e00-\u9fa5])([a-zA-Z0-9]+)')
    pattern2 = re.compile(r'([a-zA-Z0-9]+)([\u4e00-\u9fa5])')
    text = pattern1.sub(r'\1 \2', text)
    text = pattern2.sub(r'\1 \2', text)
    return text

# <--- 重点：全新的 LLM 校对函数 ---
def correct_punctuation_with_llm(text_with_commas, llm_model_name):
    """
    接收一份由逗号预处理过的“标点草稿”，让 LLM 进行校对和修正。
    """
    if not text_with_commas.strip():
        return ""

    prompt = f"""# 角色
你是一位经验丰富的中文总编审，擅长精修文稿。

# 任务
你收到了一份由初级助理处理过的文稿。助理已在每个自然的语音停顿处插入了【逗号】，形成了一份“标点草稿”。
你的任务是【审校并修正】这份草稿，将一些不恰当的逗号，根据上下文语境和句子完整性，修正为更合适的【句号】或【问号】等。

# 指导原则
1.  **修正为主**: 你的主要工作是判断哪些逗号应该被“升级”为句号。
2.  **保留为辅**: 如果一个逗号位于一个长句的中间，用于自然的停顿，那么它应该被保留。
3.  **忠实内容**: 绝对不允许增删或改动原文的任何字词。

# 学习示例 (Examples)
*   **示例 1 (需要修正)**
    *   **标点草稿**: 大家好我是小明,今天我们来聊聊人工智能,
    *   **期望输出**: 大家好,我是小明。今天我们来聊聊人工智能。

*   **示例 2 (部分修正，部分保留)**
    *   **标点草稿**: 这个项目基于Python语言,使用了多种先进的算法模型,效果非常出色,
    *   **期望输出**: 这个项目基于Python语言,使用了多种先进的算法模型,效果非常出色。


# 待校对的标点草稿
---
{text_with_commas}
---

# 输出要求
请直接输出经过你最终审校和修正后的完美文稿。
/no_think
"""

    try:
        print("   🔄 正在发送 [校对] 请求到 Ollama...")
        response = ollama.chat(
            model=llm_model_name, messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.4} # 校对任务需要一定的上下文理解，但不能太随意
        )
        corrected_text = response['message']['content'].strip()
        print("   ✅ Ollama [校对] 完成。")
        return corrected_text
    except Exception as e:
        print(f"❌ 调用 Ollama API [校对] 时出错: {e}")
        # 出错时，返回至少保证断句的草稿
        return text_with_commas

# (extract_summary_with_llm 和 save_text_file 无变化)
def extract_summary_with_llm(full_text, llm_model_name):
    # ... (无变化)
    if not full_text.strip(): return "原文为空，无法生成摘要。"
    prompt = f"""# 角色
你是一位专业的内容分析师和摘要专家。
# 任务
你的任务是为以下完整的文稿，提取核心要点，并生成一段简洁、流畅、重点突出的摘要。
# 指导原则
1. **长度控制**: 摘要的长度应保持在150到300字之间，精准概括，避免冗长。
2. **内容覆盖**: 摘要应覆盖文稿的主要话题、关键概念和最终结论。
3. **格式灵活**: 你可以使用无序列表（使用-或*）来呈现要点，也可以直接使用通顺的段落形式。
4. **保持中立**: 摘要应客观反映原文内容，不添加个人观点。
# 待处理文稿
---
{full_text}
---
# 输出要求
请直接输出摘要内容，不要包含任何额外的前缀，如“这是摘要：”。
/no_think
"""
    try:
        print("   🔄 正在发送 [摘要] 请求到 Ollama...")
        response = ollama.chat(
            model=llm_model_name, messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3}
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"摘要生成失败: {e}"

def save_text_file(text, output_path):
    # ... (无变化)
    with open(output_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)
    print(f"📁 文件已保存: {os.path.abspath(output_path)}")

# --- 2. 主流程函数 ---
def run_transcription_pipeline(model_name, media_path, language, replacement_map, llm_model_name):
    """执行完整的转录、后处理和摘要提取流程 (v7.0 - 校对增强版)。"""
    print("🚀 开始 Whisper 语音转录与智能后处理流程")
    print("=" * 70)
    
    total_start_time = datetime.datetime.now()
    # ... (前面步骤的初始化无变化)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    output_dir = script_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 步骤 1 & 2: 加载模型与转录 (无变化)
    # ... (代码省略，与前一版相同)
    step1_start = datetime.datetime.now()
    print(f"🔧 步骤 1: 正在加载 Whisper 模型 '{model_name}'...")
    model = whisper.load_model(model_name, device=device)
    step1_end = datetime.datetime.now(); step1_duration = step1_end - step1_start
    print(f"✅ 模型加载完成。耗时: {step1_duration}\n")

    step2_start = datetime.datetime.now()
    print(f"🎙️ 步骤 2: 正在转录音频文件...")
    result = model.transcribe(media_path, language=language, verbose=False)
    original_segments = result['segments']
    original_text_display = result['text'].strip()
    step2_end = datetime.datetime.now(); step2_duration = step2_end - step2_start
    print(f"✅ 音频转录完成。耗时: {step2_duration}")
    print("\n📝 原始转录文本 (无标点拼接):\n" + "-"*50 + f"\n{original_text_display}\n" + "-"*50 + "\n")


    # 步骤 3 & 4: 预处理 (无变化)
    # ... (代码省略，与前一版相同)
    step3_4_start = datetime.datetime.now()
    print("🔄 步骤 3 & 4: 正在进行热词替换与中英文空格处理...")
    processed_segments = []
    for segment in original_segments:
        new_segment = segment.copy()
        text = apply_replacements(new_segment['text'], replacement_map)
        text = add_spaces_around_english(text)
        new_segment['text'] = text
        processed_segments.append(new_segment)
    step3_4_end = datetime.datetime.now(); step3_4_duration = step3_4_end - step3_4_start
    print(f"✅ 预处理完成。耗时: {step3_4_duration}\n")


    # <--- 重点修改：全新的步骤5 ---
    # 步骤 5.1: 代码生成标点草稿
    step5_start = datetime.datetime.now()
    print("✍️  步骤 5.1: 正在由代码生成“标点草稿”...")
    texts_to_join = [s['text'].strip() for s in processed_segments if s['text'].strip()]
    text_with_commas = ",".join(texts_to_join) + "," # 在末尾也加上逗号，让LLM决定最后用什么标点
    print(text_with_commas)
    print("✅ “标点草稿”生成完成。\n")

    # 步骤 5.2: LLM 校对修正
    print(f"🤖 步骤 5.2: 正在调用 LLM '{llm_model_name}' 进行校对修正...")
    final_text = correct_punctuation_with_llm(text_with_commas, llm_model_name)
    step5_end = datetime.datetime.now()
    step5_duration = step5_end - step5_start # 整个步骤5的耗时
    print(f"✅ LLM 校对处理完成。耗时: {step5_duration}\n")
    print("✨ 最终全文 (经校对):\n" + "="*50 + f"\n{final_text}\n" + "="*50 + "\n")
    # <--- 修改结束 ---

    # 后续步骤无变化
    # 步骤 6: LLM 摘要提取
    step6_start = datetime.datetime.now()
    print(f"📜 步骤 6: 正在调用 LLM '{llm_model_name}' 进行摘要提取...")
    summary_text = extract_summary_with_llm(final_text, llm_model_name)
    step6_end = datetime.datetime.now(); step6_duration = step6_end - step6_start
    print(f"✅ LLM 摘要提取完成。耗时: {step6_duration}\n")
    print("💡 文本摘要:\n" + "*"*50 + f"\n{summary_text}\n" + "*"*50 + "\n")
    
    # 步骤 7: 保存文件
    # ... (代码省略，与前一版相同)
    step7_start = datetime.datetime.now()
    print("💾 步骤 7: 正在保存所有结果文件...")
    base_name = os.path.splitext(os.path.basename(media_path))[0]
    
    final_txt_path = os.path.join(output_dir, f"{base_name}_final_text.txt")
    save_text_file(final_text, final_txt_path)
    
    summary_txt_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    save_text_file(summary_text, summary_txt_path)
    
    original_txt_path = os.path.join(output_dir, f"{base_name}_original.txt")
    save_text_file(original_text_display, original_txt_path)
    
    step7_end = datetime.datetime.now(); step7_duration = step7_end - step7_start
    print(f"✅ 文件保存完成。耗时: {step7_duration}\n")

    # 总结统计
    # ... (代码省略，与前一版相同)
    total_end_time = datetime.datetime.now()
    total_duration = total_end_time - total_start_time
    
    print("🎉 全部流程完成!")
    print("=" * 70)
    print("⏱️ 各步骤耗时统计:")
    print(f"   步骤1 (模型加载): {step1_duration}")
    print(f"   步骤2 (音频转录): {step2_duration}")
    print(f"   步骤3&4 (预处理):  {step3_4_duration}")
    print(f"   步骤5 (校对流程): {step5_duration}")
    print(f"   步骤6 (LLM摘要):   {step6_duration}")
    print(f"   步骤7 (文件保存): {step7_duration}")
    print(f"   ⏰ 总计耗时:      {total_duration}")
    print("\n📁 输出文件:")
    print(f"   - 最终全文: {final_txt_path}")
    print(f"   - 文本摘要: {summary_txt_path}")
    print(f"   - 原始文本: {original_txt_path}")
    print("=" * 70)


# --- 3. 配置与执行 ---
if __name__ == "__main__":
    # ... (无变化)
    REPLACEMENT_MAP = {
        "乌班图": "Ubuntu", "吉特哈布": "GitHub", "Moddy Talk": "MultiTalk",
        "马克道": "Markdown", "VS 扣德": "VS Code", "派森": "Python",
    }
    
    WHISPER_MODEL_NAME = "large-v3"
    MEDIA_FILE = "../audio/139.wav"
    LANGUAGE = "zh"
    LLM_MODEL_NAME = "qwen3:32b"
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    media_path_full = os.path.join(script_dir, MEDIA_FILE)
    
    if not os.path.exists(media_path_full):
        print(f"❌ 错误: 音频文件不存在 -> {media_path_full}")
    else:
        run_transcription_pipeline(
            model_name=WHISPER_MODEL_NAME, media_path=media_path_full,
            language=LANGUAGE, replacement_map=REPLACEMENT_MAP,
            llm_model_name=LLM_MODEL_NAME
        )