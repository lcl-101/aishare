import gradio as gr
import edge_tts
import asyncio
import tempfile
import os
import re
from typing import List, Tuple
try:
    from mutagen.mp3 import MP3  # 用于获取音频时长
except ImportError:  # 若未安装，后续会提示
    MP3 = None

# 获取可用的语音列表
async def get_voices():
    voices = await edge_tts.list_voices()
    return voices

# 获取语言和声音选项（启动时预加载）
voices = asyncio.run(get_voices())
language_options = sorted({v['Locale'] for v in voices})
voice_options = {lang: [v['ShortName'] for v in voices if v['Locale'] == lang] for lang in language_options}
# 便于大小写无关匹配
_voice_options_lower = {lang.lower(): names for lang, names in voice_options.items()}

def _split_text(text: str) -> List[str]:
    """根据中文/英文常见标点切分文本，返回去除标点后的片段列表。
    保持原始顺序，去除首尾空白与空片段。
    """
    # 包含中文逗号、顿号、句号、分号、问号、感叹号以及对应英文标点
    pieces = re.split(r'[，、。；;！？!?,\.]', text)
    return [p.strip() for p in pieces if p and p.strip()]

def _format_srt_timestamp(ms: int) -> str:
    """ms -> HH:MM:SS,mmm"""
    h = ms // 3600000
    ms_rem = ms % 3600000
    m = ms_rem // 60000
    ms_rem %= 60000
    s = ms_rem // 1000
    ms_final = ms_rem % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms_final:03d}"

def tts_with_srt(text: str, voice: str) -> Tuple[str, str]:
    """生成语音文件与字幕(SRT内容)。

    实现要点：
    1. 使用 edge_tts.Communicate.stream_sync() 获取音频块与 WordBoundary 时间信息。
    2. 收集逐“字/词”时间戳（offset 为起始 100ns 单位，duration 为持续 100ns 单位）。
    3. 按标点切分后的片段顺序映射这些时间，计算每段开始与结束时间。
    4. 生成标准 SRT 字幕文本。
    """
    communicate = edge_tts.Communicate(text, voice)
    audio_bytes = bytearray()
    submaker = edge_tts.SubMaker()  # 作为兜底
    # units: 每个“字符”或细粒度单元 (char, start_ms, end_ms)
    units: List[Tuple[str, int, int]] = []

    for chunk in communicate.stream_sync():
        ctype = chunk.get("type")
        if ctype == "audio":
            audio_bytes.extend(chunk.get("data", b""))
            continue
        if ctype in ("WordBoundary", "SentenceBoundary"):
            # 先喂给 SubMaker（以防我们逻辑失败可直接输出官方字幕）
            submaker.feed(chunk)
        if ctype == "WordBoundary":
            offset = chunk.get("offset", 0)  # 100ns
            duration = chunk.get("duration", 0)
            raw_text = chunk.get("text", "") or ""
            start_ms = offset // 10_000
            end_ms_all = (offset + duration) // 10_000 if duration else start_ms
            if not raw_text:
                continue
            # 若一个 boundary 里包含多个汉字/字符，则平均切分时间
            chars = list(raw_text)
            seg_count = len(chars)
            if seg_count == 1:
                if end_ms_all == start_ms:
                    end_ms_all = start_ms + 150
                units.append((chars[0], start_ms, end_ms_all))
            else:
                total_dur = max(end_ms_all - start_ms, 150 * seg_count)
                per = total_dur / seg_count
                for i, ch in enumerate(chars):
                    c_start = int(start_ms + per * i)
                    c_end = int(start_ms + per * (i + 1))
                    units.append((ch, c_start, c_end))

    # 写入临时音频文件（后面可能需要读取总时长）
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        fp.write(audio_bytes)
        audio_path = fp.name

    # —— 构建精准或近似时间线 ——
    # 1) 如果有 WordBoundary，我们尝试精细到字符；否则使用整段均匀分配
    punctuation_pattern = r'[，、。；;！？!?,\.\s]'
    plain_text = re.sub(punctuation_pattern, '', text)
    char_total = len(plain_text)

    # 过滤掉标点的 units，用于精准映射
    units_no_punct = [u for u in units if not re.match(punctuation_pattern, u[0])]

    precise = False
    if units_no_punct and len(units_no_punct) >= char_total and char_total > 0:
        precise = True
    elif not units:  # 没有任何边界
        precise = False
    else:
        # 数据不足，降级
        precise = False

    if precise:
        # 修补 0 持续时间单元：使用下一个开始或最小 120ms
        for i in range(len(units_no_punct) - 1):
            ch, s, e = units_no_punct[i]
            if e <= s:
                next_start = units_no_punct[i + 1][1]
                units_no_punct[i] = (ch, s, max(next_start, s + 120))
        if units_no_punct:
            ch_last, s_last, e_last = units_no_punct[-1]
            if e_last <= s_last:
                units_no_punct[-1] = (ch_last, s_last, s_last + 200)

    # 映射到片段
    segments = _split_text(text)
    # 构建发音字符（去标点、空白）
    if precise:
        pronounced_text = plain_text
        pronounced_len = char_total
        if pronounced_len > len(units_no_punct):
            pronounced_len = len(units_no_punct)
    else:
        # 近似模式：按整体时长平均分配
        pronounced_text = plain_text
        pronounced_len = char_total
        # 获取总时长
        if units_no_punct:
            total_ms = units_no_punct[-1][2]
        else:
            # 没有 boundary，只能用音频时长
            if MP3 is not None:
                try:
                    total_ms = int(MP3(audio_path).info.length * 1000)
                except Exception:
                    total_ms = 1000 * max(len(text) // 5, 1)  # 粗略兜底
            else:
                total_ms = 1000 * max(len(text) // 5, 1)
        # 构造一个均匀的 units_no_punct
        units_no_punct = []
        if pronounced_len == 0:
            pronounced_len = 1
            pronounced_text = ' '
        per = total_ms / pronounced_len
        for i, ch in enumerate(pronounced_text):
            s_ms = int(per * i)
            e_ms = int(per * (i + 1))
            units_no_punct.append((ch, s_ms, e_ms))

    idx = 0
    srt_lines: List[str] = []
    counter = 1
    for seg in segments:
        pure_seg = seg.replace(' ', '')
        need = len(pure_seg)
        if need == 0:
            continue
        if idx + need > pronounced_len:
            need = max(0, pronounced_len - idx)
        if need == 0:
            break
        seg_units = units_no_punct[idx: idx + need]
        idx += need
        start_ms = seg_units[0][1]
        end_ms = seg_units[-1][2]
        start_ts = _format_srt_timestamp(start_ms)
        end_ts = _format_srt_timestamp(end_ms)
        srt_lines.append(f"{counter}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(seg)
        srt_lines.append("")
        counter += 1

    srt_content = '\n'.join(srt_lines).strip() + '\n'
    return audio_path, srt_content

def generate_audio_and_srt(text, language, voice):
    if not text:
        return None, "", None
    audio_path, srt_content = tts_with_srt(text, voice)
    # 写入临时字幕文件供下载
    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt", mode="w", encoding="utf-8") as sf:
        sf.write(srt_content)
        srt_file_path = sf.name
    return audio_path, srt_content, srt_file_path

def update_voice_list(language):
    """根据选择语言返回下拉更新对象（兼容 gradio v4: 使用 gr.update）。"""
    if not language:
        return gr.update()
    names = _voice_options_lower.get(language.lower())
    if not names:
        # 没找到对应语言，返回空列表避免前端报错
        return gr.update(choices=[], value=None)
    return gr.update(choices=names, value=names[0])

with gr.Blocks() as demo:
    gr.Markdown("# Edge-TTS WebUI\n输入文本，选择语言和声音，生成语音！")
    with gr.Row():
        text_input = gr.Textbox(label="输入文本", lines=4, placeholder="请输入要合成的内容……")
    with gr.Row():
        language_dropdown = gr.Dropdown(label="语言", choices=language_options, value=language_options[0])
        voice_dropdown = gr.Dropdown(label="声音", choices=voice_options[language_options[0]], value=voice_options[language_options[0]][0])
    with gr.Row():
        generate_btn = gr.Button("生成语音 + 字幕", variant="primary")
    with gr.Row():
        audio_output = gr.Audio(label="生成的音频", type="filepath")
    subtitle_output = gr.Textbox(label="生成的字幕 (SRT)", lines=14)
    subtitle_file = gr.File(label="下载字幕文件 (.srt)")

    # 事件绑定
    language_dropdown.change(fn=update_voice_list, inputs=language_dropdown, outputs=voice_dropdown)
    generate_btn.click(fn=generate_audio_and_srt, inputs=[text_input, language_dropdown, voice_dropdown], outputs=[audio_output, subtitle_output, subtitle_file])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
