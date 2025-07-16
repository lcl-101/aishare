"""
M4-1 接入 whisper + ollama qwen3:32b 总结字幕
- 自动下载视频元数据
- 如果有字幕，只下载字幕
- 使用 ollama 的 qwen3:32b 对字幕内容进行总结
"""

from yt_dlp import YoutubeDL
import os
import subprocess

def get_parent_cookies():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cookies.txt')

def extract_video_metadata(url):
    ydl_opts = {
        'cookiefile': get_parent_cookies(),
        'quiet': True,
        'no_warnings': False,
        'extract_flat': False,
        'extractor_args': {
            'youtube': {
                'player_client': ['tv_embedded', 'web'],
            }
        }
    }
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info
        except Exception as e:
            print(f"提取元数据失败: {e}")
            return None

def download_subtitles(url, info):
    subtitles = info.get('subtitles', {})
    if not subtitles:
        print("\n📝 没有可用字幕，跳过下载和总结。")
        return None
    print("\n📝 正在下载所有可用字幕...")
    ydl_opts = {
        'cookiefile': get_parent_cookies(),
        'skip_download': True,
        'writesubtitles': True,
        'allsubtitles': True,
        'subtitlesformat': 'srt',
        'outtmpl': 'downloads/%(title).80s.%(ext)s',
        'quiet': False,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    # 查找下载的srt字幕文件
    downloads_dir = os.path.join(os.path.dirname(__file__), 'downloads')
    srt_files = [f for f in os.listdir(downloads_dir) if f.endswith('.srt')]
    if not srt_files:
        print("❌ 没有找到下载的字幕文件")
        return None
    srt_path = os.path.join(downloads_dir, srt_files[0])
    print(f"✅ 已下载字幕: {srt_path}")
    return srt_path

def summarize_with_ollama(srt_path):
    print(f"\n🤖 正在用 ollama 的 qwen3:32b 总结字幕 ...")
    # 自动安装ollama包（如未安装）
    try:
        import ollama
    except ImportError:
        import sys, subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ollama'])
        import ollama

    # 读取并清洗SRT字幕内容，拼接为段落
    def srt_to_text(srt_content):
        import re
        lines = srt_content.splitlines()
        text_lines = []
        for line in lines:
            # 跳过序号和时间轴
            if re.match(r"^\d+$", line):
                continue
            if re.match(r"^\d{2}:\d{2}:\d{2},\d{3} --> ", line):
                continue
            if line.strip() == '':
                continue
            text_lines.append(line.strip())
        # 合并为段落，去重相邻重复
        from itertools import groupby
        merged = [k for k, _ in groupby(text_lines)]
        return ' '.join(merged)

    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    clean_text = srt_to_text(srt_content)
    # 构造 prompt
    prompt = f"""
你是一名专业的视频内容总结助手，请对下列中文视频字幕内容进行总结。

目标：让用户能在30秒内了解这期视频的核心内容。

请按以下格式输出：
1. 🎯 本期主要话题（用一句话概括主题）
2. 📌 内容要点（3-5条，每条 1 句话）
3. 🌟 精彩片段或亮点（选出最值得一提的内容，1-2条）

⚠️ 不要加入你的思考过程，不要说“我认为”或“可能”，只根据字幕原文总结。

字幕内容如下：
———
{clean_text}
———
/no_think
"""
    try:
        response = ollama.chat(
            model='qwen3:32b',
            messages=[{"role": "user", "content": prompt}]
        )
        print("\n=== 总结结果 ===\n")
        print(response['message']['content'])
    except Exception as e:
        print(f"调用 ollama 失败: {e}")

def main():
    url = "https://youtu.be/AU9F-6uWCgE"
    print("🔍 正在提取视频元数据...")
    info = extract_video_metadata(url)
    if not info:
        print("❌ 无法获取视频信息")
        return
    srt_path = download_subtitles(url, info)
    if srt_path:
        summarize_with_ollama(srt_path)

if __name__ == "__main__":
    main()
