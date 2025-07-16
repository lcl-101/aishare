"""
M4-2: 若无字幕则自动下载音频并用 whisper 转写，再用 LLM 总结
"""

from yt_dlp import YoutubeDL
import os

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
    downloads_dir = os.path.join(os.path.dirname(__file__), 'downloads')
    srt_files = [f for f in os.listdir(downloads_dir) if f.endswith('.srt')]
    if not srt_files:
        print("❌ 没有找到下载的字幕文件")
        return None
    srt_path = os.path.join(downloads_dir, srt_files[0])
    print(f"✅ 已下载字幕: {srt_path}")
    return srt_path

def download_audio(url):
    print("\n🔊 没有字幕，正在下载音频...")
    ydl_opts = {
        'cookiefile': get_parent_cookies(),
        'format': 'bestaudio/best',
        'outtmpl': 'downloads/%(title).80s.%(ext)s',
        'quiet': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    downloads_dir = os.path.join(os.path.dirname(__file__), 'downloads')
    audio_files = [f for f in os.listdir(downloads_dir) if f.endswith('.mp3')]
    if not audio_files:
        print("❌ 没有找到下载的音频文件")
        return None
    audio_path = os.path.join(downloads_dir, audio_files[0])
    print(f"✅ 已下载音频: {audio_path}")
    return audio_path

def transcribe_with_whisper(audio_path):
    print("\n📝 正在用 whisper HTTP 接口转写音频 ...")
    import subprocess
    import shutil

    # 1. 转为 wav 格式
    wav_path = audio_path.rsplit('.', 1)[0] + ".wav"
    if not os.path.exists(wav_path):
        print(f"🔄 正在转换为 wav 格式: {wav_path}")
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path
        ], check=True)
    else:
        print(f"已存在 wav 文件: {wav_path}")

    # 2. 调用 whisper HTTP 接口
    srt_path = audio_path.rsplit('.', 1)[0] + ".srt"
    print("⏳ 正在调用 whisper HTTP 服务 ...")
    curl_cmd = [
        "curl", "-X", "POST", "http://127.0.0.1:8080/transcribe/",
        "-F", f"audio_file=@{wav_path};type=audio/wav",
        "-F", "model_name=large-v3",
        "-F", "language=zh"
    ]
    try:
        result = subprocess.run(curl_cmd, capture_output=True, check=True)
        # 假设接口直接返回 SRT 字符串
        srt_content = result.stdout.decode("utf-8")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        print(f"✅ Whisper HTTP 转写完成: {srt_path}")
        return srt_path
    except subprocess.CalledProcessError as e:
        print("❌ Whisper HTTP 转写失败")
        print(e.stderr.decode("utf-8"))
        return None

def summarize_with_ollama(srt_path):
    print(f"\n🤖 正在用 ollama 的 qwen3:32b 总结字幕 ...")
    try:
        import ollama
    except ImportError:
        import sys, subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ollama'])
        import ollama

    def srt_to_text(srt_content):
        import re
        lines = srt_content.splitlines()
        text_lines = []
        for line in lines:
            if re.match(r"^\d+$", line):
                continue
            if re.match(r"^\d{2}:\d{2}:\d{2},\d{3} --> ", line):
                continue
            if line.strip() == '':
                continue
            text_lines.append(line.strip())
        from itertools import groupby
        merged = [k for k, _ in groupby(text_lines)]
        return ' '.join(merged)

    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    clean_text = srt_to_text(srt_content)
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
    url = "https://www.youtube.com/watch?v=djhQyuR79sI"
    print("🔍 正在提取视频元数据...")
    info = extract_video_metadata(url)
    if not info:
        print("❌ 无法获取视频信息")
        return
    srt_path = download_subtitles(url, info)
    if srt_path:
        summarize_with_ollama(srt_path)
    else:
        audio_path = download_audio(url)
        if not audio_path:
            print("❌ 无法下载音频")
            return
        srt_path = transcribe_with_whisper(audio_path)
        summarize_with_ollama(srt_path)

if __name__ == "__main__":
    main()
