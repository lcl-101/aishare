"""
M3-2 自动下载
根据元数据自动下载：
- 最高分辨率的视频+音频（优先1080P，其次720P、480P...）
- 最高音质的音频
- 所有可用字幕（如有）
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

def select_best_video_audio(info):
    """选择最高分辨率的视频和音频格式ID"""
    formats = info.get('formats', [])
    # 过滤出视频和音频
    video_formats = [f for f in formats if f.get('vcodec', 'none') != 'none' and 'sb' not in f.get('format_id', '')]
    audio_formats = [f for f in formats if f.get('acodec', 'none') != 'none' and f.get('vcodec', 'none') == 'none']
    # 优先级分辨率
    prefer_heights = [1080, 720, 480, 360, 240, 144]
    best_video = None
    for h in prefer_heights:
        candidates = [f for f in video_formats if f.get('height') == h]
        if candidates:
            best_video = max(candidates, key=lambda x: x.get('fps', 0))
            break
    if not best_video and video_formats:
        best_video = max(video_formats, key=lambda x: x.get('height', 0))
    best_audio = max(audio_formats, key=lambda x: x.get('abr', 0)) if audio_formats else None
    return best_video, best_audio

def download_video_audio(url, video_id, audio_id):
    print(f"\n🎬 正在下载视频+音频: {video_id}+{audio_id}")
    ydl_opts = {
        'cookiefile': get_parent_cookies(),
        'format': f'{video_id}+{audio_id}',
        'merge_output_format': 'mp4',
        'outtmpl': 'downloads/%(title).80s.%(ext)s',
        'quiet': False,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def download_audio(url, audio_id):
    print(f"\n🎵 正在下载音频: {audio_id}")
    ydl_opts = {
        'cookiefile': get_parent_cookies(),
        'format': audio_id,
        'outtmpl': 'downloads/%(title).80s.%(ext)s',
        'quiet': False,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def download_subtitles(url, info):
    subtitles = info.get('subtitles', {})
    if not subtitles:
        print("\n📝 没有可用字幕")
        return
    print("\n📝 正在下载所有可用字幕...")
    ydl_opts = {
        'cookiefile': get_parent_cookies(),
        'skip_download': True,
        'writesubtitles': True,
        'allsubtitles': True,
        'outtmpl': 'downloads/%(title).80s.%(ext)s',
        'quiet': False,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def main():
    url = "https://youtu.be/AU9F-6uWCgE"
    print("🔍 正在提取元数据...")
    info = extract_video_metadata(url)
    if not info:
        print("❌ 无法获取视频信息")
        return
    best_video, best_audio = select_best_video_audio(info)
    if best_video and best_audio:
        download_video_audio(url, best_video['format_id'], best_audio['format_id'])
    elif best_video:
        print("仅找到视频，未找到音频，尝试只下载视频...")
        download_video_audio(url, best_video['format_id'], '')
    elif best_audio:
        print("仅找到音频，未找到视频，尝试只下载音频...")
        download_audio(url, best_audio['format_id'])
    else:
        print("❌ 未找到可用的视频或音频格式")
    # 单独下载音频
    if best_audio:
        download_audio(url, best_audio['format_id'])
    # 下载字幕
    download_subtitles(url, info)

if __name__ == "__main__":
    main()
