"""
M3-1 元数据解析
演示如何从视频中解析各种元数据信息，包括：
- 不同分辨率的视频格式 (1080p, 720p, 480p等)
- 音频格式
- 字幕信息
- 视频详细信息
"""

from yt_dlp import YoutubeDL
import json

def extract_video_metadata(url, cookies_path='cookies.txt'):
    """提取视频完整元数据，不下载视频"""
    import os
    parent_cookies = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cookies.txt')
    ydl_opts = {
        'cookiefile': parent_cookies,
        'quiet': True,  # 安静模式，减少输出
        'no_warnings': False,
        'extract_flat': False,  # 提取完整信息
        'extractor_args': {
            'youtube': {
                'player_client': ['tv_embedded', 'web'],
            }
        }
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        try:
            # 只提取信息，不下载
            info = ydl.extract_info(url, download=False)
            return info
        except Exception as e:
            print(f"提取元数据失败: {e}")
            return None

def analyze_video_formats(info):
    """分析视频格式信息"""
    if not info:
        return
    
    print("=" * 50)
    print("📹 视频基本信息")
    print("=" * 50)
    print(f"标题: {info.get('title', 'Unknown')}")
    print(f"上传者: {info.get('uploader', 'Unknown')}")
    print(f"时长: {info.get('duration', 0)} 秒 ({info.get('duration', 0)//60}分{info.get('duration', 0)%60}秒)")
    print(f"观看次数: {info.get('view_count', 'Unknown')}")
    print(f"上传日期: {info.get('upload_date', 'Unknown')}")
    print(f"描述: {info.get('description', 'No description')[:100]}...")
    
    formats = info.get('formats', [])
    if not formats:
        print("没有找到可用格式")
        return
    
    print("\n" + "=" * 50)
    print("🎥 视频格式分析")
    print("=" * 50)
    
    # 分离视频格式
    video_formats = []
    audio_formats = []
    
    for f in formats:
        format_id = f.get('format_id', '')
        vcodec = f.get('vcodec', 'none')
        acodec = f.get('acodec', 'none')
        
        # 跳过故事板格式
        if 'sb' in format_id or 'storyboard' in f.get('format_note', ''):
            continue
            
        if vcodec != 'none' and acodec == 'none':
            # 纯视频格式
            video_formats.append(f)
        elif vcodec == 'none' and acodec != 'none':
            # 纯音频格式
            audio_formats.append(f)
        elif vcodec != 'none' and acodec != 'none':
            # 包含视频和音频的格式
            video_formats.append(f)
    
    # 显示视频格式
    print("\n📺 可用视频格式:")
    video_formats.sort(key=lambda x: x.get('height', 0), reverse=True)
    
    for f in video_formats:
        height = f.get('height', '?')
        width = f.get('width', '?')
        fps = f.get('fps', '?')
        vcodec = f.get('vcodec', 'Unknown')
        acodec = f.get('acodec', 'none')
        filesize = f.get('filesize', 0)
        ext = f.get('ext', 'Unknown')
        
        size_mb = f"{filesize / (1024*1024):.1f}MB" if filesize else "Unknown"
        has_audio = "✅有音频" if acodec != 'none' else "❌无音频"
        
        print(f"  • 格式ID: {f.get('format_id', '?'):>3} | "
              f"{height}p ({width}x{height}) | "
              f"{fps}fps | {ext.upper()} | "
              f"{vcodec} | {has_audio} | {size_mb}")
    
    # 显示音频格式
    print("\n🎵 可用音频格式:")
    audio_formats.sort(key=lambda x: x.get('abr', 0), reverse=True)
    
    for f in audio_formats:
        acodec = f.get('acodec', 'Unknown')
        abr = f.get('abr', '?')
        asr = f.get('asr', '?')
        filesize = f.get('filesize', 0)
        ext = f.get('ext', 'Unknown')
        
        size_mb = f"{filesize / (1024*1024):.1f}MB" if filesize else "Unknown"
        
        print(f"  • 格式ID: {f.get('format_id', '?'):>3} | "
              f"{abr}kbps | {asr}Hz | "
              f"{ext.upper()} | {acodec} | {size_mb}")

def analyze_subtitles(info):
    """分析字幕信息"""
    print("\n" + "=" * 50)
    print("📝 字幕信息")
    print("=" * 50)
    
    subtitles = info.get('subtitles', {})
    automatic_captions = info.get('automatic_captions', {})
    
    if subtitles:
        print("🎯 手动字幕 (高质量):")
        for lang, subs in subtitles.items():
            print(f"  • 语言: {lang}")
            for sub in subs:
                print(f"    - 格式: {sub.get('ext', '?')} | URL: {sub.get('url', 'No URL')[:50]}...")
    else:
        print("❌ 没有手动字幕")
    
    if automatic_captions:
        print("\n🤖 自动生成字幕:")
        for lang, subs in automatic_captions.items():
            print(f"  • 语言: {lang}")
            for sub in subs[:2]:  # 只显示前2个格式
                print(f"    - 格式: {sub.get('ext', '?')} | URL: {sub.get('url', 'No URL')[:50]}...")
    else:
        print("❌ 没有自动字幕")

def get_download_options(info):
    """获取推荐的下载选项"""
    print("\n" + "=" * 50)
    print("💡 推荐下载选项")
    print("=" * 50)
    
    formats = info.get('formats', [])
    
    # 找到最高质量的视频+音频组合
    video_formats = [f for f in formats if f.get('vcodec', 'none') != 'none' and 'sb' not in f.get('format_id', '')]
    audio_formats = [f for f in formats if f.get('acodec', 'none') != 'none' and f.get('vcodec', 'none') == 'none']
    
    if video_formats and audio_formats:
        best_video = max(video_formats, key=lambda x: x.get('height', 0))
        best_audio = max(audio_formats, key=lambda x: x.get('abr', 0))
        
        print("🏆 最高质量组合:")
        print(f"  视频: 格式{best_video.get('format_id')} - {best_video.get('height')}p")
        print(f"  音频: 格式{best_audio.get('format_id')} - {best_audio.get('abr')}kbps")
        print(f"  命令: yt-dlp -f {best_video.get('format_id')}+{best_audio.get('format_id')} --merge-output-format mp4 {info.get('webpage_url', '')}")
    
    # 常用分辨率选项
    common_heights = [1080, 720, 480, 360]
    print("\n📱 常用分辨率选项:")
    
    for height in common_heights:
        suitable_formats = [f for f in video_formats if f.get('height') == height]
        if suitable_formats:
            best_format = max(suitable_formats, key=lambda x: x.get('fps', 0))
            print(f"  {height}p: 格式{best_format.get('format_id')} | "
                  f"{best_format.get('fps')}fps | "
                  f"{best_format.get('ext', '?').upper()}")
    
    # 仅音频选项
    if audio_formats:
        best_audio = max(audio_formats, key=lambda x: x.get('abr', 0))
        print(f"\n🎵 最佳音频: 格式{best_audio.get('format_id')} | "
              f"{best_audio.get('abr')}kbps | "
              f"{best_audio.get('ext', '?').upper()}")

def save_metadata_to_file(info, filename='video_metadata.json'):
    """将元数据保存到文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"\n💾 元数据已保存到: {filename}")
    except Exception as e:
        print(f"保存元数据失败: {e}")

def main():
    """主函数"""
    url = "https://youtu.be/AU9F-6uWCgE"
    
    print("🔍 正在提取视频元数据...")
    info = extract_video_metadata(url)
    
    if info:
        analyze_video_formats(info)
        analyze_subtitles(info)
        get_download_options(info)
        save_metadata_to_file(info)
    else:
        print("❌ 无法获取视频信息")

if __name__ == "__main__":
    main()
