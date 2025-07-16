#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube 视频下载器 - 快速上手示例
使用 yt-dlp 下载 YouTube 视频的基础教程

安装方式：
方式1 - 命令行安装（Ubuntu/Debian）：
    sudo add-apt-repository ppa:tomtomtom/yt-dlp
    sudo apt update && sudo apt upgrade
    sudo apt install yt-dlp -y

方式2 - Python 环境安装（推荐）：
    mkdir ytdlp && cd ytdlp
    conda create -n ytdlp python=3.10 -y
    conda activate ytdlp
    pip install -U "yt-dlp[default]"

注意事项：
- 需要有效的 cookies.txt 文件来访问某些受限制的视频
- cookies.txt 应放在项目根目录下
- 确保有足够的磁盘空间进行下载
"""

from yt_dlp import YoutubeDL
import os

def download_video_mp4(url, cookies_path='cookies.txt'):
    """
    下载 YouTube 视频为 MP4 格式
    
    参数:
        url (str): YouTube 视频链接
        cookies_path (str): cookies 文件路径（默认为 'cookies.txt'）
    
    返回:
        str: 下载文件的完整路径
    
    功能特点:
        - 自动选择最佳视频和音频质量
        - 合并为 MP4 格式
        - 支持 cookies 认证
        - 自动创建下载目录
    """
    
    # 获取上级目录中的 cookies 文件路径
    # 这样可以在多个子项目中共享同一个 cookies 文件
    parent_cookies = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cookies.txt')
    
    # yt-dlp 下载配置
    ydl_opts = {
        'format': 'bestvideo+bestaudio',          # 选择最佳视频+音频质量
        'merge_output_format': 'mp4',             # 合并输出为 MP4 格式
        'outtmpl': 'downloads/%(title).80s.%(ext)s',  # 输出文件名模板（限制标题长度80字符）
        'cookiefile': parent_cookies,             # 🔥 关键：使用 cookies 文件绕过某些限制
        'quiet': False,                           # 显示下载进度和信息
        'no_warnings': False,                     # 显示警告信息
        'extractaudio': False,                    # 不仅提取音频，保留视频
        'audioformat': 'mp3',                     # 如果需要音频格式转换
        'embed_subs': True,                       # 嵌入字幕到视频文件中
        'writesubtitles': True,                   # 下载字幕文件
        'writeautomaticsub': True,                # 下载自动生成的字幕
    }

    # 创建下载目录
    os.makedirs('downloads', exist_ok=True)
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            print(f"🎬 开始下载视频: {url}")
            
            # 先获取视频信息
            info = ydl.extract_info(url, download=False)
            print(f"📺 视频标题: {info.get('title', '未知')}")
            print(f"👤 上传者: {info.get('uploader', '未知')}")
            print(f"⏱️ 时长: {info.get('duration', 0)} 秒")
            
            # 开始下载
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            
            print(f"✅ 下载完成: {filename}")
            return filename
            
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print("💡 可能的解决方案:")
        print("   1. 检查网络连接")
        print("   2. 确认视频链接是否有效")
        print("   3. 更新 cookies.txt 文件")
        print("   4. 检查视频是否有地区限制")
        return None


def main():
    """主函数 - 示例用法"""
    # 示例视频链接（请替换为你要下载的视频链接）
    video_url = "https://youtu.be/AU9F-6uWCgE"
    
    print("🚀 YouTube 视频下载器 - 快速上手")
    print("=" * 50)
    
    # 下载视频
    result = download_video_mp4(video_url)
    
    if result:
        print(f"\n🎉 任务完成！文件已保存到: {result}")
    else:
        print(f"\n💥 下载失败，请检查错误信息并重试")


if __name__ == "__main__":
    main()
