"""
M5-2: YouTube 内容处理器 (TypeError 修复最终版)

本脚本作为一个极其健壮的客户端，执行以下工作流：
1.  接收一个 YouTube 视频链接。
2.  使用 `yt-dlp` 并加载 Cookies、伪装 User-Agent 来下载音频，修复了 `TypeError`。
3.  使用 `requests` 将下载的音频文件发送到 M5-1 Whisper API 服务。
4.  接收 API 返回的转录全文和 SRT 字幕。
5.  使用 `ollama` 将全文发送给本地 LLM 进行摘要。
6.  保存所有产出物到文件。

如何运行:
1.  准备好 `youtube_cookies.txt` 文件并与本脚本放在同一目录。
2.  确保 M5-1-API.py 服务和 Ollama 服务正在运行。
3.  安装所有依赖: pip install yt-dlp requests ollama ffmpeg-python
4.  修改下面的配置区，然后运行此脚本: python M5-2-Youtube.py
"""
import os
import requests
import ollama
import yt_dlp
import re
import datetime
import sys
import logging

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 核心功能函数 ---

def download_youtube_audio(url, output_path=".", cookie_file="youtube_cookies.txt"):
    """
    最终修复版：使用 yt-dlp、手动导出的 Cookies 文件和伪装的 User-Agent 下载。
    修复了 'string indices must be integers' 的 TypeError。
    """
    os.makedirs(output_path, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cookie_file_path = os.path.join(script_dir, cookie_file)
    
    if not os.path.exists(cookie_file_path):
        logging.error(f"❌ Cookies 文件未找到: {cookie_file_path}")
        logging.error("请使用浏览器扩展导出 Netscape 格式的 Cookies，并保存为 'youtube_cookies.txt'。")
        return None, None
    
    logging.info(f"🍪 正在从文件加载 Cookies: {cookie_file_path}")

    # 步骤 1: 先获取视频标题
    try:
        # 使用一个临时的、安静的 ydl 实例只为了获取信息
        with yt_dlp.YoutubeDL({'quiet': True, 'logtostderr': False, 'cookiefile': cookie_file_path}) as ydl_info:
            info_dict = ydl_info.extract_info(url, download=False)
            video_title = info_dict.get('title', "untitled")
            logging.info(f"📄 视频标题: {video_title}")
    except Exception as e:
        logging.error(f"❌ 获取视频信息失败: {e}", exc_info=True)
        return None, None
        
    # 步骤 2: 根据标题构建安全的文件名和最终路径
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", video_title)
    final_audio_path = os.path.join(output_path, f"{safe_title}.mp3")

    # 步骤 3: 构建完整的 ydl_opts 用于下载
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': os.path.splitext(final_audio_path)[0], # 提供不带扩展名的最终路径
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'cookiefile': cookie_file_path,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'noplaylist': True,
        'quiet': False,
        'logtostderr': True,
        'overwrites': True, # 如果文件已存在则覆盖
    }

    try:
        logging.info(f"🔗 正在使用 yt-dlp (带 User-Agent) 连接到 YouTube: {url}")
        logging.info("📥 正在下载并转换为 mp3...")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if os.path.exists(final_audio_path):
            logging.info(f"✅ 音频下载并转换完成 -> {os.path.basename(final_audio_path)}")
            return final_audio_path, video_title
        else:
            logging.error("❌ yt-dlp 处理后未能找到输出的 mp3 文件。")
            return None, None

    except Exception as e:
        logging.error(f"❌ 使用 yt-dlp 下载时发生严重错误: {e}", exc_info=True)
        return None, None

def transcribe_via_api(api_url, audio_path, model_name="base", language=None):
    """通过调用 M5-1 API 来转录音频。"""
    logging.info(f"📤 正在将音频发送到 Whisper API: {api_url}")
    
    try:
        files = {'audio_file': (os.path.basename(audio_path), open(audio_path, 'rb'), 'audio/mpeg')}
        data = {'model_name': model_name}
        if language:
            data['language'] = language
            
        response = requests.post(api_url, files=files, data=data)
        response.raise_for_status()
        
        logging.info("✅ API 转录成功。")
        return response.json()
        
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ 调用 Whisper API 时发生网络错误: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"❌ 处理 API 响应时出错: {e}", exc_info=True)
        return None

def get_summary_from_llm(full_text, llm_model_name):
    """使用 Ollama 的 LLM 生成文本摘要。"""
    if not full_text or not full_text.strip():
        logging.warning("⚠️ 文本内容为空，无法生成摘要。")
        return "原文为空，无法生成摘要。"
        
    logging.info(f"🤖 正在调用 LLM '{llm_model_name}' 进行摘要提取...")
    prompt = f"""请为以下文章生成一个简洁、重点突出、长度在500字左右的摘要。\n\n文章内容：\n---\n{full_text}\n---\n\n摘要：/no_think"""
    
    try:
        response = ollama.chat(
            model=llm_model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        summary = response['message']['content']
        logging.info("✅ LLM 摘要提取完成。")
        return summary
    except Exception as e:
        logging.error(f"❌ 调用 Ollama API 时出错: {e}", exc_info=True)
        return "摘要生成失败。"

def save_output_files(output_dir, base_filename, full_text, srt_subtitles, summary_text):
    """将所有产出内容保存到文件。"""
    logging.info("💾 正在保存所有结果文件...")
    safe_base_filename = re.sub(r'[\\/*?:"<>|]', "_", base_filename)
    
    paths = {
        "full_text": os.path.join(output_dir, f"{safe_base_filename}_full_text.txt"),
        "srt": os.path.join(output_dir, f"{safe_base_filename}_subtitles.srt"),
        "summary": os.path.join(output_dir, f"{safe_base_filename}_summary.txt"),
    }
    
    with open(paths["full_text"], 'w', encoding='utf-8') as f: f.write(full_text)
    with open(paths["srt"], 'w', encoding='utf-8') as f: f.write(srt_subtitles)
    with open(paths["summary"], 'w', encoding='utf-8') as f: f.write(summary_text)
    
    logging.info("✅ 所有文件保存完毕。")
    for key, path in paths.items():
        logging.info(f"   - {key.capitalize()}: {os.path.abspath(path)}")

# --- 2. 主控函数 ---
def main(config):
    """执行完整的主流程。"""
    total_start_time = datetime.datetime.now()
    logging.info("🚀 开始 M5-2 YouTube 内容处理客户端流程 (TypeError 修复最终版)")
    print("="*60)

    # 步骤 1: 下载 YouTube 音频
    logging.info("--- 步骤 1: 下载音频 ---")
    audio_path, video_title = download_youtube_audio(
        config['youtube_url'], 
        output_path=config['output_dir'],
        cookie_file=config['cookie_file']
    )
    if not audio_path:
        logging.critical("下载失败，流程终止。")
        sys.exit(1)

    # 步骤 2: 通过 API 转录
    logging.info("\n--- 步骤 2: 调用 API 进行转录 ---")
    transcription_result = transcribe_via_api(
        config['whisper_api_url'],
        audio_path,
        config['whisper_model'],
        config['language']
    )
    if not transcription_result:
        logging.critical("API 转录失败，流程终止。")
        sys.exit(1)
        
    full_text = transcription_result.get("full_text", "")
    srt_subtitles = transcription_result.get("srt_subtitles", "")

    # 步骤 3: LLM 摘要
    logging.info("\n--- 步骤 3: LLM 提取摘要 ---")
    summary_text = get_summary_from_llm(full_text, config['llm_model'])
    
    # 步骤 4: 保存文件
    logging.info("\n--- 步骤 4: 保存文件 ---")
    save_output_files(config['output_dir'], video_title, full_text, srt_subtitles, summary_text)

    # 步骤 5: 清理临时文件
    if not config['keep_audio_file']:
        logging.info("\n--- 步骤 5: 清理临时文件 ---")
        try:
            os.remove(audio_path)
            logging.info(f"🗑️ 已删除临时音频文件: {os.path.basename(audio_path)}")
        except OSError as e:
            logging.warning(f"⚠️ 删除临时文件失败: {e}")
            
    total_end_time = datetime.datetime.now()
    total_duration = total_end_time - total_start_time
    
    print("="*60)
    logging.info(f"🎉 全部流程完成！总耗时: {total_duration}")
    print("="*60)

# --- 3. 配置与执行 ---
if __name__ == "__main__":
    # --- 用户配置区 ---
    CONFIG = {
        "youtube_url": "https://www.youtube.com/watch?v=SO6Vhb5XM0w",
        
        "cookie_file": "youtube_cookies.txt",
        
        "whisper_api_url": "http://127.0.0.1:8080/transcribe/",
        "whisper_model": "large-v3",
        "llm_model": "qwen3:32b",
        "language": "zh",
        
        "output_dir": "M5_outputs",
        "keep_audio_file": False,
    }
    
    # --- 程序执行区 ---
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
        
    main(CONFIG)