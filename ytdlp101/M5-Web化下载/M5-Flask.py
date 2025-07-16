import os
import json
import uuid
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import yt_dlp
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# 配置目录
DOWNLOAD_DIR = Path('downloads')
TEMP_DIR = Path('temp')
DOWNLOAD_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# 全局变量存储下载进度
download_progress = {}

class DownloadProgressHook:
    def __init__(self, task_id):
        self.task_id = task_id
        
    def __call__(self, d):
        global download_progress
        if d['status'] == 'downloading':
            try:
                # 计算下载进度
                if 'total_bytes' in d:
                    percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
                elif 'total_bytes_estimate' in d:
                    percent = (d['downloaded_bytes'] / d['total_bytes_estimate']) * 100
                else:
                    percent = 0
                
                speed = d.get('speed', 0)
                eta = d.get('eta', 0)
                
                download_progress[self.task_id] = {
                    'status': 'downloading',
                    'percent': round(percent, 2),
                    'speed': speed,
                    'eta': eta,
                    'downloaded_bytes': d.get('downloaded_bytes', 0),
                    'total_bytes': d.get('total_bytes', d.get('total_bytes_estimate', 0))
                }
            except Exception as e:
                print(f"Progress hook error: {e}")
                
        elif d['status'] == 'finished':
            download_progress[self.task_id] = {
                'status': 'finished',
                'percent': 100,
                'filename': d['filename']
            }

def get_cookies_path():
    """获取 cookies 文件路径"""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cookies.txt')

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/extract_info', methods=['POST'])
def extract_info():
    """提取视频信息"""
    try:
        url = request.json.get('url')
        if not url:
            return jsonify({'error': '请输入有效的 YouTube 链接'}), 400
        
        # 配置 yt-dlp 选项
        ydl_opts = {
            'cookiefile': get_cookies_path(),
            'quiet': True,
            'no_warnings': False,
            'extract_flat': False,
            'extractor_args': {
                'youtube': {
                    'player_client': ['tv_embedded', 'web'],
                }
            }
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # 提取基本信息
            video_info = {
                'title': info.get('title', '未知标题'),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'uploader': info.get('uploader', '未知上传者'),
                'upload_date': info.get('upload_date', ''),
                'description': info.get('description', '')[:500] + '...' if info.get('description') else '',
                'thumbnail': info.get('thumbnail', ''),
                'formats': []
            }
            
            # 处理格式信息 - 修复格式筛选逻辑
            video_formats = []
            audio_formats = []
            
            for fmt in info.get('formats', []):
                format_id = fmt.get('format_id', '')
                ext = fmt.get('ext', '')
                quality = fmt.get('format_note', fmt.get('quality', ''))
                filesize = fmt.get('filesize', 0)
                height = fmt.get('height', 0)
                width = fmt.get('width', 0)
                
                # 跳过无用格式
                if ext in ['mhtml'] or 'storyboard' in quality.lower():
                    continue
                
                # 判断格式类型
                if fmt.get('vcodec') != 'none' and fmt.get('acodec') != 'none':
                    # 视频+音频格式（复合格式）
                    format_type = 'video'
                    resolution = f"{width}x{height}" if width and height else fmt.get('resolution', '')
                    
                    format_info = {
                        'format_id': format_id,
                        'type': format_type,
                        'ext': ext,
                        'quality': quality,
                        'filesize': filesize,
                        'filesize_mb': round(filesize / 1024 / 1024, 2) if filesize else 0,
                        'resolution': resolution,
                        'fps': fmt.get('fps', ''),
                        'height': height,
                        'vbr': fmt.get('vbr', ''),
                        'abr': fmt.get('abr', '')
                    }
                    video_formats.append(format_info)
                    
                elif fmt.get('vcodec') != 'none' and fmt.get('acodec') == 'none':
                    # 仅视频格式（分离格式）
                    format_type = 'video'
                    resolution = f"{width}x{height}" if width and height else fmt.get('resolution', '')
                    
                    format_info = {
                        'format_id': format_id,
                        'type': format_type,
                        'ext': ext,
                        'quality': quality,
                        'filesize': filesize,
                        'filesize_mb': round(filesize / 1024 / 1024, 2) if filesize else 0,
                        'resolution': resolution,
                        'fps': fmt.get('fps', ''),
                        'height': height,
                        'vbr': fmt.get('vbr', ''),
                        'abr': ''
                    }
                    video_formats.append(format_info)
                    
                elif fmt.get('vcodec') == 'none' and fmt.get('acodec') != 'none':
                    # 仅音频格式
                    format_type = 'audio'
                    
                    format_info = {
                        'format_id': format_id,
                        'type': format_type,
                        'ext': ext,
                        'quality': quality,
                        'filesize': filesize,
                        'filesize_mb': round(filesize / 1024 / 1024, 2) if filesize else 0,
                        'abr': fmt.get('abr', ''),
                        'acodec': fmt.get('acodec', '')
                    }
                    audio_formats.append(format_info)
            
            # 去重并排序视频格式（按分辨率降序，为每个分辨率保留最佳格式）
            resolution_best = {}
            for fmt in video_formats:
                height = fmt.get('height', 0)
                if height > 0:  # 忽略无效高度
                    # 为每个分辨率选择最佳格式（优先mp4，其次webm）
                    if height not in resolution_best:
                        resolution_best[height] = fmt
                    else:
                        current = resolution_best[height]
                        # 优先选择mp4格式，其次是文件大小更小的
                        if (fmt['ext'] == 'mp4' and current['ext'] != 'mp4') or \
                           (fmt['ext'] == current['ext'] and fmt.get('filesize', 0) < current.get('filesize', float('inf'))):
                            resolution_best[height] = fmt
            
            # 按分辨率降序排列
            unique_video_formats = [resolution_best[height] for height in sorted(resolution_best.keys(), reverse=True)]
            
            # 去重并排序音频格式（按比特率降序）
            seen_audio = set()
            unique_audio_formats = []
            for fmt in sorted(audio_formats, key=lambda x: x.get('abr', 0), reverse=True):
                key = f"{fmt.get('abr', 0)}_{fmt['ext']}"
                if key not in seen_audio:
                    seen_audio.add(key)
                    unique_audio_formats.append(fmt)
            
            # 合并所有格式
            video_info['formats'] = unique_video_formats + unique_audio_formats
            
            # 检查字幕 - 去重处理
            subtitles = []
            seen_languages = set()
            
            # 处理普通字幕
            for lang, subs in info.get('subtitles', {}).items():
                if lang not in seen_languages and subs:
                    # 优先选择srt格式，其次vtt
                    best_sub = None
                    for sub in subs:
                        if sub.get('ext') == 'srt':
                            best_sub = sub
                            break
                        elif sub.get('ext') == 'vtt' and not best_sub:
                            best_sub = sub
                        elif not best_sub:
                            best_sub = sub
                    
                    if best_sub:
                        subtitles.append({
                            'language': lang,
                            'ext': best_sub.get('ext', 'srt'),
                            'url': best_sub.get('url', '')
                        })
                        seen_languages.add(lang)
            
            # 处理自动字幕（仅添加没有普通字幕的语言）
            for lang, subs in info.get('automatic_captions', {}).items():
                auto_lang = f"{lang} (自动)"
                if lang not in seen_languages and subs:
                    # 同样优先选择srt格式
                    best_sub = None
                    for sub in subs:
                        if sub.get('ext') == 'srt':
                            best_sub = sub
                            break
                        elif sub.get('ext') == 'vtt' and not best_sub:
                            best_sub = sub
                        elif not best_sub:
                            best_sub = sub
                    
                    if best_sub:
                        subtitles.append({
                            'language': auto_lang,
                            'ext': best_sub.get('ext', 'srt'),
                            'url': best_sub.get('url', '')
                        })
            
            video_info['subtitles'] = subtitles[:10]  # 限制字幕数量
            
            return jsonify(video_info)
            
    except Exception as e:
        return jsonify({'error': f'获取视频信息失败: {str(e)}'}), 500

@app.route('/download', methods=['POST'])
def download():
    """开始下载"""
    try:
        data = request.json
        url = data.get('url')
        format_id = data.get('format_id')
        download_type = data.get('type', 'video')
        
        if not url or not format_id:
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 生成任务 ID
        task_id = str(uuid.uuid4())
        
        # 初始化进度
        download_progress[task_id] = {
            'status': 'starting',
            'percent': 0
        }
        
        # 在后台线程中开始下载
        thread = threading.Thread(target=download_video, args=(url, format_id, download_type, task_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
        
    except Exception as e:
        return jsonify({'error': f'下载启动失败: {str(e)}'}), 500

def download_video(url, format_id, download_type, task_id):
    """下载视频的后台函数"""
    try:
        # 获取视频信息以获取标题
        info_opts = {
            'cookiefile': get_cookies_path(),
            'quiet': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['tv_embedded', 'web'],
                }
            }
        }
        
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            safe_title = secure_filename(info.get('title', 'video'))[:50]
        
        # 配置下载选项
        output_filename = f"{safe_title}_{task_id}.%(ext)s"
        output_path = DOWNLOAD_DIR / output_filename
        
        ydl_opts = {
            'cookiefile': get_cookies_path(),
            'format': format_id,
            'outtmpl': str(output_path),
            'progress_hooks': [DownloadProgressHook(task_id)],
            'extractor_args': {
                'youtube': {
                    'player_client': ['tv_embedded', 'web'],
                }
            }
        }
        
        # 如果是音频，添加后处理器
        if download_type == 'audio':
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # 查找实际下载的文件
        downloaded_file = None
        for file in DOWNLOAD_DIR.glob(f"{safe_title}_{task_id}.*"):
            downloaded_file = file
            break
        
        if downloaded_file:
            download_progress[task_id] = {
                'status': 'completed',
                'percent': 100,
                'filename': downloaded_file.name,
                'download_url': f'/download_file/{downloaded_file.name}'
            }
        else:
            download_progress[task_id] = {
                'status': 'error',
                'percent': 0,
                'error': '找不到下载的文件'
            }
            
    except Exception as e:
        download_progress[task_id] = {
            'status': 'error',
            'percent': 0,
            'error': str(e)
        }

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """获取下载进度"""
    return jsonify(download_progress.get(task_id, {'status': 'not_found'}))

@app.route('/download_file/<filename>')
def download_file(filename):
    """提供文件下载"""
    try:
        file_path = DOWNLOAD_DIR / filename
        if file_path.exists():
            return send_file(file_path, as_attachment=True)
        else:
            return "文件不存在", 404
    except Exception as e:
        return f"下载失败: {str(e)}", 500

@app.route('/subtitle_download', methods=['POST'])
def download_subtitle():
    """下载字幕"""
    try:
        data = request.json
        url = data.get('url')
        language = data.get('language')
        
        if not url or not language:
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 生成任务 ID
        task_id = str(uuid.uuid4())
        
        # 下载字幕
        ydl_opts = {
            'cookiefile': get_cookies_path(),
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': [language.replace(' (自动)', '')],
            'subtitlesformat': 'srt',
            'skip_download': True,
            'outtmpl': str(DOWNLOAD_DIR / f'subtitle_{task_id}.%(ext)s'),
            'extractor_args': {
                'youtube': {
                    'player_client': ['tv_embedded', 'web'],
                }
            }
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # 查找下载的字幕文件
        subtitle_file = None
        for file in DOWNLOAD_DIR.glob(f'subtitle_{task_id}.*'):
            subtitle_file = file
            break
        
        if subtitle_file:
            return jsonify({
                'success': True,
                'download_url': f'/download_file/{subtitle_file.name}'
            })
        else:
            return jsonify({'error': '字幕下载失败'}), 500
            
    except Exception as e:
        return jsonify({'error': f'字幕下载失败: {str(e)}'}), 500

@app.route('/debug_formats', methods=['POST'])
def debug_formats():
    """调试接口：查看所有可用格式"""
    try:
        url = request.json.get('url')
        if not url:
            return jsonify({'error': '请输入有效的 YouTube 链接'}), 400
        
        ydl_opts = {
            'cookiefile': get_cookies_path(),
            'quiet': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['tv_embedded', 'web'],
                }
            }
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # 返回所有原始格式信息用于调试
            all_formats = []
            for fmt in info.get('formats', []):
                format_debug = {
                    'format_id': fmt.get('format_id'),
                    'ext': fmt.get('ext'),
                    'quality': fmt.get('format_note', fmt.get('quality', '')),
                    'resolution': fmt.get('resolution', ''),
                    'height': fmt.get('height', 0),
                    'width': fmt.get('width', 0),
                    'fps': fmt.get('fps', ''),
                    'vcodec': fmt.get('vcodec', ''),
                    'acodec': fmt.get('acodec', ''),
                    'filesize': fmt.get('filesize', 0),
                    'abr': fmt.get('abr', ''),
                    'vbr': fmt.get('vbr', ''),
                    'protocol': fmt.get('protocol', ''),
                    'format_note': fmt.get('format_note', ''),
                }
                all_formats.append(format_debug)
            
            return jsonify({
                'title': info.get('title', ''),
                'total_formats': len(all_formats),
                'all_formats': all_formats
            })
            
    except Exception as e:
        return jsonify({'error': f'调试失败: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 启动 YouTube 下载器 Web 服务")
    print("📡 访问地址: http://localhost:7860")
    print("🔧 调试模式: 已启用")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=7860)
