import asyncio
import json
import os
import re
import subprocess
import yt_dlp
from pathlib import Path
from typing import List, Dict, Optional

# 检查并安装 crawl4ai
try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    import sys, subprocess
    print("📦 正在安装 crawl4ai...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'crawl4ai'])
    from crawl4ai import AsyncWebCrawler

class YouTubeVideoAnalyzer:
    def __init__(self, download_dir="downloads", summaries_dir="summaries"):
        self.download_dir = Path(download_dir)
        self.summaries_dir = Path(summaries_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.summaries_dir.mkdir(exist_ok=True)
        
        # 初始化 ollama
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            import sys, subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ollama'])
            import ollama
            self.ollama = ollama
    
    def get_parent_cookies(self):
        """获取 cookies 文件路径"""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cookies.txt')

    async def search_youtube_videos(self, keyword: str, max_videos: int = 100) -> List[Dict]:
        """搜索 YouTube 视频并解析结果"""
        search_url = f"https://www.youtube.com/results?search_query={keyword}"
        
        async with AsyncWebCrawler() as crawler:
            # 使用 JavaScript 来滚动页面加载更多内容
            js_code = """
            async function loadMoreVideos() {
                let videos = [];
                let lastHeight = 0;
                let scrollAttempts = 0;
                const maxScrolls = 10; // 最多滚动10次
                
                while (scrollAttempts < maxScrolls) {
                    // 滚动到页面底部
                    window.scrollTo(0, document.body.scrollHeight);
                    
                    // 等待内容加载
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    
                    let newHeight = document.body.scrollHeight;
                    
                    // 如果页面高度没有变化，说明没有更多内容了
                    if (newHeight === lastHeight) {
                        scrollAttempts++;
                    } else {
                        scrollAttempts = 0; // 重置计数
                    }
                    
                    lastHeight = newHeight;
                    
                    // 检查当前页面的视频数量
                    const videoElements = document.querySelectorAll('ytd-video-renderer, ytd-compact-video-renderer');
                    if (videoElements.length >= 80) { // 获取足够的视频就停止
                        break;
                    }
                }
                
                return document.documentElement.outerHTML;
            }
            
            return await loadMoreVideos();
            """
            
            result = await crawler.arun(
                url=search_url,
                js_code=js_code,
                wait_for="css:ytd-video-renderer"
            )
            html_content = result.html
        
        # 解析视频信息
        videos = self._parse_video_info(html_content)
        
        print(f"从搜索结果中解析到 {len(videos)} 个视频")
        
        # 返回所有解析到的视频，不在这里过滤
        return videos[:max_videos]  # 最多返回指定数量

    def _parse_video_info(self, html_content: str) -> List[Dict]:
        """从 HTML 中解析视频信息"""
        videos = []
        
        # 多种正则表达式模式来提取视频信息
        patterns = [
            # 模式1: 标准的搜索结果格式
            r'"videoId":"([^"]+)".*?"title":{"runs":\[{"text":"([^"]+)"}.*?"viewCountText":{"simpleText":"([^"]+)"}',
            # 模式2: 另一种可能的格式
            r'"videoId":"([^"]+)".*?"title":{"accessibility":{"accessibilityData":{"label":"([^"]+)"}}.*?"viewCountText":{"simpleText":"([^"]+)"}',
            # 模式3: 简化模式
            r'"videoId":"([^"]+)".*?"text":"([^"]+)".*?"viewCountText".*?"simpleText":"([^"]+)"',
        ]
        
        # 尝试所有模式
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, html_content, re.DOTALL)
            all_matches.extend(matches)
        
        # 去重处理
        seen_video_ids = set()
        for match in all_matches:
            if len(match) >= 3:
                video_id, title, view_text = match[0], match[1], match[2]
                
                # 去重
                if video_id in seen_video_ids:
                    continue
                seen_video_ids.add(video_id)
                
                # 清理标题中可能的HTML标签和特殊字符
                title = re.sub(r'<[^>]+>', '', title)
                title = title.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                
                view_count = self._parse_view_count(view_text)
                
                video_info = {
                    'video_id': video_id,
                    'title': title,
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'view_count': view_count,
                    'view_text': view_text
                }
                videos.append(video_info)
        
        # 如果正则表达式失败，尝试备用方法
        if len(videos) < 10:
            print("⚠️ 正则表达式解析结果较少，尝试备用方法...")
            backup_videos = self._parse_video_info_backup(html_content)
            videos.extend(backup_videos)
            
            # 再次去重
            unique_videos = {}
            for video in videos:
                if video['video_id'] not in unique_videos:
                    unique_videos[video['video_id']] = video
            videos = list(unique_videos.values())
        
        return videos
    
    def _parse_video_info_backup(self, html_content: str) -> List[Dict]:
        """备用的视频信息解析方法"""
        videos = []
        
        # 查找所有的videoId
        video_ids = re.findall(r'"videoId":"([^"]+)"', html_content)
        
        # 查找所有可能的标题
        titles = re.findall(r'"title":{"runs":\[{"text":"([^"]+)"}', html_content)
        titles.extend(re.findall(r'"text":"([^"]+(?:鱿鱼游戏|squid game)[^"]*)"', html_content, re.IGNORECASE))
        
        # 查找观看次数
        view_counts = re.findall(r'"viewCountText":{"simpleText":"([^"]+)"}', html_content)
        
        # 尝试匹配
        min_length = min(len(video_ids), len(titles), len(view_counts))
        for i in range(min_length):
            video_info = {
                'video_id': video_ids[i],
                'title': titles[i],
                'url': f'https://www.youtube.com/watch?v={video_ids[i]}',
                'view_count': self._parse_view_count(view_counts[i]),
                'view_text': view_counts[i]
            }
            videos.append(video_info)
        
        return videos

    def _parse_view_count(self, view_text: str) -> int:
        """解析观看次数文本为数字"""
        # 移除逗号和空格
        view_text = view_text.replace(',', '').replace(' ', '').lower()
        
        # 提取数字
        number_match = re.search(r'(\d+(?:\.\d+)?)', view_text)
        if not number_match:
            return 0
        
        number = float(number_match.group(1))
        
        # 处理单位 (万, k, m, 等)
        if '万' in view_text:
            number *= 10000
        elif 'k' in view_text:
            number *= 1000
        elif 'm' in view_text:
            number *= 1000000
        elif 'b' in view_text:
            number *= 1000000000
        
        return int(number)

    def download_video_info(self, video_url: str) -> Dict:
        """使用 yt-dlp 获取视频信息"""
        ydl_opts = {
            'cookiefile': self.get_parent_cookies(),
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
            try:
                info = ydl.extract_info(video_url, download=False)
                return info
            except Exception as e:
                print(f"获取视频信息失败: {e}")
                return {}

    def download_video_with_subtitles(self, video_url: str, video_id: str, video_title: str = "") -> Dict:
        """下载视频和字幕"""
        # 创建以视频标题命名的目录
        safe_title = self._clean_filename(video_title) if video_title else video_id
        output_path = self.download_dir / f"{video_id}_{safe_title}"
        output_path.mkdir(exist_ok=True)
        
        # 尝试下载字幕
        subtitle_opts = {
            'cookiefile': self.get_parent_cookies(),
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['zh-Hans', 'zh', 'en'],
            'subtitlesformat': 'srt',
            'skip_download': True,
            'outtmpl': str(output_path / '%(title)s.%(ext)s'),
            'extractor_args': {
                'youtube': {
                    'player_client': ['tv_embedded', 'web'],
                }
            }
        }
        
        subtitle_downloaded = False
        subtitle_file = None
        
        with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
            try:
                ydl.download([video_url])
                # 查找下载的字幕文件 (SRT 或 VTT)
                for ext in ['*.srt', '*.vtt']:
                    for file in output_path.glob(ext):
                        subtitle_file = file
                        subtitle_downloaded = True
                        break
                    if subtitle_downloaded:
                        break
            except Exception as e:
                print(f"下载字幕失败: {e}")
        
        # 如果没有字幕，下载音频
        audio_file = None
        if not subtitle_downloaded:
            audio_opts = {
                'cookiefile': self.get_parent_cookies(),
                'format': 'bestaudio/best',
                'outtmpl': str(output_path / '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'extractor_args': {
                    'youtube': {
                        'player_client': ['tv_embedded', 'web'],
                    }
                }
            }
            
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                try:
                    ydl.download([video_url])
                    # 查找下载的音频文件
                    for file in output_path.glob("*.mp3"):
                        audio_file = file
                        break
                except Exception as e:
                    print(f"下载音频失败: {e}")
        
        return {
            'subtitle_file': subtitle_file,
            'audio_file': audio_file,
            'subtitle_downloaded': subtitle_downloaded,
            'output_path': output_path  # 返回输出目录路径
        }

    def extract_text_from_subtitle(self, subtitle_file: Path) -> str:
        """从字幕文件中提取文本"""
        try:
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 判断是 VTT 还是 SRT 格式
            if subtitle_file.suffix.lower() == '.srt':
                text_content = self._srt_to_text(content)
            else:
                text_content = self._vtt_to_text(content)
            
            # 保存提取的字幕文本到视频目录
            subtitle_text_file = subtitle_file.parent / "subtitle_text.txt"
            with open(subtitle_text_file, 'w', encoding='utf-8') as f:
                f.write("=== 字幕文本 ===\n")
                f.write(f"字幕文件: {subtitle_file.name}\n")
                f.write("=" * 50 + "\n")
                f.write(text_content)
            print(f"💾 字幕文本已保存到: {subtitle_text_file}")
            
            return text_content
                
        except Exception as e:
            print(f"提取字幕文本失败: {e}")
            return ""
    
    def _srt_to_text(self, srt_content: str) -> str:
        """处理 SRT 格式字幕"""
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
    
    def _vtt_to_text(self, vtt_content: str) -> str:
        """处理 VTT 格式字幕"""
        lines = vtt_content.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # 跳过时间戳行和空行
            if not line or '-->' in line or line.startswith('WEBVTT') or line.isdigit():
                continue
            # 移除 HTML 标签
            line = re.sub(r'<[^>]+>', '', line)
            if line:
                text_lines.append(line)
        
        return ' '.join(text_lines)

    def transcribe_audio(self, audio_file: Path) -> str:
        """使用本地 whisper HTTP 服务转录音频"""
        try:
            # 直接使用 MP3 文件，不需要转换为 WAV
            print(f"� 使用音频文件: {audio_file}")

            # 调用 whisper HTTP 接口，直接使用 MP3
            print("⏳ 正在调用 whisper HTTP 服务 ...")
            curl_cmd = [
                "curl", "-X", "POST", "http://127.0.0.1:8080/transcribe/",
                "-F", f"audio_file=@{str(audio_file)};type=audio/mpeg",
                "-F", "model_name=large-v3",
                "-F", "language=zh",
                "--connect-timeout", "30",    # 连接超时30秒
                "--max-time", "900"           # 总超时15分钟（900秒）
            ]
            
            result = subprocess.run(curl_cmd, capture_output=True, check=True)
            # Whisper HTTP 服务返回 JSON 格式
            response_text = result.stdout.decode("utf-8")
            
            try:
                # 解析 JSON 响应
                response_data = json.loads(response_text)
                
                if response_data.get("success"):
                    # 优先使用 srt_subtitles，如果没有则使用 full_text
                    if "srt_subtitles" in response_data:
                        srt_content = response_data["srt_subtitles"]
                        transcribed_text = self._srt_to_text(srt_content)
                    else:
                        transcribed_text = response_data.get("full_text", "")
                else:
                    print("❌ Whisper 服务返回错误")
                    return ""
                    
            except json.JSONDecodeError:
                # 如果不是 JSON，尝试直接作为 SRT 处理（向后兼容）
                print("⚠️ 响应不是 JSON 格式，尝试直接解析为 SRT")
                transcribed_text = self._srt_to_text(response_text)
            
            # 保存转写文本到视频目录
            transcript_file = audio_file.parent / "transcript.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write("=== 音频转写文本 ===\n")
                f.write(f"音频文件: {audio_file.name}\n")
                f.write("=" * 50 + "\n")
                f.write(transcribed_text)
            print(f"💾 转写文本已保存到: {transcript_file}")
            
            return transcribed_text
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else ''
            if 'timed out' in error_msg or 'timeout' in error_msg.lower():
                print("⏰ Whisper 转写超时（15分钟）")
                print("💡 建议：此音频可能较长，可以稍后手动重试")
            else:
                print("❌ Whisper HTTP 转写失败")
                print(f"错误输出: {error_msg}")
                print(f"标准输出: {e.stdout.decode('utf-8') if e.stdout else '无'}")
            return ""
        except Exception as e:
            print(f"音频转录失败: {e}")
            return ""

    def summarize_text(self, text: str, video_title: str) -> str:
        """使用 ollama qwen3:32b 总结文本"""
        prompt = f"""
你是一名专业的视频内容分析师，请对下列视频内容进行详细总结。

视频标题：{video_title}

总结要求：
1. 提取主要观点和论述
2. 记录重要的细节和数据  
3. 识别独特的见解和分析角度
4. 保留有价值的争议点和讨论点
5. 整理可用于内容创作的素材

请按以下格式输出：

## 🎯 主要观点
- [核心观点1]
- [核心观点2]
...

## � 重要细节
- [细节1]
- [细节2]
...

## 💡 独特见解
- [见解1]
- [见解2]
...

## ⚡ 争议讨论点
- [争议点1]
- [争议点2]
...

## � 其他有价值信息
- [信息1]
- [信息2]
...

⚠️ 不要加入你的思考过程，不要说"我认为"或"可能"，只根据内容原文总结。

视频内容如下：
———
{text}
———
/no_think
        """
        
        try:
            response = self.ollama.chat(
                model='qwen3:32b',
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            print(f"ollama 总结失败: {e}")
            return f"总结失败: {e}"

    def save_summary(self, video_id: str, video_title: str, summary: str, output_path: Path = None):
        """保存总结到文件"""
        if output_path:
            # 保存到视频专属目录
            summary_file = output_path / "summary.txt"
        else:
            # 兼容旧的保存方式
            safe_title = self._clean_filename(video_title)
            summary_file = self.summaries_dir / f"{video_id}_{safe_title[:50]}.txt"
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"视频标题: {video_title}\n")
                f.write(f"视频ID: {video_id}\n")
                f.write("=" * 50 + "\n")
                f.write(summary)
            print(f"💾 总结已保存到: {summary_file}")
        except Exception as e:
            print(f"保存总结失败: {e}")

    def create_final_summary(self, keyword: str) -> str:
        """对所有总结进行最终汇总"""
        # 读取所有总结文件（包括新的目录结构和旧的文件结构）
        all_summaries = []
        video_sources = []  # 记录视频来源信息
        
        # 读取新结构：从各个视频目录中的 summary.txt
        for video_dir in self.download_dir.glob("*"):
            if video_dir.is_dir():
                summary_file = video_dir / "summary.txt"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            all_summaries.append(content)
                            
                            # 从文件内容中提取视频信息
                            lines = content.split('\n')
                            video_title = "未知标题"
                            video_id = "未知ID"
                            
                            for line in lines:
                                if line.startswith("视频标题:"):
                                    video_title = line.replace("视频标题:", "").strip()
                                elif line.startswith("视频ID:"):
                                    video_id = line.replace("视频ID:", "").strip()
                            
                            video_sources.append({
                                'title': video_title,
                                'id': video_id,
                                'url': f'https://www.youtube.com/watch?v={video_id}'
                            })
                            
                    except Exception as e:
                        print(f"读取总结文件失败: {e}")
        
        # 读取旧结构：从 summaries 目录中的文件（向后兼容）
        for summary_file in self.summaries_dir.glob("*.txt"):
            if not summary_file.name.startswith("final_summary_"):
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        all_summaries.append(content)
                        
                        # 从文件内容中提取视频信息
                        lines = content.split('\n')
                        video_title = "未知标题"
                        video_id = "未知ID"
                        
                        for line in lines:
                            if line.startswith("视频标题:"):
                                video_title = line.replace("视频标题:", "").strip()
                            elif line.startswith("视频ID:"):
                                video_id = line.replace("视频ID:", "").strip()
                        
                        video_sources.append({
                            'title': video_title,
                            'id': video_id,
                            'url': f'https://www.youtube.com/watch?v={video_id}'
                        })
                        
                except Exception as e:
                    print(f"读取总结文件失败: {e}")
        
        if not all_summaries:
            return "没有找到任何总结文件"
        
        # 生成来源列表
        sources_text = "\n📋 分析来源视频：\n"
        for i, source in enumerate(video_sources, 1):
            sources_text += f"{i}. 《{source['title']}》\n"
            sources_text += f"   视频ID: {source['id']} | URL: {source['url']}\n\n"
        
        combined_text = sources_text + "\n" + "="*50 + "\n\n".join(all_summaries)
        
        final_prompt = f"""
你是一名专业的观点提取专家，请从以下关于"{keyword}"的多个视频总结中，提取所有有价值的观点、见解和分析角度。

目标：为YouTube内容创作者提供丰富的素材库，包含各种不同的观点、分析角度、有趣细节和创作灵感。

重要要求：
- 逐一提取每个视频中的独特观点和见解，不要合并或概括
- 保持原始表达的新颖性和独特性
- 当提到观点时，必须标注来源视频序号，如 (来源：视频1)
- 重点关注可用于内容创作的有趣角度、争议点、细节发现

请按以下格式输出：

## � 角色分析观点
（从各视频中提取的关于角色的不同解读和分析）

## �🎯 剧情解读角度  
（各视频对剧情的不同理解、细节发现、逻辑分析）

## 🔥 争议话题收集
（观众争议的焦点、不同立场、批评和赞美的具体点）

## 💡 创作灵感素材
（有趣的细节、彩蛋发现、幕后信息、制作分析）

## 🎭 文化解读视角
（社会意义、隐喻分析、文化背景解读）

## 📊 观众反响整理
（收视数据、观众评价、网络讨论热点）

## � 未来展望观点
（对续集、衍生、影响的各种预测和期待）

注意：每条观点都要标注来源，保持观点的原始性和多样性，避免同质化表达。

视频总结内容：
———
{combined_text}
———
/no_think
        """
        
        try:
            response = self.ollama.chat(
                model='qwen3:32b',
                messages=[{"role": "user", "content": final_prompt}]
            )
            
            final_summary = response['message']['content']
            
            # 保存最终总结（包含来源信息）
            final_file = self.summaries_dir / f"final_summary_{keyword}.txt"
            with open(final_file, 'w', encoding='utf-8') as f:
                f.write(f"关键词: {keyword}\n")
                f.write("=" * 50 + "\n")
                
                # 先写入来源视频列表
                f.write(sources_text)
                f.write("\n" + "=" * 50 + "\n")
                f.write("🔍 综合分析报告\n")
                f.write("=" * 50 + "\n")
                f.write(final_summary)
            
            print(f"最终总结已保存到: {final_file}")
            return final_summary
            
        except Exception as e:
            print(f"创建最终总结失败: {e}")
            return f"最终总结失败: {e}"

    def filter_videos_with_llm(self, videos: List[Dict], keyword: str, top_n: int = 5) -> List[Dict]:
        """使用 LLM 智能筛选最相关的视频"""
        if not videos:
            return []
        
        print(f"🤖 使用 LLM 筛选最相关的 {top_n} 个视频...")
        
        # 准备视频信息给 LLM 分析
        video_list_text = ""
        for i, video in enumerate(videos, 1):
            video_list_text += f"{i}. 标题: {video['title']}\n"
            video_list_text += f"   观看量: {video['view_text']} ({video['view_count']}次)\n"
            video_list_text += f"   视频ID: {video['video_id']}\n\n"
        
        prompt = f"""
你是一名专业的内容分析师，请仔细分析每个视频标题，从以下视频列表中筛选出与关键词"{keyword}"最相关且最有价值的前{top_n}个视频。

## 筛选标准（按重要性排序）：

### 1. � 内容相关性（最重要）
- 标题直接提到"{keyword}"或相关词汇
- 涉及该主题的深度分析、解读、评论
- 包含剧情分析、角色解读、细节解析等内容

### 2. 📚 内容质量判断
- **优先选择**：
  - 深度解析类："解析"、"分析"、"解读"、"意义"、"隐藏"
  - 完整评论类："一次看懂"、"全解析"、"完整版"、"结局"
  - 专业评论类：避免过度夸张的标题

- **避免选择**：
  - 纯片段剪辑："片段"、"cut"、"剪辑"
  - 标题党内容：过多感叹号、夸张词汇
  - 无关内容：与"{keyword}"主题不符的视频

### 3. 🔍 题材分类
- 剧情分析：角色心理、情节逻辑、隐喻含义
- 细节发现：彩蛋、线索、制作细节
- 文化解读：社会意义、背景分析
- 观众反响：评论、讨论、争议点

## 分析要求：
1. **逐个分析每个标题的相关性**
2. **优先选择内容质量高、与主题最相关的视频**
3. **播放量仅作为次要参考，不是主要标准**
4. **确保选择的视频能提供不同角度的分析观点**

请仔细阅读每个标题，分析其与"{keyword}"的相关性，然后返回最相关的{top_n}个视频序号。

只返回序号，用逗号分隔，例如：1,3,5,7,9

视频列表：
{video_list_text}

分析后选择的序号："""
        
        try:
            print("🔄 正在调用 LLM 进行视频筛选...")
            response = self.ollama.chat(
                model='qwen3:32b',
                messages=[{"role": "user", "content": prompt}]
            )
            
            # 解析 LLM 的回答，提取视频序号
            response_text = response['message']['content'].strip()
            
            # 尝试多种方式解析视频序号
            video_ids = []
            
            # 方法1: 查找最后一行包含数字和逗号的内容（LLM 的最终答案）
            lines = response_text.split('\n')
            for line in reversed(lines):  # 从最后一行开始查找
                line = line.strip()
                # 寻找包含数字和逗号的行，这通常是最终答案
                if ',' in line and re.search(r'\d+', line):
                    # 提取所有数字
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 3:  # 至少要有3个数字才认为是有效答案
                        video_ids = numbers
                        break
            
            # 方法2: 查找包含关键词的行
            if not video_ids:
                for line in response_text.split('\n'):
                    if any(keyword in line for keyword in ['序号', '选择', '答案', '结果']):
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            video_ids = numbers
                            break
            
            # 方法3: 如果前面方法都失败，查找所有数字（但限制数量）
            if not video_ids:
                all_numbers = re.findall(r'\d+', response_text)
                # 过滤掉明显不是序号的数字（如年份、播放量等）
                valid_numbers = []
                for num in all_numbers:
                    num_int = int(num)
                    if 1 <= num_int <= len(videos):  # 只保留有效的视频序号范围
                        valid_numbers.append(num)
                video_ids = valid_numbers[:top_n * 2]  # 最多取top_n的2倍，避免过多
            
            print(f"🔍 LLM 回答: {response_text[:200]}...")
            print(f"📝 解析到的视频序号: {video_ids[:10]}...")  # 只显示前10个
            
            if not video_ids:
                # 如果解析失败，fallback到按播放量排序（去重）
                print("⚠️ LLM 筛选失败，使用播放量排序")
                # 去重处理
                unique_videos = {}
                for video in videos:
                    if video['video_id'] not in unique_videos:
                        unique_videos[video['video_id']] = video
                
                sorted_videos = sorted(unique_videos.values(), key=lambda x: x['view_count'], reverse=True)
                return sorted_videos[:top_n]
            
            # 根据LLM选择的序号筛选视频，去重处理
            selected_videos = []
            seen_video_ids = set()  # 用于去重
            
            for num_str in video_ids:
                try:
                    index = int(num_str) - 1  # 转换为0索引
                    if 0 <= index < len(videos):
                        video = videos[index]
                        # 检查是否已经选择过这个视频
                        if video['video_id'] not in seen_video_ids:
                            selected_videos.append(video)
                            seen_video_ids.add(video['video_id'])
                            
                            if len(selected_videos) >= top_n:
                                break
                except ValueError:
                    continue
            
            # 如果选择的视频不够，用播放量高的视频补充
            if len(selected_videos) < top_n:
                print(f"⚠️ LLM 选择的视频不够({len(selected_videos)})，用高播放量视频补充...")
                remaining_videos = [v for v in videos if v['video_id'] not in seen_video_ids]
                sorted_remaining = sorted(remaining_videos, key=lambda x: x['view_count'], reverse=True)
                
                for video in sorted_remaining:
                    if len(selected_videos) >= top_n:
                        break
                    selected_videos.append(video)
                    seen_video_ids.add(video['video_id'])
            
            print(f"🤖 LLM 筛选完成，选择了 {len(selected_videos)} 个视频")
            for i, video in enumerate(selected_videos, 1):
                print(f"  {i}. {video['title']} ({video['view_text']})")
            
            return selected_videos
            
        except Exception as e:
            print(f"❌ LLM 筛选失败: {e}")
            # fallback到按播放量排序（去重）
            unique_videos = {}
            for video in videos:
                if video['video_id'] not in unique_videos:
                    unique_videos[video['video_id']] = video
            
            sorted_videos = sorted(unique_videos.values(), key=lambda x: x['view_count'], reverse=True)
            return sorted_videos[:top_n]

    async def process_videos(self, keyword: str, max_videos: int = 5):
        """处理整个流程 - 优化为先批量下载，再批量处理"""
        print(f"🔍 开始搜索关键词: {keyword}")
        
        # 1. 搜索视频，获取100个候选视频
        print("📊 第一步：采集候选视频...")
        all_videos = await self.search_youtube_videos(keyword, max_videos=100)
        
        if not all_videos:
            print("❌ 没有找到任何视频")
            return
        
        # 2. 使用 LLM 智能筛选最相关的前5个视频
        print("🤖 第二步：LLM 智能筛选最相关视频...")
        selected_videos = self.filter_videos_with_llm(all_videos, keyword, top_n=max_videos)
        
        if not selected_videos:
            print("❌ 筛选后没有找到合适的视频")
            return
        
        print(f"✅ 最终选择处理 {len(selected_videos)} 个视频")
        
        # 显示选中的视频列表
        print("\n📋 选中的视频列表：")
        for i, video in enumerate(selected_videos, 1):
            print(f"  {i}. {video['title']}")
            print(f"     观看量: {video['view_text']} | ID: {video['video_id']}")
        
        # 3. 阶段一：批量下载所有视频的字幕/音频
        print(f"\n{'='*60}")
        print("🎯 阶段一：批量下载字幕和音频")
        print(f"{'='*60}")
        
        download_results = []
        for i, video in enumerate(selected_videos, 1):
            print(f"\n� 下载第 {i}/{len(selected_videos)} 个视频")
            print(f"标题: {video['title']}")
            
            # 下载视频信息
            video_info = self.download_video_info(video['url'])
            if not video_info:
                print("❌ 无法获取视频详细信息，跳过")
                download_results.append({
                    'video': video,
                    'success': False,
                    'reason': '无法获取视频信息'
                })
                continue
            
            # 下载字幕或音频
            download_result = self.download_video_with_subtitles(video['url'], video['video_id'], video['title'])
            
            if download_result['subtitle_downloaded'] or download_result['audio_file']:
                print("✅ 下载成功")
                download_results.append({
                    'video': video,
                    'success': True,
                    'download_result': download_result
                })
            else:
                print("❌ 下载失败")
                download_results.append({
                    'video': video,
                    'success': False,
                    'reason': '字幕和音频都下载失败'
                })
        
        # 统计下载结果
        successful_downloads = [r for r in download_results if r['success']]
        print(f"\n📊 下载阶段完成: {len(successful_downloads)}/{len(selected_videos)} 个视频下载成功")
        
        if not successful_downloads:
            print("❌ 没有成功下载任何视频内容，无法继续处理")
            return
        
        # 4. 阶段二：批量转写和摘要
        print(f"\n{'='*60}")
        print("🎯 阶段二：批量转写和摘要")
        print(f"{'='*60}")
        
        successful_count = 0
        for i, result in enumerate(successful_downloads, 1):
            video = result['video']
            download_result = result['download_result']
            
            print(f"\n📝 处理第 {i}/{len(successful_downloads)} 个视频")
            print(f"标题: {video['title']}")
            
            # 提取文本
            text_content = ""
            if download_result['subtitle_downloaded'] and download_result['subtitle_file']:
                print("📝 使用字幕进行总结")
                text_content = self.extract_text_from_subtitle(download_result['subtitle_file'])
            elif download_result['audio_file']:
                print("🔊 使用音频转录进行总结")
                text_content = self.transcribe_audio(download_result['audio_file'])
            
            if not text_content:
                print("❌ 无法提取文本内容，跳过")
                continue
            
            print(f"📊 提取的文本长度: {len(text_content)} 字符")
            
            # 生成总结
            print("🤖 正在生成总结...")
            summary = self.summarize_text(text_content, video['title'])
            
            # 保存总结到视频目录
            self.save_summary(video['video_id'], video['title'], summary, download_result['output_path'])
            successful_count += 1
            
            print(f"✅ 视频 {i} 处理完成")
        
        # 5. 生成最终总结
        if successful_count > 0:
            print(f"\n🎯 处理完成统计:")
            print(f"   📥 下载成功: {len(successful_downloads)}/{len(selected_videos)} 个视频")
            print(f"   📝 摘要成功: {successful_count}/{len(successful_downloads)} 个视频")
            print("📋 生成最终综合总结...")
            final_summary = self.create_final_summary(keyword)
            print("\n" + "="*60)
            print("=== 最终综合总结 ===")
            print("="*60)
            print(final_summary)
        else:
            print("❌ 没有成功处理任何视频，无法生成最终总结")

    def _clean_filename(self, filename: str) -> str:
        """清理文件名，移除非法字符"""
        # 移除或替换文件系统不允许的字符
        import re
        # 替换非法字符为下划线
        cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # 移除多余的空格和点
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # 限制长度，避免路径过长
        if len(cleaned) > 50:
            cleaned = cleaned[:50].rstrip()
        return cleaned

async def main():
    keyword = "鱿鱼游戏3"
    analyzer = YouTubeVideoAnalyzer()
    await analyzer.process_videos(keyword, max_videos=5)

if __name__ == "__main__":
    asyncio.run(main())
