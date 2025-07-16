#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube视频批量下载和分析 - 优化观点提取版本
专门为YouTube内容创作者设计，提取可用于内容创作的观点库

主要改进：
1. 优化最终综合总结的提示词，更适合内容创作
2. 按照观点价值和创作用途分类整理
3. 突出争议点和独特见解
4. 提供具体的创作建议
"""

import asyncio
import json
import os
import re
import requests
import time
from pathlib import Path
from urllib.parse import quote_plus, urljoin
import yt_dlp
from crawl4ai import AsyncWebCrawler


class YouTubeAnalyzer:
    def __init__(self, downloads_dir="downloads", summaries_dir="summaries"):
        self.downloads_dir = Path(downloads_dir)
        self.summaries_dir = Path(summaries_dir)
        self.downloads_dir.mkdir(exist_ok=True)
        self.summaries_dir.mkdir(exist_ok=True)
        
        # 初始化 ollama（与 M4-3 保持一致）
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            import sys, subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ollama'])
            import ollama
            self.ollama = ollama
        
        # Whisper 服务配置
        self.whisper_url = "http://127.0.0.1:8080/transcribe/"
        
        # cookies 文件路径
        self.cookies_file = "/workspace/ytdlp/cookies.txt"
        
    def clean_filename(self, filename):
        """清理文件名，移除特殊字符"""
        # 移除或替换不能用于文件名的字符
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'[^\w\s-]', '', filename)
        filename = re.sub(r'[-\s]+', '-', filename)
        return filename.strip('-')
    
    async def search_youtube_videos(self, keyword, max_videos=20):
        """使用 crawl4ai 搜索 YouTube 视频"""
        print(f"🔍 搜索关键词: {keyword}")
        
        search_url = f"https://www.youtube.com/results?search_query={quote_plus(keyword)}"
        
        async with AsyncWebCrawler(verbose=True) as crawler:
            # 增加滚动次数和等待时间，获取更多视频
            js_code = """
            for(let i = 0; i < 8; i++) {
                window.scrollTo(0, document.body.scrollHeight);
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
            """
            
            result = await crawler.arun(
                url=search_url,
                js_code=js_code,
                wait_for="css:ytd-video-renderer",
                delay_before_return_html=3.0
            )
            
            if result.success:
                videos = self._parse_search_results(result.html)
                print(f"✅ 找到 {len(videos)} 个视频")
                return videos[:max_videos]
            else:
                print(f"❌ 搜索失败: {result.error_message}")
                return []
    
    def _parse_search_results(self, html):
        """解析搜索结果页面"""
        videos = []
        
        # 多种解析方法
        patterns = [
            # 方法1: JSON 数据解析
            r'"videoId":"([^"]+)".*?"title":{"runs":\[{"text":"([^"]+)"}.*?"viewCountText":{"simpleText":"([^"]+)"}',
            # 方法2: 直接从HTML解析
            r'watch\?v=([a-zA-Z0-9_-]{11})',
        ]
        
        video_data = []
        
        # 首先尝试提取视频ID
        video_ids = set()
        for pattern in [r'watch\?v=([a-zA-Z0-9_-]{11})', r'"videoId":"([a-zA-Z0-9_-]{11})"']:
            matches = re.findall(pattern, html)
            video_ids.update(matches)
        
        print(f"发现 {len(video_ids)} 个唯一视频ID")
        
        # 尝试通过多种方式获取视频信息
        for video_id in list(video_ids)[:30]:  # 限制数量避免过多
            video_info = self._extract_video_info(html, video_id)
            if video_info:
                video_data.append(video_info)
        
        # 如果通过HTML解析失败，使用基础信息
        if not video_data:
            for video_id in list(video_ids)[:20]:
                video_data.append({
                    'video_id': video_id,
                    'title': f'视频_{video_id}',  # 占位标题
                    'view_count': '未知',
                    'url': f'https://www.youtube.com/watch?v={video_id}'
                })
        
        return video_data
    
    def _extract_video_info(self, html, video_id):
        """从HTML中提取特定视频的信息"""
        try:
            # 多种title提取方法
            title_patterns = [
                rf'"videoId":"{re.escape(video_id)}".*?"title":{{"runs":\[{{"text":"([^"]+)"}}',
                rf'"videoId":"{re.escape(video_id)}".*?"title":{{"simpleText":"([^"]+)"}}',
                rf'watch\?v={re.escape(video_id)}[^>]*aria-label="([^"]*)"',
            ]
            
            title = None
            for pattern in title_patterns:
                match = re.search(pattern, html, re.DOTALL)
                if match:
                    title = match.group(1)
                    break
            
            if not title:
                title = f"视频_{video_id}"
            
            # 播放量提取
            view_patterns = [
                rf'"videoId":"{re.escape(video_id)}".*?"viewCountText":{{"simpleText":"([^"]+)"}}',
                rf'"videoId":"{re.escape(video_id)}".*?"shortViewCountText":{{"simpleText":"([^"]+)"}}',
            ]
            
            view_count = "未知"
            for pattern in view_patterns:
                match = re.search(pattern, html, re.DOTALL)
                if match:
                    view_count = match.group(1)
                    break
            
            return {
                'video_id': video_id,
                'title': title,
                'view_count': view_count,
                'url': f'https://www.youtube.com/watch?v={video_id}'
            }
            
        except Exception as e:
            print(f"解析视频 {video_id} 信息时出错: {e}")
            return None

    def filter_videos_with_llm(self, videos, keyword, target_count=8):
        """使用 LLM 筛选最相关的视频"""
        print(f"🤖 使用 LLM 筛选最相关的 {target_count} 个视频...")
        
        # 测试 ollama 连接
        try:
            print("🔍 测试 ollama 连接...")
            test_response = self.ollama.chat(
                model='qwen3:32b',
                messages=[{"role": "user", "content": "测试连接，请回复'连接正常'"}]
            )
            print(f"✅ ollama 连接正常")
        except Exception as e:
            print(f"❌ ollama 连接失败: {e}")
            print("使用备用筛选方法...")
            return videos[:target_count]
        
        # 构建视频列表文本
        videos_text = ""
        for i, video in enumerate(videos, 1):
            videos_text += f"{i}. {video['title']} (播放量: {video['view_count']})\n"
        
        prompt = f"""
请从以下关于"{keyword}"的YouTube视频中，选出最适合做深度分析的 {target_count} 个视频。

选择标准（按重要性排序）：
1. 影评、解析、深度分析类内容（最重要）
2. 剧情讨论、角色分析、细节解读
3. 幕后花絮、制作分析、彩蛋发现
4. 观众反应、评论分析
5. 播放量相对较高的视频

避免选择：
- 纯搬运、无解说的视频
- 标题党、蹭热度的低质量内容
- 过于简短的快速评论

视频列表：
{videos_text}

请只返回选中的视频序号，用逗号分隔，例如：1,3,5,7,9,12,15,18

选择的序号："""

        try:
            print("🔄 正在调用 LLM 进行视频筛选...")
            response = self.ollama.chat(
                model='qwen3:32b',
                messages=[{"role": "user", "content": prompt}]
            )
            
            # 解析回复，提取序号
            response_text = response['message']['content'].strip()
            print(f"LLM 回复: {response_text}")
            
            # 使用正则表达式提取数字
            numbers = re.findall(r'\d+', response_text)
            selected_indices = [int(num) - 1 for num in numbers if int(num) <= len(videos)]  # 转换为0索引
            
            # 确保不超过目标数量
            selected_indices = selected_indices[:target_count]
            
            selected_videos = [videos[i] for i in selected_indices if i < len(videos)]
            
            print(f"✅ 筛选出 {len(selected_videos)} 个相关视频")
            for video in selected_videos:
                print(f"  - {video['title']}")
            
            return selected_videos
            
        except Exception as e:
            print(f"❌ LLM 筛选失败: {e}")
            print("使用备用筛选方法...")
            # 备用方法：选择前几个视频
            return videos[:target_count]

    def download_video_with_subtitles(self, video_info):
        """下载视频的字幕和音频"""
        video_id = video_info['video_id']
        title = self.clean_filename(video_info['title'])
        
        # 为每个视频创建单独的目录
        video_dir = self.downloads_dir / f"{video_id}_{title}"
        video_dir.mkdir(exist_ok=True)
        
        print(f"📥 下载视频: {title}")
        
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['zh-Hans', 'zh', 'en'],
            'format': 'bestaudio/best',
            'outtmpl': str(video_dir / f'{video_id}_{title}.%(ext)s'),
            'extract_flat': False,
        }
        
        # 如果 cookies 文件存在，使用它
        if os.path.exists(self.cookies_file):
            ydl_opts['cookiefile'] = self.cookies_file
            print(f"使用 cookies 文件: {self.cookies_file}")
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_info['url']])
            
            print(f"✅ 下载完成: {title}")
            return video_dir
            
        except Exception as e:
            print(f"❌ 下载失败 {title}: {e}")
            return None

    def transcribe_audio(self, video_dir, video_id, title):
        """使用 Whisper HTTP 服务转写音频"""
        print(f"🎙️ 转写音频: {title}")
        
        # 查找音频文件
        audio_files = list(video_dir.glob(f"{video_id}_*.mp3")) + \
                     list(video_dir.glob(f"{video_id}_*.m4a")) + \
                     list(video_dir.glob(f"{video_id}_*.webm"))
        
        if not audio_files:
            print(f"❌ 未找到音频文件: {title}")
            return None
        
        audio_file = audio_files[0]
        
        try:
            import subprocess
            
            # 使用 curl 调用 whisper HTTP 服务（与 M4-3 保持一致）
            print(f"🎵 使用音频文件: {audio_file}")
            print("⏳ 正在调用 whisper HTTP 服务 ...")
            
            curl_cmd = [
                "curl", "-X", "POST", self.whisper_url,
                "-F", f"audio_file=@{str(audio_file)};type=audio/mpeg",
                "-F", "model_name=large-v3",
                "-F", "language=zh",
                "--connect-timeout", "30",    # 连接超时30秒
                "--max-time", "900"           # 总超时15分钟（900秒）
            ]
            
            result = subprocess.run(curl_cmd, capture_output=True, check=True)
            response_text = result.stdout.decode("utf-8")
            
            try:
                # 解析 JSON 响应
                response_data = json.loads(response_text)
                
                if response_data.get("success"):
                    # 优先使用 srt_subtitles，如果没有则使用 full_text
                    if "srt_subtitles" in response_data:
                        srt_content = response_data["srt_subtitles"]
                        # 简单处理 SRT 格式，提取文本
                        import re
                        lines = srt_content.splitlines()
                        text_lines = []
                        for line in lines:
                            # 跳过序号和时间轴
                            if re.match(r"^\d+$", line) or re.match(r"^\d{2}:\d{2}:\d{2},\d{3} --> ", line) or line.strip() == '':
                                continue
                            text_lines.append(line.strip())
                        transcription = ' '.join(text_lines)
                    else:
                        transcription = response_data.get("full_text", "")
                else:
                    print("❌ Whisper 服务返回错误")
                    return None
                    
            except json.JSONDecodeError:
                # 如果不是 JSON，尝试直接作为 SRT 处理（向后兼容）
                print("⚠️ 响应不是 JSON 格式，尝试直接解析为 SRT")
                import re
                lines = response_text.splitlines()
                text_lines = []
                for line in lines:
                    if re.match(r"^\d+$", line) or re.match(r"^\d{2}:\d{2}:\d{2},\d{3} --> ", line) or line.strip() == '':
                        continue
                    text_lines.append(line.strip())
                transcription = ' '.join(text_lines)
            
            if transcription:
                # 保存转写结果
                transcription_file = video_dir / f"{video_id}_transcription.txt"
                with open(transcription_file, 'w', encoding='utf-8') as f:
                    f.write("=== 音频转写文本 ===\n")
                    f.write(f"音频文件: {audio_file.name}\n")
                    f.write("=" * 50 + "\n")
                    f.write(transcription)
                
                print(f"✅ 转写完成: {title}")
                print(f"💾 转写文本已保存到: {transcription_file}")
                return transcription
            else:
                print(f"❌ 转写结果为空: {title}")
                return None
                
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else ''
            if 'timed out' in error_msg or 'timeout' in error_msg.lower():
                print("⏰ Whisper 转写超时（15分钟）")
                print("💡 建议：此音频可能较长，可以稍后手动重试")
            else:
                print("❌ Whisper HTTP 转写失败")
                print(f"错误输出: {error_msg}")
                print(f"标准输出: {e.stdout.decode('utf-8') if e.stdout else '无'}")
            return None
        except Exception as e:
            print(f"❌ 转写失败 {title}: {e}")
            return None
            return None

    def extract_text_from_subtitle(self, video_dir, video_id):
        """从字幕文件提取文本"""
        # 查找字幕文件
        subtitle_files = list(video_dir.glob(f"{video_id}_*.vtt")) + \
                        list(video_dir.glob(f"{video_id}_*.srt"))
        
        if not subtitle_files:
            return None
        
        subtitle_file = subtitle_files[0]
        
        try:
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取字幕文本（移除时间戳等）
            lines = content.split('\n')
            text_lines = []
            
            for line in lines:
                line = line.strip()
                # 跳过时间戳行和空行
                if line and not re.match(r'^\d+$', line) and not re.match(r'^\d{2}:\d{2}:\d{2}', line) and line != 'WEBVTT':
                    text_lines.append(line)
            
            subtitle_text = '\n'.join(text_lines)
            
            # 保存提取的字幕文本
            if subtitle_text:
                subtitle_text_file = video_dir / f"{video_id}_subtitle_text.txt"
                with open(subtitle_text_file, 'w', encoding='utf-8') as f:
                    f.write(subtitle_text)
                return subtitle_text
            
        except Exception as e:
            print(f"字幕文本提取失败: {e}")
            
        return None

    def summarize_content(self, content, title):
        """使用 LLM 对内容进行总结"""
        print(f"📝 生成摘要: {title}")
        
        prompt = f"""
请对以下视频内容进行详细总结。视频标题：{title}

总结要求：
1. 提取主要观点和论述
2. 记录重要的细节和数据
3. 识别独特的见解和分析角度
4. 保留有价值的争议点和讨论点
5. 整理可用于内容创作的素材

请按以下格式输出：

## 主要观点
- [核心观点1]
- [核心观点2]
...

## 重要细节
- [细节1]
- [细节2]
...

## 独特见解
- [见解1]
- [见解2]
...

## 争议讨论点
- [争议点1]
- [争议点2]
...

## 其他有价值信息
- [信息1]
- [信息2]
...

视频内容：
{content}
"""

        try:
            response = self.ollama.chat(
                model='qwen3:32b',
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = response['message']['content']
            print(f"✅ 摘要生成完成: {title}")
            return summary
            
        except Exception as e:
            print(f"❌ 摘要生成失败 {title}: {e}")
            return None

    def save_summary(self, summary, video_dir, video_id, title):
        """保存摘要到文件"""
        if summary:
            summary_file = video_dir / f"{video_id}_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"视频标题: {title}\n")
                f.write("=" * 50 + "\n")
                f.write(summary)
            print(f"💾 摘要已保存: {summary_file}")

    def create_final_summary(self, keyword, processed_videos):
        """创建最终的综合总结 - 专为YouTube内容创作优化"""
        print(f"📚 生成最终综合总结...")
        
        # 收集所有摘要
        all_summaries = []
        sources_info = []
        
        for i, (video_info, video_dir) in enumerate(processed_videos, 1):
            if video_dir:
                summary_file = video_dir / f"{video_info['video_id']}_summary.txt"
                if summary_file.exists():
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary_content = f.read()
                        all_summaries.append(f"=== 视频{i} ===\n{summary_content}")
                        
                        # 收集来源信息
                        sources_info.append({
                            'index': i,
                            'title': video_info['title'],
                            'video_id': video_info['video_id'],
                            'url': video_info['url'],
                            'view_count': video_info.get('view_count', '未知')
                        })
        
        if not all_summaries:
            print("❌ 没有找到可用的摘要文件")
            return
        
        # 生成来源视频列表
        sources_text = f"## 📹 来源视频列表\n\n"
        for source in sources_info:
            sources_text += f"**视频{source['index']}**: {source['title']}\n"
            sources_text += f"- 视频ID: {source['video_id']}\n"
            sources_text += f"- 播放量: {source['view_count']}\n"
            sources_text += f"- 链接: {source['url']}\n\n"
        
        sources_text += "\n" + "="*50 + "\n\n"
        
        # 合并所有摘要
        combined_text = sources_text + "\n" + "="*50 + "\n\n".join(all_summaries)
        
        # 针对内容创作优化的提示词
        final_prompt = f"""
你是一名专业的YouTube内容策划师，请从以下关于"{keyword}"的多个视频总结中，提取所有可用于内容创作的观点、见解和素材。

核心目标：为YouTube创作者提供完整的观点库和创作素材，帮助制作有深度的原创内容。

任务要求：
- 提取所有独特观点，不要遗漏任何有价值的见解
- 每个观点必须标注来源视频序号，如 (来源：视频1)、(来源：视频3,5)
- 识别争议性观点，这些是引发讨论的绝佳素材
- 提取具体的细节、数据、案例，增强内容说服力
- 挖掘可以延伸讨论的话题点

输出格式：

## 🎬 **主流观点与共识**
（大部分视频认同的观点，适合作为视频基调）
- [具体观点描述] (来源：视频X)
- [具体观点描述] (来源：视频X,Y)

## ⚡ **争议观点与分歧**
（有争议的话题，适合制作讨论类内容）
- [争议点]：正方观点 vs 反方观点 (来源：视频X vs 视频Y)
- [争议点]：[具体争议内容] (来源：视频X)

## 💎 **独特见解与深度分析**
（别人没说过的观点，适合作为视频亮点）
- [独特观点或深度分析] (来源：视频X)

## 🔍 **关键细节与数据**
（具体的事实、数据、案例，增强内容可信度）
- [具体细节/数据/案例] (来源：视频X)

## 🎯 **延伸话题机会**
（基于收集的观点，可以进一步探讨的方向）
- 话题：[基于XX观点的延伸讨论方向] (素材来源：视频X)

## 📺 **创作建议**
1. **推荐视频角度**：[基于观点分析的创作建议]
2. **避免雷区**：[需要谨慎处理的敏感话题]
3. **亮点素材**：[最有吸引力的讨论点]

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
                f.write("\n")
                
                # 写入最终总结
                f.write("## 🎯 **YouTube内容创作素材库**\n\n")
                f.write(final_summary)
            
            print(f"✅ 最终总结已保存: {final_file}")
            print(f"📋 包含 {len(sources_info)} 个来源视频的观点分析")
            
        except Exception as e:
            print(f"❌ 生成最终总结失败: {e}")

    async def process_videos(self, keyword, max_videos=20, target_videos=8):
        """处理视频的完整流程"""
        print(f"🚀 开始处理关键词: {keyword}")
        
        # 1. 搜索视频
        videos = await self.search_youtube_videos(keyword, max_videos)
        if not videos:
            print("❌ 没有找到视频")
            return
        
        # 2. LLM 筛选
        selected_videos = self.filter_videos_with_llm(videos, keyword, target_videos)
        if not selected_videos:
            print("❌ 没有筛选出合适的视频")
            return
        
        print(f"\n📥 开始批量下载 {len(selected_videos)} 个视频...")
        
        # 3. 批量下载
        downloaded_videos = []
        for i, video_info in enumerate(selected_videos, 1):
            print(f"\n--- 下载进度: {i}/{len(selected_videos)} ---")
            try:
                video_dir = self.download_video_with_subtitles(video_info)
                downloaded_videos.append((video_info, video_dir))
            except Exception as e:
                print(f"❌ 下载视频失败: {video_info['title']}: {e}")
                downloaded_videos.append((video_info, None))
        
        print(f"\n🎙️ 开始批量转写和摘要...")
        
        # 4. 批量转写和摘要
        processed_videos = []
        for i, (video_info, video_dir) in enumerate(downloaded_videos, 1):
            if video_dir is None:
                print(f"跳过失败的视频: {video_info['title']}")
                processed_videos.append((video_info, None))
                continue
                
            print(f"\n--- 处理进度: {i}/{len(downloaded_videos)} ---")
            
            try:
                video_id = video_info['video_id']
                title = video_info['title']
                
                # 获取文本内容（字幕优先，然后转写）
                content = self.extract_text_from_subtitle(video_dir, video_id)
                if not content:
                    content = self.transcribe_audio(video_dir, video_id, title)
                
                if content:
                    # 生成摘要
                    summary = self.summarize_content(content, title)
                    self.save_summary(summary, video_dir, video_id, title)
                    processed_videos.append((video_info, video_dir))
                else:
                    print(f"❌ 无法获取视频内容: {title}")
                    processed_videos.append((video_info, None))
                    
            except Exception as e:
                print(f"❌ 处理视频失败: {video_info['title']}: {e}")
                processed_videos.append((video_info, None))
        
        # 5. 生成最终综合总结
        self.create_final_summary(keyword, processed_videos)
        
        print(f"\n🎉 处理完成！关键词: {keyword}")
        print(f"📊 总视频数: {len(selected_videos)}")
        print(f"📥 成功下载: {len([v for v in processed_videos if v[1] is not None])}")
        print(f"📝 已生成摘要和最终总结")


async def main():
    """主函数"""
    analyzer = YouTubeAnalyzer()
    
    # 可以批量处理多个关键词
    keywords = ["鱿鱼游戏3"]  # 可以添加更多关键词
    
    for keyword in keywords:
        try:
            await analyzer.process_videos(
                keyword=keyword,
                max_videos=25,   # 搜索更多视频
                target_videos=8  # 筛选出8个最相关的
            )
            print(f"✅ 关键词 '{keyword}' 处理完成\n")
        except Exception as e:
            print(f"❌ 处理关键词 '{keyword}' 时出错: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
