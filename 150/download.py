#!/usr/bin/env python3
"""
下载 Kimi-VL Demo 示例文件到 examples 目录
"""

import os
import requests
from urllib.parse import urlparse
from pathlib import Path
import time

# 示例文件URL映射
DEMO_FILES = {
    # Demo 1: 基础图像分析 - 猫咪识别
    "demo1_cat.jpeg": "https://huggingface.co/spaces/moonshotai/Kimi-VL-A3B-Thinking/resolve/main/images/demo6.jpeg",
    
    # Demo 2: 图像分析 - 儿童服装颜色
    "demo2_child.jpg": "https://vstar-seal.github.io/images/examples/sample4/origin.jpg",
    
    # Demo 3: 表格理解 - MathVista基准测试
    "demo3_mathvista.jpg": "https://raw.githubusercontent.com/mathvista/data/main/images/4781.jpg",
    
    # Demo 4: 数学推理 - 数字谜题
    "demo4_math_puzzle.jpg": "https://mathllm.github.io/mathvision/visualizer/data/images/64.jpg",
    
    # Demo 5: GUI Agent - 屏幕截图
    "demo5_screenshot.png": "https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506/resolve/main/figures/screenshot.png",
    
    # Demo 6: PDF分析 - arXiv论文
    "demo6_sample_paper.pdf": "https://arxiv.org/pdf/2505.23359.pdf",
    
    # Demo 7: 视频分析 - 场景分割
    "demo7_video.mp4": "https://cdn-uploads.huggingface.co/production/uploads/63047ed2412a1b9d381b09c9/vHt27D34wLJMyujNsnTDZ.mp4",
    
    # 额外的示例文件
    "extra_logo.png": "https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506/resolve/main/figures/logo.png",
    "extra_arch.png": "https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506/resolve/main/figures/arch.png",
    "extra_demo1.png": "https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506/resolve/main/figures/demo1.png",
    "extra_demo2.png": "https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506/resolve/main/figures/demo2.png",
    "extra_thinking_perf.png": "https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506/resolve/main/figures/thinking_perf.png",
}

def create_examples_dir():
    """创建 examples 目录"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    print(f"📁 Examples directory: {examples_dir.absolute()}")
    return examples_dir

def get_file_size(url):
    """获取远程文件大小"""
    try:
        response = requests.head(url, timeout=10)
        content_length = response.headers.get('content-length')
        if content_length:
            return int(content_length)
    except:
        pass
    return None

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes is None:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def download_file(url, file_path, filename):
    """下载单个文件"""
    if file_path.exists():
        print(f"✅ {filename} already exists (size: {format_size(file_path.stat().st_size)})")
        return True
    
    try:
        print(f"📥 Downloading {filename}...")
        
        # 获取文件大小
        file_size = get_file_size(url)
        if file_size:
            print(f"   File size: {format_size(file_size)}")
        
        # 下载文件
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # 写入文件
        downloaded_size = 0
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # 显示进度（仅当知道文件大小时）
                    if file_size and downloaded_size % (1024 * 1024) == 0:  # 每MB显示一次
                        progress = (downloaded_size / file_size) * 100
                        print(f"   Progress: {progress:.1f}% ({format_size(downloaded_size)}/{format_size(file_size)})")
        
        final_size = file_path.stat().st_size
        print(f"✅ {filename} downloaded successfully (size: {format_size(final_size)})")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {filename}: {e}")
        # 删除不完整的文件
        if file_path.exists():
            file_path.unlink()
        return False
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        if file_path.exists():
            file_path.unlink()
        return False

def download_all_files(file_dict, examples_dir):
    """下载所有文件"""
    total_files = len(file_dict)
    successful = 0
    failed = []
    
    print(f"🚀 Starting download of {total_files} files...\n")
    
    for i, (filename, url) in enumerate(file_dict.items(), 1):
        print(f"[{i}/{total_files}] Processing {filename}")
        file_path = examples_dir / filename
        
        if download_file(url, file_path, filename):
            successful += 1
        else:
            failed.append(filename)
        
        print()  # 空行分隔
        
        # 短暂延迟避免过于频繁的请求
        if i < total_files:
            time.sleep(0.5)
    
    return successful, failed

def main():
    """主函数"""
    print("🌟 Kimi-VL Demo Files Downloader")
    print("=" * 50)
    
    # 创建目录
    examples_dir = create_examples_dir()
    
    # 显示将要下载的文件列表
    print("\n📋 Files to download:")
    for filename, url in DEMO_FILES.items():
        print(f"  • {filename}")
        print(f"    Source: {url}")
        
        # 检查文件是否已存在
        file_path = examples_dir / filename
        if file_path.exists():
            size = format_size(file_path.stat().st_size)
            print(f"    Status: ✅ Already exists ({size})")
        else:
            print(f"    Status: ⏳ Need to download")
        print()
    
    # 确认下载
    print(f"📊 Total files: {len(DEMO_FILES)}")
    choice = input("\n🤔 Do you want to proceed with the download? (y/N): ").strip().lower()
    
    if choice not in ['y', 'yes']:
        print("❌ Download cancelled.")
        return
    
    print("\n" + "=" * 50)
    
    # 开始下载
    start_time = time.time()
    successful, failed = download_all_files(DEMO_FILES, examples_dir)
    end_time = time.time()
    
    # 显示结果
    print("=" * 50)
    print("📊 Download Summary:")
    print(f"  ✅ Successful: {successful}/{len(DEMO_FILES)}")
    print(f"  ❌ Failed: {len(failed)}")
    print(f"  ⏱️ Time taken: {end_time - start_time:.1f} seconds")
    
    if failed:
        print(f"\n❌ Failed downloads:")
        for filename in failed:
            print(f"  • {filename}")
        print("\n💡 You can run this script again to retry failed downloads.")
    else:
        print("\n🎉 All files downloaded successfully!")
    
    # 显示文件列表
    print(f"\n📁 Files in examples directory:")
    for file_path in sorted(examples_dir.iterdir()):
        if file_path.is_file():
            size = format_size(file_path.stat().st_size)
            print(f"  • {file_path.name} ({size})")

if __name__ == "__main__":
    main()
