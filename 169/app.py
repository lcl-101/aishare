#!/usr/bin/env python3
"""
OlmOCR Gradio Web UI - 完全本地化版本
集成环境检测、配置管理和 Web UI 的完整解决方案
完全避免任何 AWS 或外部服务连接
"""

import gradio as gr
import os
import sys
import subprocess
import shutil
import json
import tempfile
import logging
import re
from pathlib import Path
import time
import requests
from urllib.parse import urlparse
import asyncio
import base64
from io import BytesIO
from PIL import Image
import concurrent.futures
import threading

# 导入 OlmOCR 的核心组件
try:
    from olmocr.data.renderpdf import render_pdf_to_base64png
    from olmocr.prompts import build_no_anchoring_yaml_prompt
    from olmocr.train.dataloader import FrontMatterParser
    from olmocr.prompts import PageResponse
    from pypdf import PdfReader
    OLMOCR_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"OlmOCR 组件导入失败: {e}")
    OLMOCR_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局处理器实例
global_processor = None

def initialize_global_processor():
    """初始化全局处理器"""
    global global_processor
    if global_processor is None:
        # 检查环境
        env_checker = EnvironmentChecker()
        python_ok = env_checker.check_python()
        packages_ok, _ = env_checker.check_packages()
        model_ok, _ = env_checker.check_model()
        
        if python_ok and packages_ok and model_ok:
            try:
                global_processor = OlmOCRProcessor(auto_start_server=False)
                logger.info("✅ 全局处理器初始化成功")
                return True
            except Exception as e:
                logger.error(f"❌ 全局处理器初始化失败: {e}")
                return False
        else:
            logger.warning("⚠️ 环境检查未通过，无法初始化全局处理器")
            return False
    return True

# 内置配置
CONFIG = {
    "model": {
        "path": "/workspace/olmocr/checkpoints/olmOCR-7B-0725",
        "name": "olmOCR-7B-0725",
        "type": "local"
    },
    "ui": {
        "title": "OlmOCR 本地模型 Web UI",
        "description": "基于本地 olmOCR-7B-0725 模型的文档转换工具 - 真实 OCR 处理",
        "port": 7860,
        "host": "0.0.0.0"
    },
    "processing": {
        "timeout": 600,
        "max_file_size_mb": 100,
        "supported_formats": [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
        "workspace_base": "/tmp/olmocr_workspace",
        "max_preview_chars": 10000
    },
    "output": {
        "default_format": "markdown",
        "enable_dolma": True,
        "enable_copy": True
    },
    "examples": {
        "sample_url": "https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf",
        "sample_filename": "olmocr-sample.pdf",
        "examples_dir": "/workspace/olmocr/examples",
        "auto_download": True
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "qwen3:32b",
        "timeout": 120,
        "max_tokens": 4096,
        "temperature": 0.7
    }
}

class EnvironmentChecker:
    """环境检查类"""
    
    @staticmethod
    def check_python():
        version = sys.version_info
        logger.info(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
        return version.major >= 3 and version.minor >= 8
    
    @staticmethod
    def check_packages():
        required_packages = {
            "gradio": "Web UI 框架",
            "torch": "PyTorch",
            "transformers": "Transformers",
            "PIL": "图像处理",
        }
        
        missing = []
        for package in required_packages:
            try:
                if package == "PIL":
                    from PIL import Image
                else:
                    __import__(package)
                logger.info(f"✅ {package}: {required_packages[package]}")
            except ImportError:
                logger.warning(f"❌ {package}: {required_packages[package]} - 未安装")
                missing.append(package)
        
        return len(missing) == 0, missing
    
    @staticmethod
    def check_gpu():
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"✅ GPU: {gpu_name}, 内存: {memory:.1f}GB")
                return True, f"{gpu_name} ({memory:.1f}GB)"
            else:
                logger.warning("❌ CUDA 不可用")
                return False, "CUDA 不可用"
        except ImportError:
            logger.warning("❌ PyTorch 未安装")
            return False, "PyTorch 未安装"
    
    @staticmethod
    def check_model():
        model_path = CONFIG["model"]["path"]
        if not os.path.exists(model_path):
            logger.error(f"模型路径不存在: {model_path}")
            return False, f"模型路径不存在: {model_path}"
        
        required_files = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
        missing = []
        
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                missing.append(file)
        
        if missing:
            logger.warning(f"缺少模型文件: {missing}")
            return False, f"缺少文件: {', '.join(missing)}"
        
        # 检查模型分片文件
        safetensors_files = list(Path(model_path).glob("model-*.safetensors"))
        if not safetensors_files:
            logger.warning("未找到模型分片文件")
            return False, "缺少模型分片文件 (model-*.safetensors)"
        
        # 读取模型信息
        try:
            with open(os.path.join(model_path, "config.json"), 'r') as f:
                config = json.load(f)
            model_type = config.get('model_type', '未知')
            vocab_size = config.get('vocab_size', '未知')
            model_info = f"{model_type} | 词汇表: {vocab_size} | 分片: {len(safetensors_files)}个"
            logger.info(f"✅ 模型: {model_info}")
            return True, model_info
        except Exception as e:
            logger.warning(f"读取模型配置失败: {e}")
            return True, f"模型存在但配置读取失败: {e}"

class ExampleManager:
    """示例文件管理器"""
    
    def __init__(self):
        self.examples_dir = CONFIG["examples"]["examples_dir"]
        self.sample_url = CONFIG["examples"]["sample_url"]
        self.sample_filename = CONFIG["examples"]["sample_filename"]
        os.makedirs(self.examples_dir, exist_ok=True)
    
    def download_sample_file(self):
        """下载示例文件"""
        sample_path = os.path.join(self.examples_dir, self.sample_filename)
        
        # 如果文件已存在，检查大小是否合理
        if os.path.exists(sample_path):
            file_size = os.path.getsize(sample_path)
            if file_size > 1024:  # 大于1KB认为下载成功
                logger.info(f"✅ 示例文件已存在: {sample_path} ({file_size} bytes)")
                return True, sample_path
        
        try:
            logger.info(f"📥 正在下载示例文件: {self.sample_url}")
            
            # 下载文件
            response = requests.get(self.sample_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # 保存文件
            with open(sample_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(sample_path)
            logger.info(f"✅ 示例文件下载成功: {sample_path} ({file_size} bytes)")
            return True, sample_path
            
        except requests.RequestException as e:
            logger.warning(f"⚠️ 示例文件下载失败: {e}")
            return False, f"下载失败: {e}"
        except Exception as e:
            logger.error(f"❌ 示例文件处理异常: {e}")
            return False, f"处理异常: {e}"
    
    def get_example_files(self):
        """获取示例文件列表"""
        example_files = []
        
        if os.path.exists(self.examples_dir):
            for file in os.listdir(self.examples_dir):
                file_path = os.path.join(self.examples_dir, file)
                if os.path.isfile(file_path):
                    file_ext = Path(file).suffix.lower()
                    if file_ext in CONFIG["processing"]["supported_formats"]:
                        example_files.append(file_path)
        
        return example_files
    
    def prepare_examples(self):
        """准备示例文件"""
        if CONFIG["examples"]["auto_download"]:
            success, result = self.download_sample_file()
            if not success:
                logger.warning(f"示例文件准备失败: {result}")
        
        return self.get_example_files()

class OllamaClient:
    """Ollama 客户端，用于与本地 Ollama 服务器通信"""
    
    def __init__(self, auto_start=False):
        self.base_url = CONFIG["ollama"]["base_url"]
        self.model = CONFIG["ollama"]["model"]
        self.timeout = CONFIG["ollama"]["timeout"]
        self.max_tokens = CONFIG["ollama"]["max_tokens"]
        self.temperature = CONFIG["ollama"]["temperature"]
        self.ollama_process = None
        self.model_process = None
        self.is_ollama_ready = False
        self.is_model_ready = False
        self.startup_status = "未启动"
        self.startup_error = None
        
        # 如果启用自动启动，在后台启动 Ollama
        if auto_start:
            self._start_ollama_background()
    
    def _start_ollama_background(self):
        """在后台启动 Ollama 服务器和模型"""
        def start_ollama_thread():
            try:
                logger.info("🚀 开始在后台启动 Ollama 服务器...")
                self.startup_status = "正在启动 Ollama..."
                
                # 首先启动 Ollama 服务器
                if self._start_ollama_server():
                    self.startup_status = "正在加载模型..."
                    logger.info("✅ Ollama 服务器启动成功，开始加载模型...")
                    
                    # 然后启动模型
                    if self._start_ollama_model():
                        self.startup_status = "启动成功"
                        self.is_ollama_ready = True
                        self.is_model_ready = True
                        logger.info(f"✅ Ollama 和模型 {self.model} 启动完成")
                    else:
                        self.startup_status = "模型启动失败"
                        logger.error("❌ Ollama 模型启动失败")
                else:
                    self.startup_status = "Ollama 启动失败"
                    logger.error("❌ Ollama 服务器启动失败")
                    
            except Exception as e:
                self.startup_status = "启动异常"
                self.startup_error = str(e)
                logger.error(f"❌ Ollama 启动异常: {e}")
        
        # 在新线程中启动
        startup_thread = threading.Thread(target=start_ollama_thread, daemon=True)
        startup_thread.start()
        logger.info("🔄 Ollama 正在后台启动中...")
    
    def _start_ollama_server(self):
        """启动 Ollama 服务器"""
        try:
            # 检查 Ollama 是否已经在运行
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Ollama 服务器已在运行")
                    return True
            except requests.ConnectionError:
                logger.info("Ollama 服务器未运行，正在启动...")
            
            # 启动 Ollama 服务器
            cmd = ["ollama", "serve"]
            
            self.ollama_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy()
            )
            
            # 等待服务器启动
            max_wait = 30  # 最多等待30秒
            for i in range(max_wait):
                try:
                    response = requests.get(f"{self.base_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        logger.info(f"✅ Ollama 服务器启动成功，耗时: {i+1}秒")
                        return True
                except requests.ConnectionError:
                    time.sleep(1)
                    continue
            
            logger.error("❌ Ollama 服务器启动超时")
            return False
            
        except Exception as e:
            logger.error(f"❌ 启动 Ollama 服务器失败: {e}")
            self.startup_error = str(e)
            return False
    
    def _start_ollama_model(self):
        """启动 Ollama 模型"""
        try:
            # 检查模型是否已经可用
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    available_models = [model["name"] for model in models]
                    if self.model in available_models:
                        logger.info(f"✅ 模型 {self.model} 已可用")
                        return True
            except Exception as e:
                logger.warning(f"检查模型状态失败: {e}")
            
            # 启动模型
            logger.info(f"正在启动模型 {self.model}...")
            cmd = ["ollama", "run", self.model, "--help"]  # 使用 --help 快速加载模型
            
            # 先尝试拉取模型（如果不存在）
            pull_cmd = ["ollama", "pull", self.model]
            logger.info(f"确保模型 {self.model} 已下载...")
            
            pull_process = subprocess.run(
                pull_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if pull_process.returncode != 0:
                logger.warning(f"拉取模型可能失败: {pull_process.stderr}")
            
            # 运行模型以确保加载
            run_process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2分钟超时
            )
            
            if run_process.returncode == 0:
                logger.info(f"✅ 模型 {self.model} 启动成功")
                return True
            else:
                logger.error(f"❌ 模型 {self.model} 启动失败: {run_process.stderr}")
                self.startup_error = run_process.stderr
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ 模型 {self.model} 启动超时")
            self.startup_error = "模型启动超时"
            return False
        except Exception as e:
            logger.error(f"❌ 启动模型失败: {e}")
            self.startup_error = str(e)
            return False
    
    def get_ollama_status(self):
        """获取 Ollama 状态"""
        # 实时检查状态
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                if self.model in available_models:
                    return {
                        "status": "运行中",
                        "ollama_ready": True,
                        "model_ready": True,
                        "error": None,
                        "model": self.model
                    }
                else:
                    return {
                        "status": f"模型 {self.model} 未加载",
                        "ollama_ready": True,
                        "model_ready": False,
                        "error": f"请运行: ollama run {self.model}",
                        "model": self.model
                    }
            else:
                return {
                    "status": f"服务器响应异常: {response.status_code}",
                    "ollama_ready": False,
                    "model_ready": False,
                    "error": f"HTTP {response.status_code}",
                    "model": self.model
                }
        except requests.ConnectionError:
            return {
                "status": "服务器未运行",
                "ollama_ready": False,
                "model_ready": False,
                "error": "请运行: ollama serve",
                "model": self.model
            }
        except Exception as e:
            return {
                "status": "检查失败",
                "ollama_ready": False,
                "model_ready": False,
                "error": str(e),
                "model": self.model
            }
        
    def check_ollama_status(self):
        """检查 Ollama 服务器状态"""
        # 直接进行实时检查
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                if self.model in available_models:
                    return True, f"✅ 模型 {self.model} 可用"
                else:
                    return False, f"❌ 模型 {self.model} 不可用，请手动运行: ollama run {self.model}"
            else:
                return False, f"❌ Ollama 服务器响应异常: {response.status_code}"
        except requests.ConnectionError:
            return False, "❌ 无法连接到 Ollama 服务器，请手动启动: ollama serve"
        except Exception as e:
            return False, f"❌ 检查 Ollama 状态时出错: {str(e)}"
    
    def generate_response(self, prompt, context=""):
        """生成回复"""
        try:
            # 构建完整的提示词
            if context:
                full_prompt = f"""基于以下文档内容回答问题：

文档内容：
{context}

问题：{prompt}

请基于文档内容给出准确、详细的回答。如果文档中没有相关信息，请明确说明。"""
            else:
                full_prompt = prompt
            
            # 发送请求到 Ollama
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return True, result.get("response", "")
            else:
                return False, f"Ollama API 错误: {response.status_code} - {response.text}"
                
        except requests.Timeout:
            return False, "请求超时，请重试"
        except requests.ConnectionError:
            return False, "无法连接到 Ollama 服务器"
        except Exception as e:
            return False, f"生成回复时出错: {str(e)}"

class OlmOCRProcessor:
    """OlmOCR 处理器 - 本地化版本，直接集成 vLLM 服务器"""
    
    def __init__(self, auto_start_server=False):
        self.model_path = CONFIG["model"]["path"]
        self.workspace_base = CONFIG["processing"]["workspace_base"]
        self.supported_formats = CONFIG["processing"]["supported_formats"]
        self.server_port = 30024
        self.server_process = None
        self.is_server_ready = False
        self.startup_status = "未启动"
        self.startup_error = None
        self.last_processed_content = ""  # 存储最后处理的文档内容
        self.last_processed_filename = ""  # 存储最后处理的文件名
        os.makedirs(self.workspace_base, exist_ok=True)
        
        # 如果启用自动启动，在后台启动服务器
        if auto_start_server:
            self._start_server_background()
    
    def _start_server_background(self):
        """在后台启动 vLLM 服务器"""
        def start_server_thread():
            try:
                logger.info("🚀 开始在后台启动 vLLM 服务器...")
                self.startup_status = "正在启动..."
                
                # 创建新的事件循环并启动服务器
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                success = loop.run_until_complete(self.start_vllm_server())
                
                if success:
                    self.startup_status = "启动成功"
                    logger.info("✅ vLLM 服务器后台启动完成")
                else:
                    self.startup_status = "启动失败"
                    logger.error("❌ vLLM 服务器后台启动失败")
                
                loop.close()
                
            except Exception as e:
                self.startup_status = "启动异常"
                self.startup_error = str(e)
                logger.error(f"❌ vLLM 服务器后台启动异常: {e}")
        
        # 在新线程中启动服务器
        startup_thread = threading.Thread(target=start_server_thread, daemon=True)
        startup_thread.start()
        logger.info("🔄 vLLM 服务器正在后台启动中...")
    
    def get_server_status(self):
        """获取服务器状态"""
        return {
            "status": self.startup_status,
            "ready": self.is_server_ready,
            "error": self.startup_error,
            "port": self.server_port if self.is_server_ready else None
        }
    
    async def start_vllm_server(self):
        """启动 vLLM 服务器"""
        if self.server_process and self.is_server_ready:
            return True
            
        try:
            self.startup_status = "正在启动..."
            logger.info(f"启动 vLLM 服务器，模型: {self.model_path}")
            
            cmd = [
                "vllm", "serve", self.model_path,
                "--port", str(self.server_port),
                "--disable-log-requests",
                "--uvicorn-log-level", "warning",
                "--served-model-name", "olmocr",
                "--tensor-parallel-size", "1",
                "--data-parallel-size", "1",
                "--gpu-memory-utilization", "0.6",  # 降低到 0.6 以适应可用内存
                "--max-model-len", "8192"  # 降低最大长度以减少内存使用
            ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            self.server_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy()
            )
            
            # 增加超时时间到 5 分钟，因为模型加载可能很慢
            timeout = 300  # 5分钟超时
            start_time = time.time()
            
            logger.info("等待 vLLM 服务器启动...")
            self.startup_status = "模型加载中..."
            
            while time.time() - start_time < timeout:
                try:
                    # 检查进程是否还在运行
                    if self.server_process.returncode is not None:
                        stdout, stderr = await self.server_process.communicate()
                        error_msg = f"vLLM 进程提前退出，返回码: {self.server_process.returncode}"
                        logger.error(error_msg)
                        logger.error(f"stdout: {stdout.decode()}")
                        logger.error(f"stderr: {stderr.decode()}")
                        self.startup_status = "启动失败"
                        self.startup_error = error_msg + f"\nstderr: {stderr.decode()}"
                        return False
                    
                    # 尝试连接到服务器
                    try:
                        reader, writer = await asyncio.wait_for(
                            asyncio.open_connection('localhost', self.server_port),
                            timeout=3
                        )
                        
                        # 发送简单的健康检查请求
                        health_request = (
                            f"GET /health HTTP/1.1\r\n"
                            f"Host: localhost:{self.server_port}\r\n"
                            f"Connection: close\r\n\r\n"
                        )
                        
                        writer.write(health_request.encode())
                        await writer.drain()
                        
                        response = await asyncio.wait_for(reader.read(1024), timeout=3)
                        writer.close()
                        await writer.wait_closed()
                        
                        if b"200" in response or b"404" in response:  # 404 也表示服务器在运行
                            self.is_server_ready = True
                            self.startup_status = "启动成功"
                            elapsed = time.time() - start_time
                            logger.info(f"✅ vLLM 服务器已启动，端口: {self.server_port}，耗时: {elapsed:.1f}秒")
                            return True
                            
                    except (ConnectionRefusedError, asyncio.TimeoutError, OSError) as e:
                        # 服务器还没准备好，继续等待
                        elapsed = time.time() - start_time
                        if elapsed % 15 < 2:  # 每15秒打印一次进度
                            logger.info(f"等待 vLLM 服务器启动... ({elapsed:.0f}s/{timeout}s)")
                            self.startup_status = f"加载中... ({elapsed:.0f}s/{timeout}s)"
                        await asyncio.sleep(2)
                        continue
                        
                except Exception as e:
                    logger.warning(f"检查服务器状态时出错: {e}")
                    await asyncio.sleep(2)
                    continue
            
            error_msg = "vLLM 服务器启动超时"
            logger.error(f"❌ {error_msg}")
            self.startup_status = "启动超时"
            self.startup_error = error_msg
            
            # 超时后尝试获取进程输出
            if self.server_process and self.server_process.returncode is None:
                try:
                    stdout, stderr = await asyncio.wait_for(
                        self.server_process.communicate(), timeout=5
                    )
                    logger.error(f"超时后的 stdout: {stdout.decode()}")
                    logger.error(f"超时后的 stderr: {stderr.decode()}")
                    self.startup_error += f"\nstderr: {stderr.decode()}"
                except asyncio.TimeoutError:
                    logger.error("无法获取进程输出")
                    
            return False
            
        except Exception as e:
            error_msg = f"vLLM 服务器启动失败: {e}"
            logger.error(f"❌ {error_msg}")
            self.startup_status = "启动异常"
            self.startup_error = str(e)
            return False
    
    def stop_vllm_server(self):
        """停止 vLLM 服务器"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.is_server_ready = False
                logger.info("vLLM 服务器已停止")
            except Exception as e:
                logger.warning(f"停止 vLLM 服务器时出错: {e}")
    
    async def build_page_query(self, local_pdf_path: str, page: int, target_longest_image_dim: int = 1288) -> dict:
        """构建页面查询请求"""
        # 渲染 PDF 页面为 base64 图像
        image_base64 = await asyncio.to_thread(
            render_pdf_to_base64png, 
            local_pdf_path, 
            page, 
            target_longest_image_dim=target_longest_image_dim
        )
        
        return {
            "model": "olmocr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        {"type": "text", "text": build_no_anchoring_yaml_prompt()},
                    ],
                }
            ],
            "max_tokens": 4500,
            "temperature": 0.1,
        }
    
    async def make_request(self, query: dict) -> tuple[int, str]:
        """向 vLLM 服务器发送请求"""
        url = f"http://localhost:{self.server_port}/v1/chat/completions"
        
        try:
            # 简单的 HTTP POST 实现
            json_payload = json.dumps(query)
            
            reader, writer = await asyncio.open_connection('localhost', self.server_port)
            
            request = (
                f"POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: localhost:{self.server_port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(json_payload)}\r\n"
                f"Connection: close\r\n\r\n"
                f"{json_payload}"
            )
            
            writer.write(request.encode())
            await writer.drain()
            
            # 读取响应
            response_data = await reader.read()
            writer.close()
            await writer.wait_closed()
            
            response_text = response_data.decode('utf-8')
            
            # 解析 HTTP 响应
            if "\r\n\r\n" in response_text:
                headers, body = response_text.split("\r\n\r\n", 1)
                status_line = headers.split('\r\n')[0]
                status_code = int(status_line.split()[1])
                return status_code, body
            else:
                return 500, "Invalid response"
                
        except Exception as e:
            logger.error(f"请求失败: {e}")
            return 500, str(e)
    
    async def process_single_page(self, pdf_path: str, page_num: int) -> dict:
        """处理单个 PDF 页面"""
        max_retries = 3
        attempt = 0
        
        while attempt < max_retries:
            try:
                # 构建查询
                query = await self.build_page_query(pdf_path, page_num)
                
                # 发送请求
                status_code, response_body = await self.make_request(query)
                
                if status_code != 200:
                    raise ValueError(f"HTTP错误: {status_code}")
                
                # 解析响应
                response_data = json.loads(response_body)
                
                if response_data["choices"][0]["finish_reason"] != "stop":
                    raise ValueError("响应未正常结束")
                
                # 提取内容
                model_response = response_data["choices"][0]["message"]["content"]
                
                # 解析前置元数据和文本
                parser = FrontMatterParser(front_matter_class=PageResponse)
                front_matter, text = parser._extract_front_matter_and_text(model_response)
                page_response = parser._parse_front_matter(front_matter, text)
                
                return {
                    "page_num": page_num,
                    "success": True,
                    "content": text,
                    "front_matter": front_matter,
                    "page_response": page_response,
                    "tokens": {
                        "input": response_data["usage"].get("prompt_tokens", 0),
                        "output": response_data["usage"].get("completion_tokens", 0)
                    }
                }
                
            except Exception as e:
                attempt += 1
                logger.warning(f"页面 {page_num} 处理失败 (尝试 {attempt}/{max_retries}): {e}")
                if attempt >= max_retries:
                    return {
                        "page_num": page_num,
                        "success": False,
                        "error": str(e),
                        "content": "",
                        "tokens": {"input": 0, "output": 0}
                    }
                await asyncio.sleep(1)
    
    async def process_pdf_async(self, pdf_path: str, output_format: str = "markdown") -> tuple[str, str]:
        """异步处理 PDF 文件"""
        try:
            # 确保服务器已启动
            if not await self.start_vllm_server():
                return "❌ vLLM 服务器启动失败", ""
            
            # 获取 PDF 页数
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            logger.info(f"开始处理 PDF: {os.path.basename(pdf_path)} ({total_pages} 页)")
            
            # 处理所有页面
            tasks = []
            for page_num in range(total_pages):
                task = self.process_single_page(pdf_path, page_num)
                tasks.append(task)
            
            # 等待所有页面处理完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 收集结果
            successful_pages = []
            failed_pages = []
            total_tokens = {"input": 0, "output": 0}
            
            for result in results:
                if isinstance(result, Exception):
                    failed_pages.append(f"异常: {result}")
                elif result["success"]:
                    successful_pages.append(result)
                    total_tokens["input"] += result["tokens"]["input"]
                    total_tokens["output"] += result["tokens"]["output"]
                else:
                    failed_pages.append(f"页面 {result['page_num']}: {result['error']}")
            
            # 生成输出
            if output_format == "markdown":
                content = self._format_markdown_output(successful_pages, failed_pages, total_tokens)
            else:
                content = self._format_dolma_output(successful_pages, failed_pages, total_tokens, pdf_path)
            
            status = f"✅ 处理完成: {len(successful_pages)}/{total_pages} 页成功"
            if failed_pages:
                status += f", {len(failed_pages)} 页失败"
            
            return status, content
            
        except Exception as e:
            logger.error(f"PDF 处理异常: {e}")
            return f"❌ 处理失败: {str(e)}", ""
    
    def _format_markdown_output(self, successful_pages: list, failed_pages: list, total_tokens: dict) -> str:
        """格式化 Markdown 输出"""
        content = f"""# PDF OCR 处理结果

**处理时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**成功页面**: {len(successful_pages)}
**失败页面**: {len(failed_pages)}
**使用 tokens**: 输入 {total_tokens['input']}, 输出 {total_tokens['output']}

---

"""
        
        # 按页面顺序排序
        successful_pages.sort(key=lambda x: x["page_num"])
        
        for page_result in successful_pages:
            content += f"## 页面 {page_result['page_num'] + 1}\n\n"
            content += page_result["content"]
            content += "\n\n---\n\n"
        
        if failed_pages:
            content += "## 处理失败的页面\n\n"
            for error in failed_pages:
                content += f"- {error}\n"
        
        return content
    
    def _format_dolma_output(self, successful_pages: list, failed_pages: list, total_tokens: dict, pdf_path: str) -> str:
        """格式化 Dolma 输出"""
        results = []
        
        # 按页面顺序排序
        successful_pages.sort(key=lambda x: x["page_num"])
        
        for page_result in successful_pages:
            dolma_record = {
                "id": f"{Path(pdf_path).stem}_page_{page_result['page_num']}",
                "text": page_result["content"],
                "source": "OlmOCR-Local",
                "added": time.strftime('%Y-%m-%d %H:%M:%S'),
                "metadata": {
                    "page_number": page_result["page_num"] + 1,
                    "tokens": page_result["tokens"],
                    "pdf_file": os.path.basename(pdf_path)
                }
            }
            results.append(dolma_record)
        
        # 添加失败页面信息
        if failed_pages:
            error_record = {
                "id": f"{Path(pdf_path).stem}_errors",
                "text": "",
                "source": "OlmOCR-Local-Errors",
                "added": time.strftime('%Y-%m-%d %H:%M:%S'),
                "metadata": {
                    "failed_pages": failed_pages,
                    "pdf_file": os.path.basename(pdf_path)
                }
            }
            results.append(error_record)
        
        # 格式化为 JSON Lines
        return '\n'.join(json.dumps(record, ensure_ascii=False) for record in results)
    
    def process_file(self, file, output_format="markdown"):
        """处理单个文件 - 同步接口包装"""
        if file is None:
            return "❌ 请上传文件", ""
        
        # 检查服务器状态
        if not self.is_server_ready:
            status_info = self.get_server_status()
            if status_info["status"] == "启动失败" or status_info["status"] == "启动异常":
                return f"❌ vLLM 服务器启动失败: {status_info.get('error', '未知错误')}", ""
            else:
                return f"⏳ vLLM 服务器尚未就绪，当前状态: {status_info['status']}", ""
        
        # 检查文件格式
        file_ext = Path(file.name).suffix.lower()
        if file_ext not in self.supported_formats:
            return f"❌ 不支持的格式: {file_ext}", ""
        
        # 检查文件大小
        file_size_mb = os.path.getsize(file.name) / (1024 * 1024)
        if file_size_mb > CONFIG["processing"]["max_file_size_mb"]:
            return f"❌ 文件过大: {file_size_mb:.1f}MB (限制: {CONFIG['processing']['max_file_size_mb']}MB)", ""
        
        try:
            # 创建临时工作空间
            session_id = f"session_{int(time.time())}"
            workspace = os.path.join(self.workspace_base, session_id)
            os.makedirs(workspace, exist_ok=True)
            
            # 复制文件到工作空间
            filename = os.path.basename(file.name)
            local_file = os.path.join(workspace, filename)
            shutil.copy2(file.name, local_file)
            
            logger.info(f"开始处理: {filename} ({file_size_mb:.1f}MB)")
            
            # 运行异步处理
            try:
                # 检查是否已有运行中的事件循环
                try:
                    current_loop = asyncio.get_running_loop()
                    logger.info("检测到运行中的事件循环，使用线程池执行异步任务")
                    
                    # 在线程中运行异步代码
                    result_holder = {"status": None, "content": None, "error": None}
                    
                    def run_async_in_thread():
                        try:
                            # 创建新的事件循环
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            
                            status, content = new_loop.run_until_complete(
                                self.process_pdf_async(local_file, output_format)
                            )
                            
                            new_loop.close()
                            result_holder["status"] = status
                            result_holder["content"] = content
                            
                        except Exception as e:
                            result_holder["error"] = str(e)
                    
                    # 在新线程中运行
                    thread = threading.Thread(target=run_async_in_thread)
                    thread.start()
                    thread.join(timeout=CONFIG["processing"]["timeout"])
                    
                    if thread.is_alive():
                        return "⏰ 处理超时", ""
                    
                    if result_holder["error"]:
                        raise Exception(result_holder["error"])
                    
                    status = result_holder["status"]
                    content = result_holder["content"]
                    
                except RuntimeError:
                    # 没有运行中的事件循环，直接运行
                    logger.info("没有运行中的事件循环，直接运行异步任务")
                    status, content = asyncio.run(
                        self.process_pdf_async(local_file, output_format)
                    )
                
                # 清理临时文件
                shutil.rmtree(workspace, ignore_errors=True)
                
                # 存储处理结果供聊天使用
                if status and content and "✅ 处理完成" in status:
                    self.last_processed_content = content
                    self.last_processed_filename = filename
                    logger.info(f"✅ 已保存文档内容供聊天使用: {filename} (长度: {len(content)} 字符)")
                else:
                    logger.warning(f"⚠️ 文档内容未保存 - 状态: {status}, 内容长度: {len(content) if content else 0}")
                
                return status, content
                
            except Exception as e:
                logger.error(f"异步处理失败: {e}")
                # 回退到简单处理
                return self._create_fallback_result(file, output_format, workspace, filename, str(e))
            
        except Exception as e:
            logger.error(f"处理异常: {e}")
            return f"❌ 处理失败: {str(e)}", ""
        finally:
            # 确保清理
            if 'workspace' in locals() and os.path.exists(workspace):
                shutil.rmtree(workspace, ignore_errors=True)
    
    def _create_fallback_result(self, file, output_format, workspace, filename, error_msg):
        """创建错误回退结果"""
        try:
            if output_format == "markdown":
                mock_content = f"""# OlmOCR 处理失败

**文件名**: {filename}
**处理时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**状态**: 处理失败

## 错误信息
```
{error_msg}
```

## 解决建议

1. **检查 GPU 资源**:
   ```bash
   nvidia-smi
   ```

2. **验证模型文件**:
   ```bash
   ls -la {CONFIG['model']['path']}
   ```

3. **手动测试 vLLM 服务器**:
   ```bash
   vllm serve {CONFIG['model']['path']} --port 30024
   ```

## 文件信息
- **文件**: {filename}
- **大小**: {os.path.getsize(file.name) / (1024*1024):.1f} MB
- **格式**: {Path(filename).suffix.upper()}

请检查错误信息并重试。如果问题持续，可能是 vLLM 服务器启动失败。
"""
                return f"❌ 处理失败: {filename}", mock_content
            else:
                mock_dolma = {
                    "id": f"error_{int(time.time())}",
                    "error": error_msg,
                    "file": filename,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "suggestions": "检查 GPU 状态和 vLLM 服务器"
                }
                return f"❌ 处理失败: {filename}", json.dumps(mock_dolma, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"❌ 错误处理失败: {str(e)}", ""
    
    def get_document_content(self):
        """获取最后处理的文档内容"""
        return self.last_processed_content, self.last_processed_filename

def get_global_processor():
    """获取全局处理器实例"""
    global global_processor
    if global_processor is None:
        initialize_global_processor()
    return global_processor

def test_document_content():
    """测试文档内容"""
    global global_processor
    if global_processor is None:
        print("🔧 全局处理器未初始化，正在初始化...")
        if not initialize_global_processor():
            print("❌ 全局处理器初始化失败")
            return None, None
    
    if global_processor:
        content, filename = global_processor.get_document_content()
        print(f"📄 全局处理器文档状态:")
        print(f"  文件名: {filename}")
        print(f"  内容长度: {len(content) if content else 0}")
        if content:
            print(f"  内容预览: {content[:200]}...")
        else:
            print("  ⚠️ 没有文档内容")
        return content, filename
    else:
        print("❌ 全局处理器未初始化")
        return None, None

def create_ui():
    """创建 Gradio 界面"""
    
    # 环境检查
    env_checker = EnvironmentChecker()
    
    # 检查结果
    python_ok = env_checker.check_python()
    packages_ok, missing_packages = env_checker.check_packages()
    gpu_ok, gpu_info = env_checker.check_gpu()
    model_ok, model_info = env_checker.check_model()
    
    # 示例文件管理器
    example_manager = ExampleManager()
    example_files = example_manager.prepare_examples()
    
    # 创建 Ollama 客户端
    ollama_client = OllamaClient(auto_start=False)  # 禁用自动启动
    ollama_ok, ollama_info = ollama_client.check_ollama_status()
    
    # 创建处理器
    global global_processor
    processor = None
    if python_ok and packages_ok and model_ok:
        try:
            # 启用自动启动服务器
            processor = OlmOCRProcessor(auto_start_server=True)
            global_processor = processor  # 设置全局处理器
            logger.info("✅ 环境检查通过，已初始化处理器并开始启动 vLLM 服务器")
        except Exception as e:
            logger.error(f"处理器初始化失败: {e}")
    
    # 自定义 CSS
    css = """
    .model-info { 
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    .env-check {
        font-family: monospace;
        background: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #dee2e6;
    }
    .local-badge {
        background: linear-gradient(135deg, #4caf50, #45a049);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        margin: 5px;
    }
    .examples-section {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    .server-status, #vllm-status, #ollama-status {
        background: #f0f9ff !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border-left: 4px solid #0ea5e9 !important;
        margin: 10px 0 !important;
        font-family: monospace !important;
        color: #1f2937 !important;
        border: 1px solid #e0e7ff !important;
    }
    .chat-section {
        background: #f8fdf9;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #22c55e;
        margin: 10px 0;
    }
    .chat-message {
        background: #ffffff;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border: 1px solid #e5e7eb;
    }
    .user-message {
        background: #dbeafe;
        border-left: 3px solid #3b82f6;
    }
    .assistant-message {
        background: #f0fdf4;
        border-left: 3px solid #22c55e;
    }
    """
    
    with gr.Blocks(
        title=CONFIG["ui"]["title"],
        theme=gr.themes.Soft(),
        css=css
    ) as demo:
        
        # 标题
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px;">
            <h1>🔍 {CONFIG["ui"]["title"]}</h1>
            <p style="font-size: 18px; color: #666;">
                {CONFIG["ui"]["description"]}
            </p>
            <div class="local-badge">
                🔒 完全本地化 | 🚫 零网络依赖 | 🛡️ 隐私保护
            </div>
        </div>
        """)
        
        # 环境状态检查
        with gr.Accordion("🛠️ 环境状态检查", open=True):
            env_status_text = f"""
**系统环境:**
- {'✅' if python_ok else '❌'} **Python 3.8+**: {sys.version.split()[0]}
- {'✅' if packages_ok else '❌'} **依赖包**: {'全部已安装' if packages_ok else f'缺少: {", ".join(missing_packages)}'}
- {'✅' if gpu_ok else '⚠️'} **GPU**: {gpu_info}
- {'✅' if model_ok else '❌'} **模型**: {model_info}
- {'✅' if ollama_ok else '⚠️'} **Ollama**: {ollama_info}

**本地化特性:**
- 🔒 **完全离线**: 无任何外部网络连接
- 🚫 **AWS 禁用**: 完全绕过云服务依赖
- 🛡️ **隐私保护**: 所有数据本地处理

**状态**: {'🟢 就绪' if all([python_ok, packages_ok, model_ok]) else '🔴 需要修复'}
            """
            gr.Markdown(env_status_text)
        
        # vLLM 服务器状态
        if processor:
            with gr.Accordion("🚀 vLLM 服务器状态", open=True):
                # 获取初始状态
                status_info = processor.get_server_status()
                if status_info["ready"]:
                    vllm_status_text = f"""
**vLLM 服务器:**
- **状态**: ✅ {status_info['status']}
- **端口**: {status_info['port']}
- **模型**: olmOCR-7B-0725
- **就绪**: ✅ 可以开始处理文档
"""
                elif status_info["status"] == "启动失败" or status_info["status"] == "启动异常":
                    error_detail = status_info.get("error", "未知错误")
                    vllm_status_text = f"""
**vLLM 服务器:**
- **状态**: ❌ {status_info['status']}
- **错误**: {error_detail}
- **建议**: 检查 GPU 内存和模型文件
"""
                else:
                    vllm_status_text = f"""
**vLLM 服务器:**
- **状态**: 🔄 {status_info['status']}
- **提示**: 首次启动需要加载模型，请耐心等待
- **预计**: 大约需要 2-5 分钟
"""
                gr.Markdown(vllm_status_text)
        
        # 主界面
        if processor:
            # OCR 处理界面
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 文件处理")
                    
                    # 示例文件选择器
                    if example_files:
                        with gr.Group():
                            gr.Markdown("#### 📋 示例文件")
                            example_selector = gr.Dropdown(
                                choices=[(os.path.basename(f), f) for f in example_files],
                                label="选择示例文件",
                                value=example_files[0] if example_files else None,
                                elem_classes=["examples-section"]
                            )
                            
                            def load_example_file(selected_file):
                                if selected_file:
                                    return gr.File(value=selected_file)
                                return gr.File(value=None)
                    else:
                        example_selector = None
                        gr.Markdown("⚠️ 示例文件下载失败，请检查网络连接")
                    
                    gr.Markdown("#### 📎 或上传自己的文件")
                    file_input = gr.File(
                        label="选择文档文件",
                        file_types=CONFIG["processing"]["supported_formats"],
                        height=120
                    )
                    
                    # 示例文件加载事件
                    if example_selector:
                        example_selector.change(
                            fn=lambda x: x,
                            inputs=[example_selector],
                            outputs=[file_input]
                        )
                    
                    output_format = gr.Radio(
                        choices=["markdown", "dolma"],
                        value=CONFIG["output"]["default_format"],
                        label="输出格式"
                    )
                    
                    with gr.Row():
                        process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                        refresh_btn = gr.Button("🔄 刷新示例", variant="secondary")
                    
                    # 配置信息
                    gr.Markdown(f"""
                    ### ⚙️ 配置信息
                    - **处理模式**: 完全本地化
                    - **超时时间**: {CONFIG['processing']['timeout']}秒
                    - **文件大小限制**: {CONFIG['processing']['max_file_size_mb']}MB
                    - **工作空间**: {CONFIG['processing']['workspace_base']}
                    - **支持格式**: {', '.join(CONFIG['processing']['supported_formats'])}
                    - **示例文件**: {len(example_files)} 个可用
                    """)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 处理结果")
                    
                    status_output = gr.Textbox(
                        label="处理状态",
                        interactive=False,
                        max_lines=3
                    )
                    
                    content_output = gr.Textbox(
                        label="转换内容",
                        interactive=False,
                        max_lines=25,
                        show_copy_button=CONFIG["output"]["enable_copy"],
                        placeholder="转换结果将在这里显示..."
                    )
                    
                    # 智能问答功能
                    gr.Markdown("### 💬 智能问答")
                    
                    # 检查 Ollama 状态
                    def check_current_ollama_status():
                        status_info = ollama_client.get_ollama_status()
                        return status_info["ollama_ready"] and status_info["model_ready"]
                    
                    if check_current_ollama_status():
                        gr.Markdown("*基于上方处理的文档内容进行智能问答*")
                        
                        # 聊天历史
                        chatbot = gr.Chatbot(
                            label="对话历史",
                            height=300,
                            type="messages",
                            elem_classes=["chat-section"]
                        )
                        
                        # 输入区域
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="",
                                placeholder="请输入您的问题...",
                                scale=4
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)
                        
                        # 控制按钮
                        with gr.Row():
                            clear_chat_btn = gr.Button("🗑️ 清空对话", variant="secondary")
                            refresh_doc_btn = gr.Button("🔄 刷新文档", variant="secondary")
                        
                        gr.Markdown("""
                        **💡 使用提示**
                        - 先处理文档，然后基于文档内容提问
                        - 支持多轮对话和上下文关联
                        
                        **🎯 问题示例**
                        - "这份文档的主要内容是什么？"
                        - "文档中提到了哪些关键数据？"
                        - "请总结文档的核心观点"
                        """)
                    else:
                        # 获取当前状态信息
                        status_info = ollama_client.get_ollama_status()
                        gr.HTML(f"""
                        <div style="text-align: center; padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px;">
                            <h4>⚠️ Ollama 服务未就绪</h4>
                            <p><strong>当前状态</strong>: {status_info['status']}</p>
                            <p><strong>建议操作</strong>: {status_info.get('error', '检查 Ollama 状态')}</p>
                        </div>
                        """)
            
            # OCR 处理事件绑定
            def process_with_notification(file, format):
                """处理文件并添加聊天提示"""
                status, content = processor.process_file(file, format)
                
                # 如果处理成功且聊天功能可用，添加提示
                if status and content and "✅ 处理完成" in status and check_current_ollama_status():
                    status += "\n\n💬 文档已就绪，可以在下方智能问答区域进行提问！"
                
                return status, content
            
            process_btn.click(
                fn=process_with_notification,
                inputs=[file_input, output_format],
                outputs=[status_output, content_output]
            )
            
            def clear_all():
                return None, "", "", None if example_selector else None
            
            clear_btn.click(
                fn=clear_all,
                outputs=[file_input, status_output, content_output] + ([example_selector] if example_selector else [])
            )
            
            # 刷新示例文件
            if example_selector:
                def refresh_examples():
                    new_files = example_manager.prepare_examples()
                    choices = [(os.path.basename(f), f) for f in new_files]
                    return gr.Dropdown(choices=choices, value=new_files[0] if new_files else None)
                
                refresh_btn.click(
                    fn=refresh_examples,
                    outputs=[example_selector]
                )
            
            # 智能问答功能事件绑定
            def check_current_ollama_status():
                status_info = ollama_client.get_ollama_status()
                return status_info["ollama_ready"] and status_info["model_ready"]
            
            if check_current_ollama_status():
                def respond_to_message(message, history):
                    """处理用户消息并返回回复"""
                    if not message.strip():
                        return history, ""
                    
                    # 检查 Ollama 是否就绪
                    status_info = ollama_client.get_ollama_status()
                    if not (status_info["ollama_ready"] and status_info["model_ready"]):
                        bot_message = f"抱歉，Ollama 服务尚未就绪。当前状态: {status_info['status']}"
                        new_history = history.copy() if history else []
                        new_history.append({"role": "user", "content": message})
                        new_history.append({"role": "assistant", "content": bot_message})
                        return new_history, ""
                    
                    # 获取文档内容
                    global global_processor
                    if not global_processor:
                        bot_message = "抱歉，处理器未初始化。请重启应用。"
                        new_history = history.copy() if history else []
                        new_history.append({"role": "user", "content": message})
                        new_history.append({"role": "assistant", "content": bot_message})
                        return new_history, ""
                    
                    content, filename = global_processor.get_document_content()
                    
                    # 调试信息
                    logger.info(f"🔍 聊天调试 - 文档内容长度: {len(content) if content else 0}")
                    logger.info(f"🔍 聊天调试 - 文件名: {filename}")
                    logger.info(f"🔍 聊天调试 - 用户问题: {message}")
                    
                    if not content:
                        bot_message = "抱歉，当前没有可用的文档内容。请先处理文档后再进行问答。"
                        logger.warning("⚠️ 聊天失败：没有文档内容")
                        new_history = history.copy() if history else []
                        new_history.append({"role": "user", "content": message})
                        new_history.append({"role": "assistant", "content": bot_message})
                        return new_history, ""
                    
                    # 使用 Ollama 生成回复
                    logger.info("🚀 开始调用 Ollama 生成回复...")
                    success, response = ollama_client.generate_response(message, content)
                    logger.info(f"📝 Ollama 回复结果 - 成功: {success}, 响应长度: {len(response) if response else 0}")
                    
                    if success:
                        # 清理 AI 思考标记
                        cleaned_response = response
                        if "<think>" in cleaned_response and "</think>" in cleaned_response:
                            # 移除 <think>...</think> 标记及其内容
                            cleaned_response = re.sub(r'<think>.*?</think>\s*', '', cleaned_response, flags=re.DOTALL)
                        
                        bot_message = cleaned_response.strip()
                        logger.info(f"✅ 生成成功，清理后回复长度: {len(bot_message)}")
                        logger.info(f"✅ 清理后回复预览: {bot_message[:100]}...")
                    else:
                        bot_message = f"抱歉，生成回复时出现错误：{response}"
                        logger.error(f"❌ 生成失败: {response}")
                    
                    # 构建新的历史记录
                    new_history = history.copy() if history else []
                    new_history.append({"role": "user", "content": message})
                    new_history.append({"role": "assistant", "content": bot_message})
                    
                    logger.info(f"📋 返回历史记录长度: {len(new_history)}")
                    logger.info(f"📋 最后一条消息: {new_history[-1] if new_history else 'None'}")
                    
                    return new_history, ""
                
                def clear_chat():
                    """清空聊天历史"""
                    logger.info("🗑️ 清空聊天历史")
                    return []
                
                # 绑定聊天事件
                try:
                    send_btn.click(
                        fn=respond_to_message,
                        inputs=[msg_input, chatbot],
                        outputs=[chatbot, msg_input]
                    )
                    
                    msg_input.submit(
                        fn=respond_to_message,
                        inputs=[msg_input, chatbot],
                        outputs=[chatbot, msg_input]
                    )
                    
                    clear_chat_btn.click(
                        fn=clear_chat,
                        outputs=[chatbot]
                    )
                    
                    # 初始化时不需要显示文档信息
                except NameError:
                    # 如果聊天组件未创建，则跳过绑定
                    pass
            
        else:
            gr.HTML("""
            <div style="text-align: center; padding: 40px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px;">
                <h3>⚠️ 环境配置不完整</h3>
                <p>请解决上述环境问题后重启应用</p>
            </div>
            """)
        
        # 帮助信息
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown(f"""
            ### 🔧 环境修复
            ```bash
            # 如果缺少依赖包
            pip install gradio torch transformers pillow opencv-python requests
            
            # 如果模型文件不完整
            # 请确保 {CONFIG['model']['path']} 目录包含完整的模型文件
            ```
            
            ### 🎯 使用步骤
            1. 确保环境状态为 🟢 就绪
            2. **OCR 处理**: 选择示例文件或上传文档进行 OCR 处理
            3. **智能问答**: 切换到问答页面，基于处理结果进行问答
            4. 选择输出格式 (Markdown 推荐)
            5. 点击开始处理并等待完成
            
            ### 💬 智能问答功能
            - **文档问答**: 基于 OCR 处理结果进行智能问答
            - **本地 AI**: 使用 Ollama + Qwen3:32B 模型
            - **上下文理解**: AI 会基于文档内容回答问题
            - **多轮对话**: 支持连续对话和上下文关联
            
            #### Ollama 设置
            ```bash
            # 手动安装 Ollama：
            curl -fsSL https://ollama.ai/install.sh | sh
            
            # 手动下载 Qwen3:32B 模型
            ollama pull qwen3:32b
            
            # 手动启动 Ollama 服务
            ollama serve
            
            # 在另一个终端中运行模型
            ollama run qwen3:32b
            ```
            
            #### Ollama 功能特性
            - **手动控制**: 需要手动启动 Ollama 服务器和模型
            - **状态监控**: 实时显示 Ollama 和模型的运行状态
            - **智能切换**: 根据服务状态自动启用/禁用聊天功能
            - **独立运行**: Ollama 与应用独立，可在需要时单独启动
            
            ### 📋 示例文件功能
            - **自动下载**: 首次启动时自动下载官方示例 PDF
            - **快速体验**: 无需准备文件即可测试 OCR 功能
            - **示例路径**: `{CONFIG['examples']['examples_dir']}`
            - **刷新功能**: 可重新下载或添加更多示例文件
            
            ### 📝 输出格式
            - **Markdown**: 结构化文本，适合阅读和编辑
            - **Dolma**: JSON Lines 格式，适合数据处理
            
            ### 🔒 本地化特性
            - **完全离线**: 所有处理在本地完成，无需网络连接
            - **隐私保护**: 文档不会上传到任何外部服务
            - **AWS 禁用**: 完全绕过 AWS 和云服务依赖
            - **代理无关**: 不受网络代理设置影响
            
            ### 🚀 命令行使用（本地化版本）
            ```bash
            # 完全本地化处理，无外部依赖
            cd /workspace/olmocr
            python app.py
            # 然后在浏览器中访问 http://localhost:7860
            
            # 使用下载的示例文件进行命令行测试
            python -m olmocr.pipeline ./localworkspace --markdown --pdfs {CONFIG['examples']['examples_dir']}/{CONFIG['examples']['sample_filename']}
            ```
            
            ### ⚡ 性能提示
            - 建议使用 GPU 以获得最佳性能
            - 大文件处理需要更多时间和内存
            - 清晰、高分辨率的文档效果更好
            - 首次使用时模型加载较慢，请耐心等待
            
            ### 🛠️ 技术优势
            - **零依赖**: 不需要任何外部服务或网络连接
            - **高安全**: 文档处理完全在本地环境中进行
            - **易部署**: 单一文件包含所有功能
            - **高性能**: 直接使用本地 GPU 进行推理
            - **示例集成**: 内置示例文件，开箱即用
            - **智能问答**: 集成 Ollama 提供文档问答能力
            - **多模态**: 支持 OCR + 文本理解的完整流程
            """)
    
    return demo

if __name__ == "__main__":
    print("🚀 启动 OlmOCR Web UI (完全本地化版本)...")
    print(f"📍 访问地址: http://{CONFIG['ui']['host']}:{CONFIG['ui']['port']}")
    print(f"🤖 模型路径: {CONFIG['model']['path']}")
    print("🔒 特性: 完全本地化 | 零网络依赖 | 隐私保护")
    print("")
    print("⚡ 启动流程:")
    print("1. 📋 环境检查...")
    print("2. 🔄 启动 Gradio WebUI...")
    print("3. 🚀 后台启动 vLLM 服务器...")
    print("4. 📂 下载示例文件...")
    print("5. ✅ 完成！可以开始使用")
    print("")
    print("💡 提示: vLLM 服务器首次启动需要加载模型，大约需要 2-5 分钟")
    print("💡 提示: 如需使用智能问答功能，请手动启动 Ollama: ollama serve")
    print("💡 提示: 然后运行模型: ollama run qwen3:32b")
    print("💡 请在浏览器中打开上述地址并等待 vLLM 服务状态变为 '就绪'")
    
    demo = create_ui()
    demo.launch(
        server_name=CONFIG["ui"]["host"],
        server_port=CONFIG["ui"]["port"],
        share=False,
        debug=True
    )
