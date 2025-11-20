# Youtube 节目：
## NVIDIA 最强开源语音识别模型！免费可商用，性能媲美顶级服务，本地部署 Canary-Qwen-2.5b 保姆级教程
## https://youtu.be/MlNZgdTI08I

# 安装指南
## 克隆项目
mkdir canary  
cd canary  

## 创建运行环境
conda create -n canary python=3.10 -y  
conda activate canary  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
python -m pip install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"  
pip install gradio sacrebleu  

## 模型下载
hf download nvidia/canary-qwen-2.5b --local-dir checkpoints/canary-qwen-2.5b  

## 推理演示
python app.py      

  












 
















