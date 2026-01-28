# Youtube 节目：
## 70分钟录音2分钟转完！NVIDIA 开源“最强顺风耳”Canary-Qwen-2.5b：集语音精准转写与智能总结于一体，完全免费可商用！
## https://youtu.be/WqMkWmSATnU

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

  












 
















