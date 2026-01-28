# Youtube 节目：
## AI的五官终于齐了！本地部署阿里最强开源多模三态模型Qwen3-Omni，实时语音对话，看懂一切音视频！
## https://youtu.be/8-SXFXvSDrY

# 安装指南
## 克隆项目
git clone https://github.com/QwenLM/Qwen3-Omni.git  
cd Qwen3-Omni  

## 创建运行环境
apt-get update && apt-get install -y ffmpeg  
conda create -n Qwen3-Omni python=3.10 -y  
conda activate Qwen3-Omni  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install transformers accelerate  
pip install qwen-omni-utils -U  
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  
pip install gradio  

## 模型下载
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir checkpoints/Qwen3-Omni-30B-A3B-Instruct  
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Thinking --local-dir checkpoints/Qwen3-Omni-30B-A3B-Thinking  
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Captioner --local-dir checkpoints/Qwen3-Omni-30B-A3B-Captioner  

## 推理演示
python app.py    

  












 
















