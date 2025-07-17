# Youtube 节目：
## 告别静态头像！腾讯开源 Sonic：AI 驱动人像，让你的照片“声”动起来！
## https://youtu.be/XeShMd3QNx4

# 安装指南

## 创建运行环境 
git clone https://github.com/jixiaozhong/Sonic.git  
cd Sonic  
conda create -n Sonic python=3.10 -y  
conda activate Sonic  
pip install -r requirements.txt  
pip install opencv-python  
pip install accelerate  
conda install -c conda-forge ffmpeg -y  

## 准备模型文件
python3 -m pip install "huggingface_hub[cli]"  
huggingface-cli download LeonJoe13/Sonic --local-dir  checkpoints  
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir  checkpoints/stable-video-diffusion-img2vid-xt  
huggingface-cli download openai/whisper-tiny --local-dir checkpoints/whisper-tiny  

## 推理
python gradio_app.py  







