# Youtube 节目：
## 秒杀Whisper？英伟达开源音频理解模型Flamingo，不只转录，更能和你聊音乐！
## https://youtu.be/KG3c0Mk-pVI

# 安装指南
## 创建运行环境
conda create -n flamingo python=3.10 -y  
conda activate flamingo  

##安装 lfs
apt install git-lfs  
git lfs install  

## 克隆文件
git clone https://huggingface.co/spaces/nvidia/audio-flamingo-3 flamingo  
cd flamingo  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  
pip install ollama  

## 准备模型文件
huggingface-cli download nvidia/audio-flamingo-3 --local-dir checkpoints/audio-flamingo-3  
huggingface-cli download nvidia/audio-flamingo-3-chat --local-dir checkpoints/audio-flamingo-chat  

## 推理
python app.py  




  












 
















