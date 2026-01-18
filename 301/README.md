# Youtube 节目：
## 显存仅8G！NVIDIA开源Nemotron-0.6B：50分钟音频2分钟转完？抗噪强、速度快，英语转写新神器！
## https://youtu.be/oCmuNMfzzmo

# 安装指南
## 克隆项目
mkdir nemo  
cd nemo  

## 创建运行环境
conda create -n nemotron python=3.10 -y  
conda activate nemotron  

## 安装依赖组件
apt-get update && apt-get install -y libsndfile1 ffmpeg  
pip install Cython packaging  
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]  
pip install gradio   

## 模型下载
hf download nvidia/nemotron-speech-streaming-en-0.6b --local-dir checkpoints/nemotron-speech-streaming-en-0.6b  

## 推理演示
python app.py    

  












 
















