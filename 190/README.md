# Youtube 节目：
## AI视频生成新王炸！最新开源Wan2.2-S2V本地部署教程，效果惊艳，附独家WebUI！
## https://youtu.be/pxEW5qYJPoU

# 安装指南
## 创建运行环境
conda create -n wan python=3.10 -y    
conda activate wan   

## 克隆文件
git clone https://github.com/Wan-Video/Wan2.2 wan   
cd wan   

## 安装依赖组件
pip install -r requirements.txt    
pip install flash_attn   
pip install decord librosa gradio   

## 准备模型文件
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir checkpoints/Wan2.2-S2V-14B   

## 推理
python app.py   



  












 
















