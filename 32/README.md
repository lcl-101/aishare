# Youtube 节目：
## 照片、视频一键抠图！开源 Ben 体验：免费、高效，还支持绿幕！
## https://youtu.be/3nB9hSEwK3k

# 安装指南
## 创建 python 运行环境
cd /home/softice  
conda create -n BEN python=3.10 -y  
conda activate BEN  
pip install git+https://github.com/PramaLLC/BEN2.git  
pip install opencv-python  
pip install gradio  
sudo apt update  
sudo apt install ffmpeg -y  

##推理
python app.py  