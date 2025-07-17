# Youtube 节目：
## AI音频效率革命：Aero-1-Audio，一天训练完成，性能媲美Whisper！  
## https://youtu.be/BghvA_1v1Po    

# 安装指南

## 安装系统组件
sudo apt-get install git-lfs  
git lfs install  

## 创建和激活运行环境
conda create -n Aero python=3.10 -y    
conda activate Aero  

## 克隆项目
git clone https://huggingface.co/spaces/lmms-lab/Aero-1-Audio-Demo  
cd Aero-1-Audio-Demo  

## 安装依赖组件
pip install -r requirements.txt  

## 下载模型文件
huggingface-cli download lmms-lab/Aero-1-Audio-1.5B --local-dir checkpoints/Aero-1-Audio-1.5B  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python app.py   









 
















