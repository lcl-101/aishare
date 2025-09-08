# Youtube 节目：
## 18GB显存就能跑！字节跳动 Latentsync 保姆级部署教程，轻松生成AI口播视频！
## https://youtu.be/BwUapL77vGc

# 安装指南
## 创建运行环境
conda create -y -n latentsync python=3.10.13    
conda activate latentsync    

## 克隆项目
git clone https://github.com/bytedance/LatentSync.git    
cd LatentSync    

## 安装依赖组件
conda install -c conda-forge ffmpeg -y  
sudo apt -y install libgl1    
pip install -r requirements.txt    

## 下载模型
huggingface-cli download ByteDance/LatentSync-1.6 --local-dir checkpoints/LatentSync-1.6 --exclude "*.git*" "README.md"    

cp -r /export/demo-softice/models/LatentSync-1.6/ checkpoints/  

## 推理 
python gradio_app.py    

  












 
















