# Youtube 节目：
## 一张图、一句话，秒变3D模型！微软开源神器 Trellis 本地部署保姆级教程，轻松玩转文生/图生3D！
## https://youtu.be/JYgLfooLUYw

# 安装指南
## 创建运行环境
conda create -n trellis python=3.10 -y  
conda activate trellis   

## 克隆项目
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git  
cd TRELLIS  

## 安装依赖组件
pip install -r requirements.txt  
pip install git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization  
pip uninstall gradio -y  
pip install gradio  

## 下载模型
hf download microsoft/TRELLIS-image-large --local-dir checkpoints/TRELLIS-image-large  
hf download microsoft/TRELLIS-text-xlarge --local-dir checkpoints/TRELLIS-text-xlarge  

## 准备文件
python app1.py  
 

  












 
















