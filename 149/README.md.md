# Youtube 节目：
## NVIDIA 开源项目 Addit 深度体验：基于Flux模型，精准为图像添加物体 | 在线与本地部署教程
## https://youtu.be/Uj6RymO-mj8

# 安装指南
## 克隆项目
git clone https://github.com/NVlabs/addit.git  
cd addit  

## 创建运行环境
conda env create -f environment.yml  
conda activate addit  

## 安装依赖组件
pip install gradio  

## 准备模型文件
mkdir -p checkpoints  
cp -r /export/demo-softice/models/black-forest-labs/FLUX.1-dev checkpoints/FLUX.1-dev  

## 推理 
python app.py  




  












 
















