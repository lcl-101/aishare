# Youtube 节目：
## AI拆解万物！PartCrafter一键将图片转为可编辑的3D部件模型，附独家WebUI本地部署教程
## https://youtu.be/A9lWgstDURA

# 安装指南
## 创建运行环境
conda create -n partcrafter python=3.11.13 -y  
conda activate partcrafter  

## 克隆文件
git clone https://github.com/wgsxm/PartCrafter.git  
cd PartCrafter  


## 安装依赖组件
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124  
bash settings/setup.sh  
conda install -c conda-forge libegl libglu pyopengl -y  
pip install gradio  

## 准备模型文件
huggingface-cli download wgsxm/PartCrafter   --local-dir  pretrained_weights/PartCrafter  
huggingface-cli download briaai/RMBG-1.4  --local-dir pretrained_weights/RMBG-1.4  

cp /export/demo-softice/models/pretrained_weights pretrained_weights -r  

## 推理
python app.py  




  












 
















