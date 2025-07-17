# Youtube 节目：
## 开源OCR新标杆？Nanonets-OCR实测：复杂表格、公式、复选框识别超乎想象！(含本地部署) 
## https://youtu.be/gteY2sNbfIU

# 安装指南
## 准备编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  

## 创建运行环境
conda create -n nanonets python=3.10.0 -y  
conda activate nanonets  

## 克隆项目
git clone https://huggingface.co/spaces/Souvik3333/Nanonets-ocr-s nanonets  
cd nanonets  

## 安装依赖组件
pip install -r requirements.txt  
http://cache.mixazure.com:88/whl/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl  

## 准备模型文件
huggingface-cli download nanonets/Nanonets-OCR-s --local-dir checkpoints/Nanonets-OCR-s  

## 推理 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6    
find . -name "*:Zone.Identifier" -type f -delete    
python app.py  



  












 
















