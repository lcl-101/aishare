# Youtube 节目：
## 直接对比型： 新王登基？黑森林最新AI绘画模型 Krea，与 Flux 同台竞技，效果高下立判！(附两种本地部署对比方案)
## https://youtu.be/oRRf_vrVaes

# 安装指南
## 创建运行环境
conda create -n flux python=3.10 -y  
conda activate flux  

## 创建目录
mkdir flux  
cd flux  

## 安装依赖
pip install -r requirements.txt  

## 下载模型
huggingface-cli download black-forest-labs/FLUX.1-Krea-dev  --local-dir checkpoints/FLUX.1-Krea-dev  
huggingface-cli download black-forest-labs/FLUX.1-dev  --local-dir checkpoints/FLUX.1-dev  

mkdir checkpoints  
cp -r /export/demo-softice/models/black-forest-labs/FLUX.1-dev checkpoints/FLUX.1-dev  
cp -r /export/demo-softice/models/black-forest-labs/FLUX.1-Krea-dev checkpoints/FLUX.1-Krea-dev  

## 推理
python app.py  



  












 
















