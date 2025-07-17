# Youtube 节目：
## MonkeyOCR：新一代轻量级文档解析神器！3B模型挑战复杂文档 (在线体验+本地部署)
## https://youtu.be/OLz-mdTChhs

# 安装指南
## 安装系统组件
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y  
sudo apt-get install poppler-utils -y  

## 克隆项目
git clone https://github.com/Yuliang-Liu/MonkeyOCR.git  
cd MonkeyOCR  

## 创建和激活运行环境
conda create -n MonkeyOCR python=3.10 -y  
conda activate MonkeyOCR  

## 安装依赖组件
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124   
pip install -e .  
pip install gradio==5.23.3  
pip install pdf2image==1.17.0  

## 下载模型文件
python tools/download_model.py  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python tools/lmdeploy_patcher.py patch  
python demo/demo_gradio.py  

请帮我理出审批流程  



  












 
















