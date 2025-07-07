# Youtube 节目：
## 表格识别终结者！开源OCR神器OCRFlux，完美解决PDF跨页表格提取难题【本地部署教程】
## https://youtu.be/124us7J39Bo

# 安装指南
## 安装系统组件
sudo apt-get update  
sudo apt-get install poppler-utils poppler-data ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools  

## 创建和激活运行环境
conda create -n ocrflux python=3.11 -y  
conda activate ocrflux  

## 克隆项目
git clone https://github.com/chatdoc-com/OCRFlux.git ocrflux  
cd ocrflux  

## 安装依赖
pip install -e . --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/  

## 下载模型
huggingface-cli download ChatDOC/OCRFlux-3B --local-dir checkpoints/OCRFlux-3B  

cp -r /export/demo-softice/models/checkpoints/ .  

## 推理
python app.py  




  












 
















