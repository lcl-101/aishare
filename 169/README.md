# Youtube 节目：
## 告别付费OCR！ 最强开源神器 OLMOCR 本地部署保姆级教程，精准识别PDF与复杂表格！
## https://youtu.be/3k9bSHFgThw

# 安装指南
## 创建运行环境
conda create -n olmocr python=3.11 -y  
conda activate olmocr  

## 克隆项目
apt-get install poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools  

git clone https://github.com/allenai/olmocr.git  
cd olmocr  

## 安装依赖
pip install olmocr[bench]  
pip install olmocr[gpu]  --extra-index-url https://download.pytorch.org/whl/cu128  
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl  
pip install gradio  

## 下载模型
huggingface-cli download allenai/olmOCR-7B-0725  --local-dir checkpoints/olmOCR-7B-0725   

## 推理
python app.py  



  












 
















