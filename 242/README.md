# Youtube 节目：
## 腾讯混元OCR重磅开源：1B小模型吊打大模型？图片秒变Markdown/LaTeX，本地部署保姆级教程！
## https://youtu.be/GUAPM2xszSc

# 安装指南
## 克隆项目
git clone https://github.com/Tencent-Hunyuan/HunyuanOCR.git  
cd HunyuanOCR  

## 创建运行环境
conda create -n hunyuanocr python=3.10 -y  
conda activate hunyuanocr   

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4  
pip install accelerate gradio  

## 模型下载
hf download tencent/HunyuanOCR --local-dir checkpoints/HunyuanOCR   

## 推理演示
python app.py      

  












 
















