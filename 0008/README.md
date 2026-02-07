# Youtube 节目： 
##  告别PDF乱码！智谱GLM-OCR开源神器：复杂公式表格一键转Markdown，RAG数据清洗救星！
##  https://youtu.be/-G6oTAk_gRk
 
# 安装指南 
## 创建运行环境 
conda create -n glmocr python=3.10 -y 
conda activate glmocr 
 
## 克隆项目 
mkdir glmocr 
cd glmocr 
 
## 安装依赖组件 
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 
pip install git+https://github.com/huggingface/transformers 
pip install pillow pdf2image gradio accelerate 
 
## 模型下载 
hf download zai-org/GLM-OCR --local-dir checkpoints/GLM-OCR 
 
## 推理演示 
python app.py 
 