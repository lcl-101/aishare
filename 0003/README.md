# Youtube 节目： 
##  🔥 DeepSeek-OCR2 本地部署！1B小模型吊打闭源巨头？告别乱码，重塑文档识别天花板
##  https://youtu.be/wh8773oQ5U0
 
# 安装指南 
## 创建运行环境 
conda create -n deepseek-ocr2 python=3.10 -y  
conda activate deepseek-ocr2  
 
## 克隆项目 
mkdir deepseekocr2  
cd deepseekocr2  
 
## 安装依赖组件 
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124   
pip install flash_attn==2.7.4.post1   
pip install transformers==4.46.3 tokenizers==0.20.3 einops addict easydict gradio pdf2image PyMuPDF matplotlib requests   
 
## 模型下载 
hf download deepseek-ai/DeepSeek-OCR-2 --local-dir checkpoints/DeepSeek-OCR-2   
 
## 推理演示 
python app.py  
 