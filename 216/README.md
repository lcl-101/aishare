# Youtube 节目：
## 终结手动抄写！最强开源OCR神器 DeepSeek-OCR 本地部署保姆级教程 (附独家WebUI)
## https://youtu.be/JJbOjBcGjeE

# 安装指南
## 创建运行环境
conda create -n deepseek-ocr python=3.12.9 -y  
conda activate deepseek-ocr   

## 克隆项目
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git  
cd DeepSeek-OCR  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118  
pip install https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu121-cp38-abi3-manylinux1_x86_64.whl  
pip install -r requirements.txt  
pip install flash-attn==2.7.3 --no-build-isolation  
pip install gradio    

## 模型下载
hf download deepseek-ai/DeepSeek-OCR --local-dir checkpoints/DeepSeek-OCR  
  
## 推理演示
python app.py     

  












 
















