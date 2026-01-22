# Youtube 节目： 
##  PDF提取噩梦终结者！这款1B参数开源OCR神器，4G显存完美还原复杂表格，RAG数据清洗必备！
##  https://youtu.be/vfGhB0eF344
 
# 安装指南 
## 创建运行环境 
conda create -n LightOnOCR2 python=3.10 -y 
conda activate LightOnOCR2 
 
## 克隆项目 
mkdir lightonocr2 
cd lightonocr2 
 
## 安装依赖组件 
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 
pip install git+https://github.com/huggingface/transformers 
pip install pillow pypdfium2 gradio 
 
## 模型下载 
hf download lightonai/LightOnOCR-2-1B --local-dir checkpoints/LightOnOCR-2-1B 
 
## 推理演示 
python app.py 
 