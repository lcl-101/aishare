# Youtube 节目：
## Google王炸！Function Gemma：仅几百兆的“特种部队”，打造你的私人本地 AI Agent！
## https://youtu.be/hl-xOgFQWDU

# 安装指南
## 克隆项目
mkdir functiongemma  
cd functiongemma  

## 创建运行环境
conda create -n functiongemma python=3.10 -y  
conda activate functiongemma  
 
## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install transformers gradio crawl4ai accelerate  
playwright install-deps  

## 模型下载
hf download google/functiongemma-270m-it --local-dir checkpoints/functiongemma-270m-it  

## 推理演示
python app.py    

  












 
















