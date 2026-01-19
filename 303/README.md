# Youtube 节目：
## 吊打商业API？Google王牌翻译模型 Translate Gemma 27B 本地部署全攻略！
## https://youtu.be/sTyunhyMsKo

# 安装指南
## 克隆项目
mkdir translategemma  
cd translategemma  

## 创建运行环境
conda create -n translategemma python=3.10 -y  
conda activate translategemma  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install gradio transformers accelerate  

## 模型下载
hf download google/translategemma-27b-it --local-dir checkpoints/translategemma-27b-it  

## 推理演示
python app.py    

  












 
















