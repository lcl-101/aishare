# Youtube 节目：
## AI绘画终极控图！字节跳动USO，风格、人物精准复刻！
## https://youtu.be/4oPJFTL1N0M

# 安装指南
## 创建运行环境
conda create -n uso python=3.10 -y  
conda activate uso  

## 克隆文件
git clone https://github.com/bytedance/USO.git uso  
cd uso  

## 安装依赖组件
pip install -r requirements.txt  

## 下载模型
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/FLUX.1-dev   
huggingface-cli download bytedance-research/USO --local-dir checkpoints/USO  
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir checkpoints/siglip-so400m-patch14-384  
huggingface-cli download openai/clip-vit-large-patch14 --local-dir checkpoints/clip-vit-large-patch14  
huggingface-cli download xlabs-ai/xflux_text_encoders --local-dir checkpoints/xflux_text_encoders   

## 推理
python app.py  


  












 
















