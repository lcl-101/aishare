# Youtube 节目：
## 继图片转吉卜力风之后，AI现在能用文字生成吉卜力世界了！
## https://youtu.be/KVANfnUsOjo

# 安装指南

## 安装系统组件
sudo apt update && sudo apt upgrade -y  
sudo apt install build-essential -y    
   
## 创建目录
mkdir Ghibli  
cd Ghibli  

## 创建和激活运行环境
conda create -n Ghibli python=3.10 -y  
conda activate Ghibli  

## 安装依赖
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  

## 下载模型
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/FLUX.1-dev   
huggingface-cli download nitrosocke/Ghibli-Diffusion --local-dir checkpoints/Ghibli-Diffusion  
huggingface-cli download openfree/flux-chatgpt-ghibli-lora --local-dir checkpoints/flux-chatgpt-ghibli-lora  

## 推理
find . -name "*:Zone.Identifier" -type f -delete    
python app.py  



 
















