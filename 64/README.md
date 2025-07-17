# Youtube 节目：
## 【吐槽】Cube3D：AI文字生成3D模型，好看是好看，就是有点“素”！
## https://youtu.be/r8l09jFUwls

# 安装指南

## 安装 lfs 组件
sudo apt update && sudo apt upgrade  
sudo apt-get install git-lfs 
git lfs install 

## 创建运行环境
conda create -n cube python=3.10 -y 
conda activate cube 

## 克隆项目
git clone https://huggingface.co/spaces/Roblox/cube3d-interactive 
cd cube3d-interactive 

## 安装依赖
pip install -r requirements.txt 

## 下载模型
huggingface-cli download Roblox/cube3d-v0.1 --local-dir ./model_weights
huggingface-cli download openai/clip-vit-large-patch14 config.json  model.safetensors tokenizer_config.json vocab.json  merges.txt tokenizer.json special_tokens_map.json --local-dir openai/clip-vit-large-patch14 

## 推理
find . -name "*:Zone.Identifier" -type f -delete 
python app.py 











