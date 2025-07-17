# Youtube 节目：
## Qwen-Omni 3B：告别高配置！AI多模态模型轻量级也能玩！  
## https://youtu.be/2P6oeqAzVs0 

# 安装指南

## 安装编译工具
sudo apt install build-essential -y  

## 创建和激活运行环境
conda create -n drape1 python=3.10 -y    
conda activate drape1  

## 克隆项目
git clone https://github.com/uwear-ai/drape1.git  
cd drape1  

## 安装依赖组件
pip install -e .  
pip install gradio  

## 下载模型文件
huggingface-cli download Uwear-ai/Drape1 --local-dir checkpoints/Drape1  
huggingface-cli download madebyollin/sdxl-vae-fp16-fix --local-dir checkpoints/sdxl-vae-fp16-fix   

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete    
python demo.py  









 
















