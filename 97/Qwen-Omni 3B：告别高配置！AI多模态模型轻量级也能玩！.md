# Youtube 节目：
## Qwen-Omni 3B：告别高配置！AI多模态模型轻量级也能玩！  
## https://youtu.be/RDTI5RWuod0   

# 安装指南

## 安装编译工具
sudo apt install build-essential -y  

## 创建和激活运行环境
conda create -n Omni python=3.10 -y    
conda activate Omni  

## 克隆项目
git clone https://github.com/QwenLM/Qwen2.5-Omni.git  
cd Qwen2.5-Omni  

## 安装依赖组件
find . -name "*:Zone.Identifier" -type f -delete  
pip install -r requirements_web_demo.txt  

## 下载模型文件
huggingface-cli download Qwen/Qwen2.5-Omni-7B--local-dir checkpoints/Qwen2.5-Omni-7B  
huggingface-cli download Qwen/Qwen2.5-Omni-3B--local-dir checkpoints/Qwen2.5-Omni-3B  

## 启动程序
python web_demo.py --flash-attn2  









 
















