# Youtube 节目：
## 告别单口相声！Dia：开源AI语音合成，实现多人对话！
## https://youtu.be/inhZDohOJFI

# 安装指南

## 准备系统工具
sudo apt update && sudo apt upgrade -y  
sudo apt install ffmpeg -y  

## 克隆项目
git clone https://github.com/nari-labs/dia.git  
cd dia  

## 创建和激活运行环境
conda create -n Dia python=3.10 -y    
conda activate Dia  

## 安装依赖组件
pip install -e .  

## 下载模型
huggingface-cli download nari-labs/Dia-1.6B --local-dir checkpoints/Dia-1.6B  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete    
python app.py  










 
















