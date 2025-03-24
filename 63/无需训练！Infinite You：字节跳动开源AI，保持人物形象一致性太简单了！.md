# Youtube 节目：
## 无需训练！Infinite You：字节跳动开源AI，保持人物形象一致性太简单了！
## https://youtu.be/et0CBGYNgzo

# 安装指南

## 安装编译工具
sudo apt update && sudo apt upgrade  
sudo apt install build-essential -y  

## 创建运行环境
conda create -n InfiniteYou python=3.10 -y  
conda activate InfiniteYou  

## 克隆项目
git clone https://github.com/bytedance/InfiniteYou.git  
cd InfiniteYou  

## 安装依赖
pip install -r requirements.txt  

## 下载模型文件
huggingface-cli download ByteDance/InfiniteYou --local-dir models/InfiniteYou  
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir models/FLUX.1-dev   
find . -name "*:Zone.Identifier" -type f -delete  

## 启动程序
## 修改 app.py 230 行，缓存设置为 False
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/anaconda3/envs/InfiniteYou/lib/  
python app.py  











