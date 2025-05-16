# Youtube 节目：
## AI 智能调光太神了！LBM 一键让主体完美融入背景光影 (在线本地体验)
## https://youtu.be/g4oZJcEx4tE

# 安装指南

## 安装系统组件
sudo apt update && sudo apt upgrade -y         
sudo apt install git-lfs -y      
git lfs install       

## 克隆项目
git clone https://huggingface.co/spaces/jasperai/LBM_relighting  
cd LBM_relighting  

## 创建和激活运行环境
conda create -n LBM python=3.10 -y  
conda activate LBM  

## 安装依赖组件
## 修改 torch==2.7.0  
pip install -r requirements.txt  

## 下载模型
pip install "huggingface_hub[cli]"
huggingface-cli download jasperai/LBM_relighting --local-dir checkpoints/LBM_relighting  
huggingface-cli download ZhengPeng7/BiRefNet --local-dir checkpoints/BiRefNet  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python app.py  












 
















