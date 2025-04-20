# Youtube 节目：
## 谛韵 diffrhythm：AI一键生成流行音乐！歌词、风格、乐器全掌控！
## https://youtu.be/mp5WFqH7hxg

# 安装指南

## 安装系统组件
sudo apt update && sudo apt upgrade -y   
sudo apt-get install espeak-ng -y  
sudo apt install build-essential -y  

## 克隆项目
git clone https://github.com/ASLP-lab/DiffRhythm.git  
cd DiffRhythm  

## 创建和激活运行环境
conda create -n diffrhythm python=3.10 -y  
conda activate diffrhythm  

## 安装依赖
pip install -r requirements.txt  
pip install py3langid  


## 推理
python app.py  












