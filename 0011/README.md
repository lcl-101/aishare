# Youtube 节目： 
##  免费商用！媲美Suno的开源音乐生成神器 ACE-Step-1.5，4GB显存流畅运行，保姆级本地部署教程！
##  https://youtu.be/U34tBgO8XSw
 
# 安装指南 
## 创建运行环境 
conda create -n ace-step-1-5 python=3.10 -y 
conda activate ace-step-1-5 
 
## 克隆项目 
git clone https://github.com/ACE-Step/ACE-Step-1.5.git 
cd ACE-Step-1.5 
 
## 安装依赖组件 
pip install -r requirements-inference-ubuntu.txt 
 
## 模型下载 
hf download ACE-Step/Ace-Step1.5 --local-dir checkpoints/Ace-Step1.5 
 
## 推理演示 
python app.py 
 