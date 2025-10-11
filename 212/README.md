# Youtube 节目：
## 克隆任何人的声音和情绪！B站开源神器 IndexTTS2 保姆级本地部署教程
## https://youtu.be/oLnQ2epDwD8

# 安装指南
## 创建运行环境
apt update && apt install git-lfs  
git lfs install  
conda create -n indextts python=3.10 -y  
conda activate indextts    

## 克隆项目
git clone https://github.com/index-tts/index-tts.git  
cd index-tts  
git lfs pull    

## 安装依赖组件
pip install -e .  
pip install gradio  

## 下载模型
hf download IndexTeam/IndexTTS-2 --local-dir checkpoints  
  
## 准备文件
python webui.py   
 

  












 
















