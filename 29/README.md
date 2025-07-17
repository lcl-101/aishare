# Youtube 节目：
## 让视频“隐身术”成真！阿里开源 AI Diffuseraser 体验！
## https://youtu.be/Qx8xOZs39Yg


# 安装指南

## 克隆项目
git clone https://github.com/lixiaowen-xw/DiffuEraser.git
cd DiffuEraser/
## 创建运行环境
conda create -n diffueraser python=3.9.19 -y  
conda activate diffueraser
## 安装依赖
pip install -r requirements.txt 
pip install gradio
conda install -c conda-forge ffmpeg -y
## 推理
python app.py 