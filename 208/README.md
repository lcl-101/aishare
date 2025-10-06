# Youtube 节目：
## 让任何图片跳舞！阿里开源神器 WAN Animate，保姆级本地部署教程（含独家WebUI）
## https://youtu.be/-jcW4nkmyP0

# 安装指南
## 创建运行环境
conda create -n wananimation python=3.10 -y
conda activate wananimation

## 克隆项目
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2

## 安装依赖组件
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -r requirements_s2v.txt
pip install -r requirements_animate.txt
pip install moviepy gradio

## 下载模型
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B


## 准备文件
python app.py
 

  












 
















