# Youtube 节目： 
##  MOVA 上手指南：“能听又能看”的开源视频大模型，音画同步 100% 免费可商用！
##  https://youtu.be/UW9PI0PLNbs
 
# 安装指南 
## 创建运行环境 
conda create -n mova python=3.13 -y 
conda activate mova 
 
## 克隆项目 
git clone https://github.com/OpenMOSS/MOVA.git 
cd MOVA 
 
## 安装依赖组件 
pip install -e . 
pip install gradio bitsandbytes 
pip install flash_attn --no-build-isolation 
 
## 模型下载 
hf download OpenMOSS-Team/MOVA-720p --local-dir checkpoints/MOVA-720p 
 
## 推理演示 
python app.py 
 