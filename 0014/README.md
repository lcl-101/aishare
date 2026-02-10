# Youtube 节目： 
##  视频风格迁移新SOTA？TeleStyle本地部署实测：精准锁住内容，完美复刻画风！
##  https://youtu.be/bC_iDI026NM
 
# 安装指南 
## 创建运行环境 
conda create -n TeleStyle python=3.10 -y 
conda activate TeleStyle 
 
## 克隆项目 
git clone https://github.com/Tele-AI/TeleStyle.git 
cd TeleStyle 
 
## 安装依赖组件 
pip install -r requirements.txt 
 
## 模型下载 
hf download Qwen/Qwen-Image-Edit-2509 --local-dir checkpoints/Qwen-Image-Edit-2509 
hf download Tele-AI/TeleStyle --local-dir checkpoints/TeleStyle 
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir checkpoints/Wan2.1-T2V-1.3B-Diffusers 
 
## 推理演示 
python app.py 
 