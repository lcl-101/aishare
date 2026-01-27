# Youtube 节目：
## Sora都做不到！开源王炸EgoX：任意视频一键转第一人称视角 🔥 本地部署保姆级教程
## https://youtu.be/cB0OdFO16E4

# 安装指南
## 克隆项目
git clone https://github.com/DAVIAN-Robotics/EgoX.git  
cd EgoX  

## 创建运行环境
conda create -n egox python=3.10 -y  
conda activate egox  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install gradio  

## 模型下载
hf download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers --local-dir checkpoints/Wan2.1-I2V-14B-480P-Diffusers  
hf download DAVIAN-Robotics/EgoX --local-dir checkpoints/EgoX  

## 推理演示
python app.py        

  












 
















