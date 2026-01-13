# Youtube 节目：
## 视频生成界的“终结者”？快手可灵开源 UniVideo：生成、编辑、理解三合一！保姆级本地部署教程 🚀
## https://youtu.be/Qivo6LkmXeU

# 安装指南
## 克隆项目
git clone https://github.com/KlingTeam/UniVideo.git  
cd UniVideo  

## 创建运行环境
conda create -n univideo python=3.10 -y  
conda activate univideo  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1  
pip install -r requirements.txt  
pip install gradio  

## 模型下载
hf download KlingTeam/UniVideo --local-dir checkpoints/UniVideo  
hf download hunyuanvideo-community/HunyuanVideo --local-dir checkpoints/HunyuanVideo  
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir checkpoints/Qwen2.5-VL-7B-Instruct   

## 推理演示
python app.py    

  












 
















