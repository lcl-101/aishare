# Youtube 节目：
## 超越Runway/可灵？LongVie2 来了！无限流长视频生成+精准指哪打哪，本地部署保姆级教程！🚀
## https://youtu.be/yQ-TSaupTJ4

# 安装指南
## 克隆项目
git clone https://github.com/Vchitect/LongVie.git  
cd LongVie  

## 创建运行环境
conda create -n longvie python=3.10 -y  
conda activate longvie  

## 安装依赖组件
conda install psutil -y  
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1  
python -m pip install ninja  
pip install -e .  
pip install gradio  

## 模型下载
hf download Vchitect/LongVie2 --local-dir checkpoints/LongVie2  
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir checkpoints/Wan2.1-I2V-14B-480P   

## 推理演示
python app.py        

  












 
















