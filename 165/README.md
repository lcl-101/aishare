# Youtube 节目：
## 【AI视频理解】一键总结&问答！给视频理解大模型装上WebUI，本地部署 ARC-Hunyuan-Video 保姆级教程
## https://youtu.be/tLN4lbNUB80

# 安装指南
## 克隆项目
git clone https://github.com/TencentARC/ARC-Hunyuan-Video-7B.git HunyuanVideo  
cd HunyuanVideo  
 
## 创建运行环境
conda create -n HunyuanVideo python=3.10 -y  
conda activate HunyuanVideo  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install git+https://github.com/liyz15/transformers.git@arc_hunyuan_video  
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  
pip install httpx==0.27.0  
pip install --upgrade gradio  

## 准备模型文件
huggingface-cli download TencentARC/ARC-Hunyuan-Video-7B --local-dir checkpoints/ARC-Hunyuan-Video-7B  

## 推理
python app.py  




  












 
















