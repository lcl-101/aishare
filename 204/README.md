# Youtube 节目：
## 提示词秒换视频背景！Lumen本地部署指南
## https://youtu.be/Riro5noM6nQ

# 安装指南
## 创建运行环境
conda create -n lumen python=3.10 -y  
conda activate lumen  

## 克隆文件
git clone https://github.com/Kunbyte-AI/Lumen.git  
cd Lumen  

## 安装依赖组件
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install kornia  

## 下载模型
huggingface-cli download Kunbyte/Lumen --local-dir ckpt/Lumen  
huggingface-cli download alibaba-pai/Wan2.1-Fun-1.3B-Control --local-dir ckpt/Wan2.1-Fun-1.3B-Control  
huggingface-cli download alibaba-pai/Wan2.1-Fun-14B-Control --local-dir ckpt/Wan2.1-Fun-14B-Control  
huggingface-cli download briaai/RMBG-2.0 --local-dir ckpt/RMBG-2.0  

cp -r /export/demo-softice/models/ckpt/ .  

## 推理
python app_lumen.py  

  












 
















