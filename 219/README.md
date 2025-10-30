# Youtube 节目：
## 从零到一：本地部署 Stable Video Infinity，一键生成无限循环、稳定流畅的AI视频！
## https://youtu.be/WYmwRWiehNE

# 安装指南
## 创建运行环境
conda create -n svi python=3.10 -y  
conda activate svi   

## 克隆项目
git clone https://github.com/vita-epfl/Stable-Video-Infinity.git  
cd Stable-Video-Infinity  

## 安装依赖组件
pip install -e .  
pip install flash_attn==2.8.0.post2  
conda install -c conda-forge ffmpeg -y  
conda install -c conda-forge librosa -y  
conda install -c conda-forge libiconv -y     

## 模型下载
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P  
huggingface-cli download vita-video-gen/svi-model --local-dir ./weights/Stable-Video-Infinity --include "version-1.0/*"  
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base  
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base  
huggingface-cli download MeiGen-AI/MeiGen-MultiTalk --local-dir ./weights/MeiGen-MultiTalk  
ln -s $PWD/weights/MeiGen-MultiTalk/multitalk.safetensors weights/Wan2.1-I2V-14B-480P/  
huggingface-cli download ZheWang123/UniAnimate-DiT --local-dir ./weights/UniAnimate-DiT  

## 推理演示
python app.py     

  












 
















