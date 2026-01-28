# Youtube 节目：
## 只有7B参数却能看懂《小丑》？腾讯开源 Arc-Hunyuan-Video-7B：视频理解AI的终极形态！
## https://youtu.be/JLc-QyoTuJ0

# 安装指南
## 创建项目目录
git clone https://github.com/TencentARC/ARC-Hunyuan-Video-7B.git  
cd ARC-Hunyuan-Video-7B  

## 创建运行环境
conda create -n arc-hunyuan-video-7b python=3.10 -y  
conda activate arc-hunyuan-video-7b   

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install git+https://github.com/liyz15/transformers.git@arc_hunyuan_video  
pip install flash-attn==2.7.4.post1 --no-build-isolation  
pip install --upgrade gradio  

## 模型下载
hf download TencentARC/ARC-Hunyuan-Video-7B --local-dir checkpoints/ARC-Hunyuan-Video-7B   

## 推理演示
python app.py      

  












 
















