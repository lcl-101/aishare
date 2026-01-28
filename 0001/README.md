# Youtube 节目： 
## 告别机械音！通义千问 Qwen3-TTS 重磅开源：3秒克隆、一句话“捏”出完美人声！免费可商用保姆级部署教程 
## https://youtu.be/yNS1TCoWc1k
 
# 安装指南 
## 创建运行环境 
conda create -n videomama python=3.10 -y 
conda activate videomama 
 
## 克隆项目 
git clone https://github.com/cvlab-kaist/VideoMaMa.git 
cd VideoMaMa 
 
## 安装依赖组件 
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 
pip install git+https://github.com/facebookresearch/sam2.git 
pip install accelerate diffusers transformers gradio\ 
opencv-python opencv-contrib-python Pillow "numpy>=1.24.0,<2" scipy \ 
einops tqdm safetensors imageio imageio-ffmpeg matplotlib \ 
omegaconf pydantic psutil huggingface_hub[cli] 
pip install flash_attn==2.7.4.post1 
pip install -e . --no-build-isolation 
 
## 模型下载 
mkdir -p checkpoints/sam2 
hf download SammyLim/VideoMaMa --local-dir checkpoints/VideoMaMa 
hf download stabilityai/stable-video-diffusion-img2vid-xt --local-dir checkpoints/stable-video-diffusion-img2vid-xt 
wget -O checkpoints/sam2/sam2.1_hiera_large.pt \ 
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt 
 
## 推理演示 
python app.py 
 