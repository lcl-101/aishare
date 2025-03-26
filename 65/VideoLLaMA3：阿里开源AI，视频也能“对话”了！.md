# Youtube 节目：
## VideoLLaMA3：阿里开源AI，视频也能“对话”了！
## https://youtu.be/G9ehYkEGvK4

# 安装指南

## 安装编译工具
sudo apt update && sudo apt upgrade -y   
sudo apt install build-essential -y   
sudo apt install ffmpeg -y   

## 安装 CUDA 组件
wget http://cache.mixazure.com/cuda/cuda_12.4.1_550.54.15_linux.run    
sudo sh cuda_12.4.1_550.54.15_linux.run    

## 添加环境变量
rm cuda_12.4.1_550.54.15_linux.run    
sudo nano ~/.bashrc    

export PATH=/usr/local/cuda-12.4/bin:$PATH    
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH    

source ~/.bashrc    

## 克隆项目
git clone https://github.com/DAMO-NLP-SG/VideoLLaMA3.git  
cd VideoLLaMA3  

## 创建运行环境
conda create -n VideoLLaMA3 python==3.10 -y  
conda activate VideoLLaMA3  

## 安装依赖组件
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install flash-attn --no-build-isolation  
pip install transformers==4.46.3 accelerate==1.0.1  
pip install decord ffmpeg-python imageio opencv-python  
pip install gradio  

## 下载模型
huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-7B --local-dir checkpoints/VideoLLaMA3-7B  
huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-2B --local-dir checkpoints/VideoLLaMA3-2B  
huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-7B-Image --local-dir checkpoints/VideoLLaMA3-7B-Image  
huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-2B-Image --local-dir checkpoints/VideoLLaMA3-2B-Image   
huggingface-cli download DAMO-NLP-SG/VL3-SigLIP-NaViT --local-dir checkpoints/VL3-SigLIP-NaViT  

## 推理
find . -name "*:Zone.Identifier" -type f -delete    
python app.py  












