# Youtube 节目：
## 画质换速度？淘宝 VividVR 深度体验：这是目前开源界最慢、却也是最强的视频修复模型！
## https://youtu.be/qlWvjkLErNk

# 安装指南
## 克隆项目
git clone https://github.com/csbhr/Vivid-VR.git    
cd Vivid-VR   

## 创建运行环境
conda create -n vivid-vr python=3.10 -y  
conda activate vivid-vr  

## 安装依赖组件
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install "numpy<2.0,>=1.17" "opencv-python"  
pip install einops sentencepiece gradio easyocr  
pip install 'triton==2.3.0'  
pip install "numpy<2.0"  

## 模型下载
hf download csbhr/Vivid-VR --local-dir checkpoints/Vivid-VR  
hf download zai-org/CogVideoX1.5-5B --local-dir checkpoints/CogVideoX1.5-5B  
hf download zai-org/cogvlm2-llama3-caption --local-dir checkpoints/cogvlm2-llama3-caption  
mkdir easyocr  
cd cd easyocr/  
wget https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip  
wget https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/zh_sim_g2.zip  
wget https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip  
unzip craft_mlt_25k.zip  
unzip english_g2.zip  
unzip zh_sim_g2.zip  
rm *.zip  
cd ..  
mkdir RealESRGAN  
cd RealESRGAN  
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth  
cd ..  
cp ./VRDiT/cogvlm2-llama3-caption/modeling_cogvlm.py ./checkpoints/cogvlm2-llama3-caption/modeling_cogvlm.py  

## 推理演示
python app.py      

  












 
















