# Youtube 节目：
## 效果炸裂！阿里开源 Vivid-VR 实测，低清视频秒变高清大片！独家 WebUI 让你一键上手！
## https://youtu.be/qpn8yBCCQr4

# 安装指南
## 创建运行环境
conda create -n vivid python=3.10 -y  
conda activate vivid  

## 克隆文件
git clone https://github.com/csbhr/Vivid-VR.git  
cd Vivid-VR  

## 安装依赖组件
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt    


pip install "numpy<2.0,>=1.17" "opencv-python"  
pip install einops sentencepiece gradio  

## 准备模型文件
huggingface-cli download zai-org/CogVideoX1.5-5B --local-dir ckpts/CogVideoX1.5-5B  
huggingface-cli download zai-org/cogvlm2-llama3-caption --local-dir ckpts/cogvlm2-llama3-caption  
huggingface-cli download csbhr/Vivid-VR --local-dir ckpts/Vivid-VR  

cp ./VRDiT/cogvlm2-llama3-caption/modeling_cogvlm.py ./ckpts/cogvlm2-llama3-caption/modeling_cogvlm.py  

## 推理
python app.py  




  












 
















