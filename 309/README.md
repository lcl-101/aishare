# Youtube 节目： 
##  给AI装上“上帝之耳”？VibeVoice-ASR 本地部署实战：支持60分钟长音频、热词定制与多人识别！
##  https://youtu.be/M33otYXV5kM
 
# 安装指南 
## 创建运行环境 
conda create -n vibevoice python=3.10 -y  
conda activate vibevoice  
 
## 克隆项目 
git clone https://github.com/microsoft/VibeVoice.git  
cd VibeVoice  
 
## 安装依赖组件 
pip install -e .[asr]  
pip install liger-kernel  
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1  
 
## 模型下载 
hf download microsoft/VibeVoice-ASR --local-dir checkpoints/VibeVoice-ASR  
 
## 推理演示 
python app.py 
 