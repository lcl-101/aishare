# Youtube 节目：
## 美团开源Infinite-Talk：SOTA级口型同步，支持双人对话！保姆级本地部署教程，实测显存高达46GB！
## https://youtu.be/rw0mDzAcaFo

# 安装指南
## 创建运行环境
conda create -n InfiniteTalk python=3.10 -y  
conda activate InfiniteTalk   

## 克隆文件
git clone https://github.com/MeiGen-AI/InfiniteTalk.git    
cd InfiniteTalk   

## 安装依赖组件
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124   
pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124    


pip install misaki[en]   
pip install ninja   
pip install psutil   
pip install packaging    
pip install wheel   
pip install flash_attn==2.7.4.post1   

pip install -r requirements.txt   
conda install -c conda-forge librosa    

## 准备模型文件
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir weights/Wan2.1-I2V-14B-480P   
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir weights/chinese-wav2vec2-base    
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir weights/chinese-wav2vec2-base   
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir weights/InfiniteTalk   

## 推理
python app.py   


  












 
















