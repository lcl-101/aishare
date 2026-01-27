# Youtube 节目：
## 3秒复刻任何声音！阿里最强开源语音 AI Fun-CosyVoice 3.0 保姆级本地部署：低延迟流式对话，支持18种方言 🔥
## https://youtu.be/2oLrpnYjO8I

# 安装指南
## 克隆项目
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git  
cd CosyVoice  

## 创建运行环境
conda create -n cosyvoice python=3.10 -y  
conda activate cosyvoice  

## 安装依赖组件
pip install -r requirements.txt  
apt-get install sox libsox-dev -y  

## 模型下载
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir checkpoints/Fun-CosyVoice3-0.5B-2512  
hf download FunAudioLLM/CosyVoice-ttsfrd --local-dir checkpoints/CosyVoice-ttsfrd  
cd checkpoints/CosyVoice-ttsfrd/  
unzip resource.zip -d .  
pip install ttsfrd_dependency-0.1-py3-none-any.whl  
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl  
cd .. && cd ..  

## 推理演示
python app.py        

  












 
















