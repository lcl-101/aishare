# Youtube 节目： 
## 告别机械音！通义千问 Qwen3-TTS 重磅开源：3秒克隆、一句话“捏”出完美人声！免费可商用保姆级部署教程 
## https://youtu.be/yNS1TCoWc1k
 
# 安装指南 
## 创建运行环境 
conda create -n qwen3-tts python=3.10 -y   
conda activate qwen3-tts   
 
## 克隆项目 
git clone https://github.com/QwenLM/Qwen3-TTS.git   
cd Qwen3-TTS   
 
## 安装依赖组件 
pip install -e .   
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124   
pip install flash_attn==2.7.4.post1 --no-build-isolation   
 
## 模型下载 
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir checkpoints/Qwen3-TTS-12Hz-1.7B-CustomVoice   
hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir checkpoints/Qwen3-TTS-12Hz-1.7B-VoiceDesign   
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir checkpoints/Qwen3-TTS-12Hz-1.7B-Base   
hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir checkpoints/Qwen/Qwen3-TTS-Tokenizer-12Hz   
   
## 推理演示 
python app.py   
 