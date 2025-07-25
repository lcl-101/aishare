# Youtube 节目：
## 秒杀配音演员？开源AI语音模型Higgs-Audio，支持情感合成、声音克隆与多人对话！保姆级本地部署教程 (附WebUI)
## https://youtu.be/6_pEEMGSYF4

# 安装指南
## 克隆文件
git clone https://github.com/boson-ai/higgs-audio.git higgsaudio  
cd higgsaudio  

## 安装依赖组件
pip install -r requirements.txt  
pip install -e .  
pip install gradio  

## 准备模型文件
huggingface-cli download bosonai/higgs-audio-v2-generation-3B-base --local-dir checkpoints/higgs-audio-v2-generation-3B-base  

huggingface-cli download bosonai/higgs-audio-v2-tokenizer --local-dir checkpoints/higgs-audio-v2-tokenizer  

cp /export/demo-softice/models/checkpoints/ checkpoints/ -r  

## 推理
python app.py  


请将这个视频分割为场景，为每个场景提供开始时间、结束时间和详细描述。  




  












 
















