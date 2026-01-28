# Youtube 节目：
## 唯快不破！VibeVoice-realtime-0.5B 上手实测：五步搞定本地流式语音交互
## https://youtu.be/48hn-_H6w_Y

# 安装指南
## 克隆项目
git clone https://github.com/microsoft/VibeVoice.git  
cd VibeVoice  

## 创建运行环境
conda create -n vibevoice python=3.10 -y  
conda activate vibevoice   

## 安装依赖组件
pip install -e . 

## 模型下载
hf download microsoft/VibeVoice-Realtime-0.5B --local-dir checkpoints/VibeVoice-Realtime-0.5B  
hf download Qwen/Qwen2.5-0.5B --local-dir checkpoints/Qwen2.5-0.5B   

## 推理演示
python app.py       

  












 
















