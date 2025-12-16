# Youtube 节目：
## 语音识别六边形战士！FunASR本地部署实战：支持粤语/客家话，抗噪能力Max，比Whisper更懂中文！
## https://youtu.be/rePYDmgcf9I

# 安装指南
## 克隆项目
git clone https://github.com/FunAudioLLM/Fun-ASR.git  
cd Fun-ASR  

## 创建运行环境
conda create -n fun-asr python=3.10 -y  
conda activate fun-asr  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  

## 模型下载
hf download FunAudioLLM/Fun-ASR-Nano-2512 --local-dir checkpoints/Fun-ASR-Nano-2512  

## 推理演示
python app.py        

  












 
















