# Youtube 节目：
## 别再浪费时间了！OpenAudio 开源TTS项目深度评测：从部署到放弃的全过程
## https://youtu.be/fmeAGKNZ99o

# 安装指南
## 创建运行环境
conda create -n openaudio python=3.10 -y
conda activate openaudio

conda install -c conda-forge portaudio sox ffmpeg -y

## 克隆文件
git clone https://github.com/fishaudio/fish-speech.git openaudio
cd openaudio

## 安装依赖组件
pip install -e .

## 下载模型
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

## 推理
python app.py





  












 
















