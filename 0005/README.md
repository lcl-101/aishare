# Youtube 节目： 
##  碾压商业软件？阿里 Qwen3-ASR 开源：精准过滤背景音、支持热词提醒、秒出 SRT 字幕！
##  https://youtu.be/o8F5NVdaKvA
 
# 安装指南 
## 创建运行环境 
conda create -n qwen3-asr python=3.10 -y 
conda activate qwen3-asr 
 
## 克隆项目 
git clone https://github.com/QwenLM/Qwen3-ASR.git 
cd Qwen3-ASR 
 
## 安装依赖组件 
pip install -e . 
 
## 模型下载 
hf download Qwen/Qwen3-ASR-1.7B --local-dir checkpoints/Qwen3-ASR-1.7B 
hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir checkpoints/Qwen3-ForcedAligner-0.6B 
 
## 推理演示 
python app.py 
 