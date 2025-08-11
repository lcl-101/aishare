# Youtube 节目：
## AI生成的视频是“默片”？索尼官方神器 MMAudio 来了！一句话让你的视频拥有电影级音效。
## https://youtu.be/OPP727tMAc0

# 安装指南
## 创建运行环境
conda create -n MMAudio python=3.10 -y  
conda activate MMAudio  

## 克隆文件
git clone https://github.com/hkchengrex/MMAudio.git  
cd MMAudio  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -e .  

## 准备模型文件
huggingface-cli download hkchengrex/MMAudio --local-dir checkpoints/MMAudio  

## 推理
python gradio_demo.py  





  












 
















