# Youtube 节目：
## 输入歌词，秒变AI歌手！腾讯开源音乐生成器 SongGeneration 保姆级本地部署教程，还带独家WebUI！
## https://youtu.be/UbmkdquTS-c

# 安装指南
## 创建运行环境
conda create -n SongGeneration python=3.10 -y
conda activate SongGeneration

## 克隆项目
git clone https://github.com/tencent-ailab/SongGeneration.git
cd SongGeneration

## 安装组件
pip install -r requirements.txt
pip install -r requirements_nodeps.txt --no-deps
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install gradio
pip install huggingface_hub==0.23.5

## 准备模型文件
huggingface-cli download tencent/SongGeneration --local-dir checkpoints/SongGeneration

## 推理
python app.py



  












 
















