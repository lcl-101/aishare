# Youtube 节目：
## 吊打Suno？开源音乐生成神器Heart-MuLa来了！本地部署+精准分段控制，免费生成6分钟广播级全长歌曲！
## https://youtu.be/dOVyA1mOPwc

# 安装指南
## 克隆项目
git clone https://github.com/HeartMuLa/heartlib.git  
cd heartlib  

## 创建运行环境
conda create -n heartlib python=3.10 -y  
conda activate heartlib  
 
## 安装依赖组件
pip install -e .  
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install gradio  

## 模型下载
hf download HeartMuLa/HeartMuLaGen --local-dir checkpoints/HeartMuLaGen  
hf download HeartMuLa/HeartCodec-oss --local-dir checkpoints/HeartCodec-oss  
hf download HeartMuLa/HeartMuLa-oss-3B --local-dir checkpoints/HeartMuLa-oss-3B  

## 推理演示
python app.py    

  












 
















