# Youtube 节目：
## AI生成的视频没声音？腾讯混元Foley一键拯救！保姆级本地部署教程+独家优化版WebUI
## https://youtu.be/eCPU5wicn6U

# 安装指南
## 创建运行环境
conda create -n foley python=3.10 -y  
conda activate foley  

## 克隆文件
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley  
cd HunyuanVideo-Foley  
cd examples  
for i in {1..8}; do  
  wget "https://media.githubusercontent.com/media/Tencent-Hunyuan/HunyuanVideo-Foley/refs/heads/main/examples/${i}_video.mp4?download=true" -O "${i}_video.mp4"  
done  

cd ..  

## 安装依赖组件
pip install -r requirements.txt  

## 准备模型文件
huggingface-cli download tencent/HunyuanVideo-Foley --local-dir checkpoints/HunyuanVideo-Foley  

cp -r /export/demo-softice/models/checkpoints/ checkpoints  

## 推理
python app.py  


  












 
















