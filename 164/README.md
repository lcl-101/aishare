# Youtube 节目：
## AI音乐创作神器 Jamify：保姆级教程，10GB显存本地跑！支持参考曲风、自定义歌词、提示词，精准生成你的专属歌曲（附一键启动WebUI）
## https://youtu.be/HmbwufCpkXw

# 安装指南
## 克隆文件
git clone https://github.com/declare-lab/jamify  
cd jamify  

## 创建运行环境
conda create -n jamify python=3.10 -y  
conda activate jamify  

## 安装依赖组件
bash install.sh  
pip install gradio  
pip install pyloudnorm  

## 准备模型文件
huggingface-cli download declare-lab/jam-0.5 --local-dir checkpoints/jam-0.5  
huggingface-cli download OpenMuQ/MuQ-MuLan-large --local-dir checkpoints/MuQ-MuLan-large  

## 推理
python app.py  




  












 
















