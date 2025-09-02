# Youtube 节目：
## 腾讯开源数字人项目 Sonic 保姆级教程，14GB显存本地部署，口型精准，中英文通吃！
## https://youtu.be/oVkp9rFfZNU

# 安装指南
## 创建运行环境
conda create -n sonic python=3.10 -y  
conda activate sonic  

## 克隆文件
git clone https://github.com/jixiaozhong/Sonic.git  
cd Sonic  

## 安装依赖组件
pip install -r requirements.txt  

## 下载模型
huggingface-cli download LeonJoe13/Sonic --local-dir  checkpoints  
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir  checkpoints/stable-video-diffusion-img2vid-xt  
huggingface-cli download openai/whisper-tiny --local-dir checkpoints/whisper-tiny  

## 推理
python gradio_app.py  


  












 
















