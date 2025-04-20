# Youtube 节目：
## 告别暗淡！Light-A-Video：AI 一键优化视频光线
## https://youtu.be/-d8Qo1zqrTE

# 安装指南

## 创建 python 运行环境
conda create -n LightAVideo python=3.10.0 -y  
conda activate LightAVideo  

## 克隆项目和下载模型
git clone https://github.com/bcmi/Light-A-Video.git  
cd Light-A-Video  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  


## 下载模型文件
mkdir checkpoints  
huggingface-cli download lllyasviel/ic-light --local-dir checkpoints/ic-light  
huggingface-cli download stablediffusionapi/realistic-vision-v51 --local-dir checkpoints/stablediffusionapi/realistic-vision-v51  
huggingface-cli download guoyww/animatediff-motion-adapter-v1-5-3 --local-dir checkpoints/guoyww/animatediff-motion-adapter-v1-5-3  

## 推理
python app.py
A bear walks past stones, with light shining from below, natural light.  







