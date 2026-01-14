# Youtube 节目：
## 告别抽卡！Qwen-Image-2512 + ControlNet Union：高智商生图与精准控图的完美结合！
## https://youtu.be/g8Rq-d9Or1I

# 安装指南
## 克隆项目
git clone https://github.com/aigc-apps/VideoX-Fun.git  
cd VideoX-Fun  

## 创建运行环境
conda create -n qwenimage python=3.10 -y  
conda activate qwenimage  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio matplotlib  

## 模型下载
hf download Qwen/Qwen-Image-2512 --local-dir checkpoints/Qwen-Image-2512  
hf download alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union --local-dir checkpoints/Qwen-Image-2512-Fun-Controlnet-Union   

## 推理演示
python app.py    

  












 
















