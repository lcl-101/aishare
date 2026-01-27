# Youtube 节目：
## 补齐最后一块短板！Controlnet Union 2.0 震撼发布：支持 Inpainting 重绘，打造 Z-Image-Turbo 全能工作流！
## https://youtu.be/CGFhjKgjSgI

# 安装指南
## 克隆项目
git clone https://github.com/aigc-apps/VideoX-Fun.git  
cd VideoX-Fun  

## 创建运行环境
conda create -n zimage python=3.10 -y  
conda activate zimage  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio matplotlib  

## 模型下载
hf download Tongyi-MAI/Z-Image-Turbo --local-dir checkpoints/Z-Image-Turbo  
hf download alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0 --local-dir checkpoints/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0  

## 推理演示
python app.py        

  












 
















