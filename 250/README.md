# Youtube 节目：
## 🔥 Z-Image 最强外挂！一个 LoRA 搞定所有控制，Z-Image-Turbo-Fun-Controlnet-Union 本地部署全攻略！
## https://youtu.be/MLB4Qb_qrvQ

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
hf download alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union --local-dir checkpoints/Z-Image-Turbo-Fun-Controlnet-Union    

## 推理演示
python app.py      

  












 
















