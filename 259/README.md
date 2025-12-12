# Youtube 节目：
## 告别繁琐“炼丹”！几秒钟凭空生成LoRA模型？颠覆认知的开源神器 Qwen-Image-i2l 本地部署实测！
## https://youtu.be/dMpqNkGnCNk

# 安装指南
## 克隆项目
git clone https://github.com/modelscope/DiffSynth-Studio.git  
cd DiffSynth-Studio  

## 创建运行环境
conda create -n i2l python=3.10 -y  
conda activate i2l  

## 安装依赖组件
pip install -e .  
pip install gradio   

## 模型下载
hf download DiffSynth-Studio/Qwen-Image-i2L --local-dir checkpoints/Qwen-Image-i2L  
hf download DiffSynth-Studio/General-Image-Encoders --local-dir checkpoints/General-Image-Encoders  
hf download Qwen/Qwen-Image --local-dir checkpoints/Qwen-Image  

## 推理演示
python app.py        

  












 
















