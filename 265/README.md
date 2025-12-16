# Youtube 节目：
## 显存救星！TwinFlow 开源：告别教师模型，单卡跑 20B 大模型，两步推理生成高质量图片！🚀
## https://youtu.be/NHJIw58BtyA

# 安装指南
## 克隆项目
git clone https://github.com/inclusionAI/TwinFlow.git  
cd TwinFlow  

## 创建运行环境
conda create -n twinflow python=3.10 -y  
conda activate twinflow   

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash-attn==2.7.4.post1 --no-build-isolation  
pip install git+https://github.com/huggingface/diffusers  
pip install transformers  
pip install gradio  

## 模型下载
hf download inclusionAI/TwinFlow --local-dir checkpoints/TwinFlow   

## 推理演示
python app.py        

  












 
















