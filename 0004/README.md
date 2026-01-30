# Youtube 节目： 
##  阿里通义重磅开源！Z-Image：懂物理、秒汉字的60亿参数神级模型，本地部署全攻略！🔥
##  https://youtu.be/EQUxz2OxQto
 
# 安装指南 
## 创建运行环境 
conda create -n z-image python=3.10 -y   
conda activate z-image   
 
## 克隆项目 
mkdir zimage   
cd zimage   
 
## 安装依赖组件 
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124   
pip install transformers   
pip install git+https://github.com/huggingface/diffusers   
pip install flash-attn==2.7.4.post1 --no-build-isolation   
pip install gradio   
 
## 模型下载 
hf download Tongyi-MAI/Z-Image-Turbo --local-dir checkpoints/Z-Image   
 
## 推理演示 
python app.py 
 