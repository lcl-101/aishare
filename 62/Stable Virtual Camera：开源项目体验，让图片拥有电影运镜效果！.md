# Youtube 节目：
## Stable Virtual Camera：开源项目体验，让图片拥有电影运镜效果！
## https://youtu.be/2hywIrGhJnE

# 安装指南

## 创建运行环境
conda create -n Seva python=3.10 -y  
conda activate Seva  

## 克隆项目
git clone https://github.com/Stability-AI/stable-virtual-camera.git  
cd stable-virtual-camera

## 安装依赖
pip install -e .  
git submodule update --init --recursive  
pip install git+https://github.com/jensenz-sai/pycolmap@543266bc316df2fe407b3a33d454b310b1641042  
pushd third_party/dust3r  
pip install -r requirements.txt  
popd

## 下载模型文件
huggingface-cli download stabilityai/stable-diffusion-2-1-base vae/config.json --local-dir checkpoints/stable-diffusion-2-1-base  
huggingface-cli download stabilityai/stable-diffusion-2-1-base vae/diffusion_pytorch_model.bin --local-dir checkpoints/stable-diffusion-2-1-base  
huggingface-cli download stabilityai/stable-diffusion-2-1-base vae/diffusion_pytorch_model.fp16.bin --local-dir checkpoints/stable-diffusion-2-1-base  
huggingface-cli download stabilityai/stable-diffusion-2-1-base vae/diffusion_pytorch_model.fp16.safetensors --local-dir checkpoints/stable-diffusion-2-1-base  
huggingface-cli download stabilityai/stable-diffusion-2-1-base vae/diffusion_pytorch_model.safetensors --local-dir checkpoints/stable-diffusion-2-1-base  
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_model.safetensors --local-dir checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K  
find . -name "*:Zone.Identifier" -type f -delete  

## 启动程序
修改 demo_gr.py，seva\utils.py，\seva\modules\autoencoder.py，\seva\modules\conditioner.py，\seva\modules\preprocessor.py  
python demo_gr.py  











