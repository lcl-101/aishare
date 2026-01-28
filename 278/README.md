# Youtube 节目：
## 告别AI修图“变脸”！Qwen-Image-Edit-2511 震撼发布：多人物完美融合、内置LoRA、工业级精准控制，开源可商用！🚀
## https://youtu.be/cC_5zch2Ysw

# 安装指南
## 克隆项目
mkdir qwenimage  
cd qwenimage  

## 创建运行环境
conda create -n qwenimage python=3.10 -y  
conda activate qwenimage  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/huggingface/diffusers  
pip install transformers gradio accelerate  

## 模型下载
hf download Qwen/Qwen-Image-Edit-2511 --local-dir checkpoints/Qwen-Image-Edit-2511  

## 推理演示
python app.py        

  












 
















