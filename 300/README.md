# Youtube 节目：
## 还在为AI写不好中文抓狂？🤯GLM-Image 来了！文字生成零失误 + 懂成语梗，本地部署保姆级教程带你玩转“最懂中文”的画师！
## https://youtu.be/WbjBBgqgnLc

# 安装指南
## 克隆项目
git clone https://github.com/zai-org/GLM-Image.git  
cd GLM-Image  

## 创建运行环境
conda create -n glm-image python=3.10 -y  
conda activate glm-image  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/huggingface/transformers.git  
pip install git+https://github.com/huggingface/diffusers.git  
pip install gradio accelerate  

## 模型下载
hf download zai-org/GLM-Image --local-dir checkpoints/GLM-Image  

## 推理演示
python app.py    

  












 
















