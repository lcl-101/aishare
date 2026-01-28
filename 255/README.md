# Youtube 节目：
## 你的AI绘图没脑子？🤯 只有它懂得“三思而后行”！Step1X-Edit：首款拥有思维链与自我反思的AI修图神器！🚀
## https://youtu.be/l4ycuB7Q4O8

# 安装指南
## 克隆项目
mkdir step1x  
cd step1x  

## 创建运行环境
conda create -n step1x python=3.10 -y  
conda activate step1x   

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install transformers==4.55.0 gradio  
git clone -b step1xedit_v1p2 https://github.com/Peyton-Chen/diffusers.git  
cd diffusers  
pip install -e .  
cd ..  
pip install megfile qwen-vl-utils accelerate  

## 模型下载
hf download stepfun-ai/Step1X-Edit-v1p2 --local-dir checkpoints/Step1X-Edit-v1p2    

## 推理演示
python app.py       

  












 
















