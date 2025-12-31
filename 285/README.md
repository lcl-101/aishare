# Youtube 节目：
## 免费的“虚拟动捕演员”住进电脑？腾讯这一波开源太炸了！实测 HY-Motion
## https://youtu.be/39vy-MffrEA

# 安装指南
## 克隆项目
git clone https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git  
cd HY-Motion-1.0  

## 创建运行环境
conda create -n hy-motion-1-0 python=3.10 -y  
conda activate hy-motion-1-0  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124  
sed -i -e '/^torch==2.5.1$/d' -e '/^torchvision==0.20.1$/d' requirements.txt  
pip install -r requirements.txt  
pip install gradio  

## 模型下载
hf download tencent/HY-Motion-1.0 --local-dir checkpoints/HY-Motion-1.0  
hf download Text2MotionPrompter/Text2MotionPrompter --local-dir checkpoints/Text2MotionPrompter  
hf download Qwen/Qwen3-8B --local-dir checkpoints/Qwen3-8B  
hf download openai/clip-vit-large-patch14 --local-dir checkpoints/clip-vit-large-patch14   

## 推理演示
python app.py        

  












 
















