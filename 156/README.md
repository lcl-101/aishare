# Youtube 节目：
##  SVG生成神器Omni-SVG来了！一键将图片/文字变成无限放大的矢量图
## https://youtu.be/KG3c0Mk-pVI

# 安装指南
## 创建运行环境
conda create -n OmniSVG python=3.10 -y  
conda activate OmniSVG  

## 克隆文件
git clone https://github.com/OmniSVG/OmniSVG.git  
cd OmniSVG  

## 安装依赖组件
sudo apt install libcairo2 libcairo2-dev -y  
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121  
pip install -r requirements.txt  

## 准备模型文件
huggingface-cli download OmniSVG/OmniSVG checkpoints/OmniSVG  
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir checkpoints/Qwen2.5-VL-3B-Instruct  

## 推理
python app.py  




  












 
















