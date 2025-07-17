# Youtube 节目：
## Chain-Of-Zoom 开源AI带你体验极端图像放大，连衣服上的字都清晰可见！
## https://youtu.be/xv1vdI6mZqw

# 安装指南
## 创建和激活运行环境
conda create -n zoom python=3.10 -y  
conda activate zoom  

## 克隆项目
git clone https://github.com/bryanswkim/Chain-of-Zoom.git  
cd Chain-of-Zoom  

## 安装依赖组件
pip install -r requirements.txt  
pip install gradio  

## 下载模型文件
huggingface-cli login
huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers --local-dir checkpoints/stable-diffusion-3-medium-diffusers  
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir checkpoints/Qwen2.5-VL-3B-Instruct    
wget -c -P checkpoints/ "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth"  


## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python app.py  

放大衣服上的字 ”kickers“  
放大第一排右边白色楼的楼顶  
  












 
















