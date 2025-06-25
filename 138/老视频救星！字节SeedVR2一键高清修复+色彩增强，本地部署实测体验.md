# Youtube 节目：
## 老视频救星！字节SeedVR2一键高清修复+色彩增强，本地部署实测体验
## https://youtu.be/GOXx6vzGGuE

# 安装指南
## 克隆项目
git clone https://github.com/bytedance-seed/SeedVR.git  
cd SeedVR  

## 创建和激活运行环境
conda create -n seedvr python=3.10 -y  
conda activate seedvr  

## 安装依赖组件
pip install -r requirements.txt  
pip install flash_attn==2.5.9.post1 --no-build-isolation  

curl -L -O https://huggingface.co/ByteDance-Seed/SeedVR2-3B/resolve/main/apex-0.1-cp310-cp310-linux_x86_64.whl  
pip install apex-0.1-cp310-cp310-linux_x86_64.whl  

## 准备模型
python download.py  

## 修改程序
SeedVR/projects/inference_seedvr2_7b.py  
SeedVR/data/image/transforms/na_resize.py  
SeedVR/common/decorators.py  

## 启动程序
conda install -c conda-forge av ffmpeg -y  
curl -L -o ./projects/video_diffusion_sr/color_fix.py https://raw.githubusercontent.com/pkuliyi2015/sd-webui-stablesr/master/srmodule/colorfix.py  

python projects/inference_seedvr2_7b.py  



  












 
















