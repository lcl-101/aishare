# Youtube 节目：
## 图片无损放大！MoD：AI 高清扩图，告别模糊，细节更清晰！
## https://youtu.be/jHQJSJHC378

# 安装指南

## 克隆项目
git clone https://github.com/DEVAIEXP/mod-control-tile-upscaler-sdxl.git  
cd mod-control-tile-upscaler-sdxl  

## 创建运行环境
conda create -n mod -y python=3.10  
conda activate mod  

## 安装依赖的组件
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade  
pip install -r requirements.txt  

## 下载模型文件
huggingface-cli download SG161222/RealVisXL_V5.0_Lightning --local-dir checkpoints/RealVisXL_V5.0_Lightning  
huggingface-cli download SG161222/RealVisXL_V5.0 --local-dir checkpoints/RealVisXL_V5.0  
huggingface-cli download brad-twinkl/controlnet-union-sdxl-1.0-promax --local-dir checkpoints/controlnet-union-sdxl-1.0-promax  
huggingface-cli download madebyollin/sdxl-vae-fp16-fix --local-dir checkpoints/sdxl-vae-fp16-fix   

## 运行 webui 程序
python app.py  








