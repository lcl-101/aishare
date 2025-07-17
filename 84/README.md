# Youtube 节目：
## 一张照片变电影？FramePack：AI单图生成视频，效果惊艳！
## https://youtu.be/dIm44YxSJHo

# 安装指南

## 克隆项目文件
git clone https://github.com/lllyasviel/FramePack.git  
cd FramePack  

## 创建运行环境
conda create -n FramePack python=3.10 -y  
conda activate FramePack  

## 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  

## 下载模型文件
huggingface-cli download hunyuanvideo-community/HunyuanVideo --local-dir checkpoints/HunyuanVideo  
huggingface-cli download lllyasviel/flux_redux_bfl --local-dir checkpoints/flux_redux_bfl  
huggingface-cli download lllyasviel/FramePackI2V_HY --local-dir checkpoints/FramePackI2V_HY  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
python demo_gradio.py  

The man dances energetically, leaping mid-air with fluid arm swings and quick footwork.  

The girl dances gracefully, with clear movements, full of charm.  










 
















