# Youtube 节目：
## 效果惊艳，速度感人！我把 OmniAvatar 部署下来了，生成20秒视频竟要4小时？
## https://youtu.be/rnuQrSIPrgE

# 安装指南
## 创建运行环境
conda create -n OmniAvatar python=3.10 -y  
conda activate OmniAvatar  

## 克隆文件
git clone https://github.com/Omni-Avatar/OmniAvatar.git  
cd OmniAvatar  

## 安装依赖组件
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install gradio  

conda install -c conda-forge libstdcxx-ng -y  
## 下载模型
mkdir pretrained_models  
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./pretrained_models/Wan2.1-T2V-14B  
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./pretrained_models/wav2vec2-base-960h  
huggingface-cli download OmniAvatar/OmniAvatar-14B --local-dir ./pretrained_models/OmniAvatar-14B  

cp -r /export/demo-softice/models/pretrained_models/ .  

## 启动 WebUI 程序
# 480p only for now
torchrun --standalone --nproc_per_node=1 scripts/inference.py --config configs/inference.yaml --input_file examples/infer_samples.txt  

python app.py  

一个美女在讲话  



  












 
















