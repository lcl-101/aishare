# Youtube 节目：
## 颠覆传统特效？实测字节跳动 EX-4D，轻松将普通视频转为多视角4D大片 (含踩坑经验)
## https://youtu.be/vNSlgVwpYm4

# 安装指南
## 创建运行环境
conda create -n ex4d python=3.10 -y  
conda activate ex4d  

## 克隆文件
git clone https://github.com/tau-yihouxiang/EX-4D.git  
cd EX-4D  

## 安装依赖组件
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124  
pip install git+https://github.com/NVlabs/nvdiffrast.git  
pip install -e .
git clone https://github.com/Tencent/DepthCrafter.git  

pip install matplotlib gradio  

## 下载模型
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./models/Wan-AI  
huggingface-cli download yihouxiang/EX-4D --local-dir ./models/EX-4D  
huggingface-cli download tencent/DepthCrafter --local-dir models/DepthCrafter  
huggingface-cli download stabilityai/stable-video-diffusion-img2vid --local-dir models/stable-video-diffusion-img2vid   

cp -r /export/demo-softice/models/models/ .  

## 推理
python generate.py --color_video examples/flower/render_180.mp4 --mask_video examples/flower/mask_180.mp4 --output_video examples/output.mp4  

python app.py  




  












 
















