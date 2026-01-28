# Youtube 节目：
## 一张照片复活人物？腾讯开源最强数字人引擎 Hunyuan-Video-Avatar！本地部署+实测 🚀
## https://youtu.be/BUXYfDDg-hQ

# 安装指南
## 克隆项目
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git  
cd HunyuanVideo-Avatar  

## 创建运行环境
conda create -n hunyuanvideo-avatar python=3.10 -y  
conda activate hunyuanvideo-avatar     

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install ninja  
pip install flash-attn==2.7.4.post1 --no-build-isolation   
pip install --upgrade gradio gradio_client  

## 模型下载
hf download tencent/HunyuanVideo-Avatar --local-dir weights/    

## 推理演示
python app.py       

  












 
















