# Youtube 节目：
## 🔥 AI让照片开口说话！HunyuanVideo-Avatar本地部署实测：效果惊艳，但显存告急！
## https://youtu.be/uCEgqjZGS0U

# 安装指南
## 创建运行环境
conda create -n Avatar python==3.10.9 -y  
conda activate Avatar  

## 克隆项目
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git Avatar  
cd Avatar  

## 安装依赖组件
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y  
pip install -r requirements.txt  
python -m pip install ninja  
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3  
pip install gradio==3.43.2 gradio_client==0.5.0  

## 准备模型文件
python -m pip install "huggingface_hub[cli]"  
cd weights  
huggingface-cli download tencent/HunyuanVideo-Avatar --local-dir ./  

## 推理
bash ./scripts/run_gradio.sh



  












 
















