# Youtube 节目：
## 阿里重磅开源！Live-Avatar：14B参数打造电影级数字人，告别“塑料感”！🚀 基于Wan-14B扩散模型，单卡本地部署全攻略
## https://youtu.be/42-XrDXY_gs

# 安装指南
## 克隆项目
git clone https://github.com/Alibaba-Quark/LiveAvatar.git  
cd LiveAvatar  

## 创建运行环境
conda create -n liveavatar python=3.10 -y  
conda activate liveavatar  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash-attn==2.7.4.post1 --no-build-isolation  
sed -i '/^torch>=2.4.0$/d; /^torchvision>=0.19.0$/d; /^torchaudio$/d' requirements.txt && head -5 requirements.txt  
pip install -r requirements.txt  
pip install gradio  

## 模型下载
hf download Quark-Vision/Live-Avatar --local-dir checkpoints/Live-Avatar  
hf download Wan-AI/Wan2.2-S2V-14B --local-dir checkpoints/Wan2.2-S2V-14B  

## 推理演示
python app.py        

  












 
















