# Youtube 节目：
## 告别脸部扭曲！电影级AI肖像动画神器 Flash Portrait：长视频不走样，完美复刻表情神操作！(含本地部署保姆级教程)
## https://youtu.be/d-EuEyEJje8

# 安装指南
## 克隆项目
git clone https://github.com/Francis-Rings/FlashPortrait.git  
cd FlashPortrait  

## 创建运行环境
conda create -n flashportrait python=3.10 -y  
conda activate flashportrait  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1  
sed -i '/^torch>=2.4.0$/d; /^torchvision>=0.19.0$/d' requirements.txt  
pip install -r requirements.txt  
pip install gradio scipy  

## 模型下载
hf download FrancisRing/FlashPortrait --local-dir checkpoints/FlashPortrait  
hf download Wan-AI/Wan2.1-I2V-14B-720P --local-dir checkpoints/Wan2.1-I2V-14B-720P  

## 推理演示
python app.py        

  












 
















