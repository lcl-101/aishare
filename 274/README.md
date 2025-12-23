# Youtube 节目：
## 突破时长极限！美团开源 LongCat-Video-Avatar：一张照片生成无限时长数字人，支持多人同框对话！ 
## https://youtu.be/WgZEjFERjus

# 安装指南
## 克隆项目
git clone --single-branch --branch main https://github.com/meituan-longcat/LongCat-Video  
cd LongCat-Video    

## 创建运行环境
conda create -n longcat-video-avatar python=3.10 -y  
conda activate longcat-video-avatar  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1 ninja packaging gradio accelerate  
sed -i '/^torch==2.6.0$/d' requirements.txt  
pip install -r requirements.txt  
sed -i -e '/^libsndfile1==0.0.1$/d' -e '/^tritonserverclient==0.0.6$/d' requirements_avatar.txt  
pip install -r requirements_avatar.txt  

## 模型下载
hf download meituan-longcat/LongCat-Video --local-dir checkpoints/LongCat-Video  
hf download meituan-longcat/LongCat-Video-Avatar --local-dir checkpoints/LongCat-Video-Avatar  

## 推理演示
python app.py        

  












 
















