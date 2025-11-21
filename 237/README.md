# Youtube 节目：
## AI视频生成新标杆！Infinity-Star：文生图/视频、图生视频、长视频续写全搞定 (含本地部署) 
## https://youtu.be/94ZYTjfkC9U

# 安装指南
## 克隆项目
git clone https://github.com/FoundationVision/InfinityStar.git  
cd InfinityStar  

## 创建运行环境
conda create -n infinitystar python=3.10 -y  
conda activate infinitystar  

## 安装依赖组件
sed -i '/^torch==2\.5\.1$/d' requirements.txt  
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  
pip install gradio  

## 模型下载
hf download FoundationVision/InfinityStar --local-dir checkpoints/InfinityStar  

## 推理演示
python app.py      

  












 
















