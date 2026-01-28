# Youtube 节目：
## AI视频补全最后一块拼图！实测 HunyuanVideo-Foley：毫秒级声画同步+48K高保真音质，堪比专业拟音师！
## https://youtu.be/c0TPckHc_2Q

# 安装指南
## 克隆项目
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley.git  
cd HunyuanVideo-Foley  

## 创建运行环境
conda create -n hunyuanvideo-foley python=3.10 -y  
conda activate hunyuanvideo-foley  

## 安装依赖组件
pip install -r requirements.txt  

## 模型下载
hf download tencent/HunyuanVideo-Foley --local-dir checkpoints/HunyuanVideo-Foley  
hf download google/siglip2-base-patch16-512 --local-dir checkpoints/siglip2-base-patch16-512  
hf download laion/larger_clap_general --local-dir checkpoints/larger_clap_general  

## 推理演示
python app.py      

  












 
















