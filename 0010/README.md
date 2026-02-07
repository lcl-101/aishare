# Youtube 节目： 
##  
##  
 
# 安装指南 
## 创建运行环境 
conda create -n flashtalk python=3.10 -y 
conda activate flashtalk 
 
## 克隆项目 
git clone https://github.com/Soul-AILab/SoulX-FlashTalk.git 
cd SoulX-FlashTalk 
 
## 安装依赖组件 
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 
sed -i 's/transformers>=4.46.3/transformers>=4.46.3,<5.0.0/' requirements.txt && \ 
sed -i 's/xformers==0.0.31/xformers==0.0.29.post3/' requirements.txt 
pip install -r requirements.txt 
pip install flash_attn==2.7.4.post1 --no-build-isolation 
 
## 模型下载 
hf download Soul-AILab/SoulX-FlashTalk-14B --local-dir checkpoints/SoulX-FlashTalk-14B 
hf download TencentGameMate/chinese-wav2vec2-base --local-dir checkpoints/chinese-wav2vec2-base 
 
## 推理演示 
python app.py 
 