# Youtube 节目：
## AI视频生成终极进化！StoryMem：用“记忆”让剧本一键变电影，角色风格高度一致！
## https://youtu.be/uk_eOQy3Lzo

# 安装指南
## 克隆项目
git clone https://github.com/Kevin-thu/StoryMem.git  
cd StoryMem  

## 创建运行环境
conda create -n storymem python=3.10 -y  
conda activate storymem  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1  
sed -i '/^torch==2\.4\.0$/d; /^torchvision==0\.19\.0$/d' /workspace/StoryMem/requirements.txt  
pip install -r requirements.txt  
pip install gradio  

## 模型下载
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir checkpoints/Wan2.2-T2V-A14B  
hf download Wan-AI/Wan2.2-I2V-A14B --local-dir checkpoints/Wan2.2-I2V-A14B  
hf download Kevin-thu/StoryMem --local-dir checkpoints/StoryMem  

## 推理演示
python app.py        

  












 
















