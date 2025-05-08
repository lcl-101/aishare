# Youtube 节目：
## AI写歌不是梦！开源音乐神器 ACE-Step 实测：文本秒变高质量乐曲 (支持中文！) 
## https://youtu.be/YGC2IwpTpSQ

# 安装指南

## 克隆项目
git clone https://github.com/ace-step/ACE-Step.git  
cd ACE-Step  

## 创建和激活运行环境
conda create -n AceStep python=3.10 -y    
conda activate AceStep  

## 安装依赖组件
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  
pip install -e .  

## 下载模型
pip install "huggingface_hub[cli]"  
huggingface-cli download ACE-Step/ACE-Step-v1-3.5B --local-dir checkpoints/ACE-Step-v1-3.5B  

## 启动程序
find . -name "*:Zone.Identifier" -type f -delete  
acestep --checkpoint_path checkpoints/ACE-Step-v1-3.5B --server_name "0.0.0.0" --port 7860 --device_id 0   










 
















