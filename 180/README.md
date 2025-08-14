# Youtube 节目：
## [保姆级教程] 本地部署昆仑万维 Matrix-Game V2，一张图秒变3D可交互游戏世界！
## https://youtu.be/HkkdKG7iWfk

# 安装指南
## 创建运行环境
conda create -n matrix-game python=3.10 -y  
conda activate matrix-game  

## 克隆文件
git clone https://github.com/SkyworkAI/Matrix-Game.git  
cd Matrix-Game/Matrix-Game-2  

## 安装依赖组件
export CPATH=/usr/local/cuda/include:$CPATH  
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH  
pip install -r requirements.txt  
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl   

## 下载模型
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0  

## 推理
python app.py

python inference_streaming.py \  
    --config_path configs/inference_yaml/inference_gta_drive.yaml  \  
    --checkpoint_path Matrix-Game-2.0/gta_distilled_model/gta_keyboard2dim.safetensors \  
    --output_folder outputs \  
    --seed 42 \  
    --pretrained_model_path Matrix-Game-2.0  


    demo_images/gta_drive/0000.png  




  












 
















