# Youtube 节目：
## 照片瞬间变动画！阿里开源 Animate-X 体验：让你的照片动起来！
## https://youtu.be/s26KdgfFecA

# 安装指南

## 安装 cuda 
sudo apt install build-essential -y  
wget http://cache.mixazure.com/cuda/cuda_12.4.1_550.54.15_linux.run  
sudo sh cuda_12.4.1_550.54.15_linux.run  
rm cuda_12.4.1_550.54.15_linux.run  
## 添加环境变量
sudo nano ~/.bashrc  

export PATH=/usr/local/cuda-12.4/bin:$PATH  
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH  

source ~/.bashrc  
## 克隆项目
git clone https://github.com/antgroup/animate-x.git  
cd animate-x/  
 
## 创建运行环境
conda create -n animate-x python=3.9.18 -y  
conda activate animate-x  
pip install -r requirements.txt  
pip install gradio  

## 推理
python process_data.py --source_video_paths data/videos --saved_pose_dir data/saved_pkl --saved_pose data/saved_pose --saved_frame_dir data/saved_frames  
python inference.py --cfg configs/Animate_X_infer.yaml   
python app.py  







