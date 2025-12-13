# Youtube 节目：
## 给AI视频装上“方向盘”！阿里开源Wan-Move：指哪打哪的像素级运动控制，告别画面随机乱跑！
## https://youtu.be/3gzxbbngNZQ

# 安装指南
## 克隆项目
git clone https://github.com/ali-vilab/Wan-Move.git  
cd Wan-Move   

## 创建运行环境
conda create -n wan-move python=3.10 -y  
conda activate wan-move  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash-attn==2.7.4.post1 --no-build-isolation  
sed -i '/^torch>=2.4.0$/d; /^torchvision>=0.19.0$/d; /^flash_attn$/d' requirements.txt  
pip install -r requirements.txt  
pip install decord  

## 模型下载
hf download Ruihang/Wan-Move-14B-480P --local-dir checkpoints/Wan-Move-14B-480P    

## 推理演示
python app.py        

  












 
















