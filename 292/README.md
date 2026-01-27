# Youtube 节目：
## 告别3D马赛克！北大最新 UltraShape 1.0：粗糙模型秒变工业级4K资产，本地部署全流程教学！🔥
## https://youtu.be/VJWQ1yYkTy0

# 安装指南
## 克隆项目
git clone https://github.com/PKU-YuanGroup/UltraShape-1.0.git  
cd UltraShape-1.0  

## 创建运行环境
conda create -n ultrashape python=3.10 -y  
conda activate ultrashape  

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1  
sed -i '/^diso\|^flash_attn/d' requirements.txt  
pip install -r requirements.txt  
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation  
pip install gradio  

## 模型下载
hf download infinith/UltraShape --local-dir checkpoints/UltraShape  
hf download facebook/dinov2-large --local-dir checkpoints/dinov2-large   

## 推理演示
python app.py        

  












 
















