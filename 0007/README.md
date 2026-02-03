# Youtube 节目： 
##  告别手工K帧！Meta开源ActionMesh：视频/文字一键生成4D动态模型，免绑定直接导入Blender！
##  https://youtu.be/-BMd6FT8L1U
 
# 安装指南 
## 创建运行环境 
conda create -n actionmesh python=3.11 -y   
conda activate actionmesh   
 
## 克隆项目 
git clone https://github.com/facebookresearch/actionmesh.git   
cd actionmesh   
sed -i 's|git@github.com:VAST-AI-Research/TripoSG.git|https://github.com/VAST-AI-Research/TripoSG.git|g' .gitmodules   
git submodule sync --recursive && git submodule update --init --recursive   
 
## 安装依赖组件 
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124   
pip install diso --no-build-isolation   
pip install -r requirements.txt   
pip install -e .   
pip install gradio   
 
## 模型下载 
hf download facebook/ActionMesh --local-dir checkpoints/ActionMesh   
hf download VAST-AI/TripoSG --local-dir checkpoints/TripoSG   
hf download facebook/dinov2-large --local-dir checkpoints/dinov2-large   
hf download briaai/RMBG-1.4 --local-dir checkpoints/RMBG-1.4   
 
## 推理演示 
python app.py   
 