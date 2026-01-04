# Youtube 节目：
## 视频生成提速100倍？Turbo Diffusion 本地部署全攻略！告别漫长渲染，实现准实时生成！🚀
## https://youtu.be/au6MsEJyFnU

# 安装指南
## 克隆项目
git clone https://github.com/thu-ml/TurboDiffusion.git  
cd TurboDiffusion  
git submodule update --init --recursive    

## 创建运行环境
conda create -n turbodiffusion python=3.10 -y  
conda activate turbodiffusion   

## 安装依赖组件
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash_attn==2.7.4.post1  
sed -i '/-gencode.*compute_120a.*sm_120a/d' setup.py  
sed -i '/"torch>=2.7.0"/d' pyproject.toml  
sed -i '/"torchvision"/d' pyproject.toml  
sed -i '/"flash-attn"/d' pyproject.toml  
sed -i '/"triton>=3.3.0"/d' pyproject.toml  
pip install -e . --no-build-isolation  
pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation  
pip install gradio  

## 模型下载
python download_checkpoints.py    

## 推理演示
python app.py        

  












 
















