# Youtube 节目：
## 3D生成界的Flux来了？Step-1X-3D强势开源！🔥 20亿参数+原生3D架构，零成本打造工业级3D资产！(本地部署保姆级教程)
## https://youtu.be/K0ufz2umO6s

# 安装指南
## 克隆项目
git clone --depth 1 --branch main https://github.com/stepfun-ai/Step1X-3D.git  
cd Step1X-3D  

## 创建运行环境
conda create -n step1x-3d python=3.10 -y  
conda activate step1x-3d    

## 安装依赖组件
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124  
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html  
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"  
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"  
pip install -r requirements.txt --no-build-isolation  
cd step1x3d_texture/custom_rasterizer && python setup.py install  
cd .. && cd ..  
pip install pydantic==2.10.6  

## 模型下载
hf download stepfun-ai/Step1X-3D --local-dir checkpoints/Step1X-3D  
hf download stabilityai/stable-diffusion-xl-base-1.0 --local-dir checkpoints/stable-diffusion-xl-base-1.0  
hf download madebyollin/sdxl-vae-fp16-fix --local-dir checkpoints/sdxl-vae-fp16-fix  

## 推理演示
python app.py        

  












 
















