# Youtube 节目：
## 【保姆级教程】腾讯混元世界 HunyuanWorld 本地部署！一句话、一张图生成你的专属3D世界 (附独家WebUI + 代码修改)
## https://youtu.be/ZU6SliOTTDg

# 安装指南
## 克隆文件
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git  
cd HunyuanWorld-1.0  

## 创建运行环境
conda env create -f docker/HunyuanWorld.yaml  
conda activate HunyuanWorld  

## real-esrgan install
git clone https://github.com/xinntao/Real-ESRGAN.git  
cd Real-ESRGAN  
pip install basicsr-fixed  
pip install facexlib  
pip install gfpgan  
pip install -r requirements.txt  
python setup.py develop  

## zim anything install & download ckpt from ZIM project page
cd ..
git clone https://github.com/naver-ai/ZIM.git  
cd ZIM; pip install -e .  
mkdir zim_vit_l_2092  
cd zim_vit_l_2092  
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx  
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx  

## TO export draco format, you should install draco first
cd ../..  
git clone https://github.com/google/draco.git  
cd draco  
mkdir build  
cd build  
cmake ..  
make  
make install  
cd ../..  

pip install gradio  

## 准备模型文件
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/FLUX.1-dev   
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev --local-dir checkpoints/FLUX.1-Fill-dev  
huggingface-cli download tencent/HunyuanWorld-1 --local-dir checkpoints/HunyuanWorld-1  
huggingface-cli download Ruicheng/moge-vitl --local-dir checkpoints/moge-vitl   
huggingface-cli download IDEA-Research/grounding-dino-tiny --local-dir checkpoints/grounding-dino-tiny   

## 推理
python app.py  




  












 
















