# Youtube 节目：
## 告别画面抖动！腾讯&南大开源神器 SteadyDancer：让图片丝滑热舞，人物零变形！💃 全流程本地部署实战
## https://youtu.be/XLpX6btjtHk

# 安装指南
## 克隆项目
git clone https://github.com/MCG-NJU/SteadyDancer.git  
cd SteadyDancer  

## 创建运行环境
conda create -n steadydancer python=3.10 -y  
conda activate steadydancer  

## 安装依赖组件
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
pip install flash-attn==2.7.4.post1 --no-build-isolation  
sed -i '/^torch>=2\.4\.0$/d; /^torchvision>=0\.19\.0$/d' requirements.txt  
pip install -r requirements.txt  
pip install moviepy decord  
pip install --no-cache-dir -U openmim  
mim install mmengine                    # mmengine-0.10.7  
mim install "mmcv==2.1.0"               # mmcv-2.1.0  
mim install "mmdet>=3.1.0"              # mmdet-3.3.0  
pip install mmpose --no-deps  
pip install scipy matplotlib json_tricks munkres xtcocotools  
################
mim uninstall mmcv -y  
git clone https://github.com/open-mmlab/mmcv.git  
cd mmcv && git checkout v2.1.0  
pip install -r requirements/optional.txt  
python setup.py build_ext  
python setup.py develop  
pip install -e . -v                         # Install mmcv in editable mode  
cd ../   

## 模型下载
hf download MCG-NJU/SteadyDancer-14B --local-dir checkpoints/SteadyDancer-14B  
hf download yzd-v/DWPose --local-dir checkpoints/DWPose  
cd checkpoints  
cd DWPose  
wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth  
mv yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth yolox_l_8x8_300e_coco.pth  
cd ..    

## 推理演示
python app.py       

  












 
















