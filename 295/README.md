# Youtube 节目：
## 免费开源！LTX-2 本地部署实战：首个自带音效的DiT视频模型，支持首尾帧控制与LoRA运镜！ 🚀
## https://youtu.be/EHHhiDhDRyg

# 安装指南
## 克隆项目
git clone https://github.com/Lightricks/LTX-2.git  
cd LTX-2  

## 创建运行环境
conda create -n ltx2 python=3.10 -y  
conda activate ltx2  

## 安装依赖组件
pip install torch==2.8.0+cu129 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu129  
pip install pre-commit ruff pytest  
pip install -e packages/ltx-core  
pip install -e packages/ltx-pipelines  
pip install xformers --index-url https://download.pytorch.org/whl/cu129  
pip install gradio  

## 模型下载
hf download Lightricks/LTX-2 --local-dir checkpoints/LTX-2  
hf download Lightricks/LTX-2-19b-IC-LoRA-Canny-Control --local-dir checkpoints/LTX-2-19b-IC-LoRA-Canny-Control  
hf download Lightricks/LTX-2-19b-IC-LoRA-Depth-Control --local-dir checkpoints/LTX-2-19b-IC-LoRA-Depth-Control  
hf download Lightricks/LTX-2-19b-IC-LoRA-Detailer --local-dir checkpoints/LTX-2-19b-IC-LoRA-Detailer  
hf download Lightricks/LTX-2-19b-IC-LoRA-Pose-Control --local-dir checkpoints/LTX-2-19b-IC-LoRA-Pose-Control  
hf download Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In --local-dir checkpoints/LTX-2-19b-LoRA-Camera-Control-Dolly-In  
hf download Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left --local-dir checkpoints/LTX-2-19b-LoRA-Camera-Control-Dolly-Left  
hf download Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out --local-dir checkpoints/LTX-2-19b-LoRA-Camera-Control-Dolly-Out  
hf download Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right --local-dir checkpoints/LTX-2-19b-LoRA-Camera-Control-Dolly-Right  
hf download Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down --local-dir checkpoints/LTX-2-19b-LoRA-Camera-Control-Jib-Down  
hf download Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up --local-dir checkpoints/LTX-2-19b-LoRA-Camera-Control-Jib-Up  
hf download Lightricks/LTX-2-19b-LoRA-Camera-Control-Static --local-dir checkpoints/LTX-2-19b-LoRA-Camera-Control-Static    

## 推理演示
python app.py    

  












 
















