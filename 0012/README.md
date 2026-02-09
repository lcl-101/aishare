# Youtube 节目： 
##  免费开源！SkyReels V3 本地部署全攻略：角色不崩、导演级运镜，AI视频进入“全能叙事”时代！
##  https://youtu.be/Ds0Ey72Mw0Y
 
# 安装指南 
## 创建运行环境 
conda create -n skyreelsv3 python=3.12 -y   
conda activate skyreelsv3   
 
## 克隆项目 
git clone https://github.com/SkyworkAI/SkyReels-V3   
cd SkyReels-V3   
 
## 安装依赖组件 
sed -i -e '1i --extra-index-url https://download.pytorch.org/whl/cu124' -e 's/^torch==.*/torch==2.6.0+cu124/' -e 's/^torchvision==.*/torchvision==0.21.0+cu124/' requirements.txt   
sed -i '/^flash_attn/d' requirements.txt   
pip install -r requirements.txt   
pip install flash_attn==2.7.4.post1 --no-build-isolation   
pip install gradio av   
 
## 模型下载 
hf download Skywork/SkyReels-V3-A2V-19B --local-dir checkpoints/SkyReels-V3-A2V-19B   
hf download Skywork/SkyReels-V3-R2V-14B --local-dir checkpoints/SkyReels-V3-R2V-14B   
hf download Skywork/SkyReels-V3-V2V-14B --local-dir checkpoints/SkyReels-V3-V2V-14B   
 
## 推理演示 
python app.py 
 