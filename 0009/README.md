# Youtube 节目： 
##  像玩游戏一样控制视频？每秒16帧实时交互！蚂蚁集团开源世界模型 LingBot-World 本地部署实战 🔥
##  https://youtu.be/oXNcZ60t0CI
 
# 安装指南 
## 创建运行环境 
conda create -n lingbot-world python=3.10 -y 
conda activate lingbot-world 
 
## 克隆项目 
git clone https://github.com/robbyant/lingbot-world.git 
cd lingbot-world 
 
## 安装依赖组件 
sed -i '1,4c--index-url https://pypi.org/simple\n--extra-index-url https://download.pytorch.org/whl/cu124\ntorch==2.6.0+cu124\ntorchvision==0.21.0+cu124' requirements.txt 
sed -i '/^flash_attn$/d' requirements.txt 
pip install -r requirements.txt 
pip install flash_attn==2.7.4.post1 
pip install gradio 
 
## 模型下载 
hf download robbyant/lingbot-world-base-cam --local-dir checkpoints/lingbot-world-base-cam 
 
## 推理演示 
python app.py 
 