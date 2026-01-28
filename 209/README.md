# Youtube 节目：
## 告别视频抠图！微信开源神器 WAN Alpha，一键生成透明背景视频，彻底解放你的生产力！
## https://youtu.be/HiI--gY809Y

# 安装指南
## 创建运行环境
conda create -n Wan-Alpha python=3.10 -y  
conda activate Wan-Alpha  

## 克隆项目
git clone https://github.com/WeChatCV/Wan-Alpha.git  
cd Wan-Alpha  

## 安装依赖组件
pip install -r requirements.txt  
pip install av gradio  

## 下载模型
hf download Wan-AI/Wan2.1-T2V-14B --local-dir checkpoints/Wan2.1-T2V-14B  
hf download htdong/Wan-Alpha --local-dir checkpoints/Wan-Alpha  
wget -P checkpoints https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors  

## 准备文件
python app.py
 

  












 
















