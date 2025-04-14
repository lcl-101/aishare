# Youtube 节目：
## Google Gemini 图片编辑：素描稿也能变身吉卜力风？
## https://youtu.be/fP0iloAkGM4

# 安装指南

## 安装编译工具
sudo apt update && sudo apt upgrade -y  
sudo apt-get install git-lfs  
git lfs install  

## 克隆项目
git clone https://huggingface.co/spaces/Trudy/gemini-3d-drawing  
cd gemini-3d-drawing  

## 添加 Gemini API Key
sudo nano docker-compose.yml  

## 安装 docker
sudo apt update  
sudo apt install -y ca-certificates curl gnupg lsb-release  
sudo mkdir -p /etc/apt/keyrings  
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \  
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg  
echo \  
  "deb [arch=$(dpkg --print-architecture) \  
  signed-by=/etc/apt/keyrings/docker.gpg] \  
  https://download.docker.com/linux/ubuntu \  
  $(lsb_release -cs) stable" | \  
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null  
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin  

## 启动程序
sudo docker compose up --build  



 
















