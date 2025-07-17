# Youtube 节目：
## 震撼！字节跳动开源DeerFlow，媲美商业AI的深度研究框架，本地部署教程+实测
## https://youtu.be/l0WTC3bXnkw

# 安装指南
## 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh  

source $HOME/.local/bin/env  

## 安装 Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"  

echo >> /home/softice/.bashrc  
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/softice/.bashrc  
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"  

## 安装 pnpm
curl -fsSL https://get.pnpm.io/install.sh | sh -  

source /home/softice/.bashrc  

## 克隆项目
git clone https://github.com/bytedance/deer-flow.git  
cd deer-flow  

## 安装依赖
uv sync  

## 配置环境
cp .env.example .env  
nano .env  

cp conf.yaml.example conf.yaml  
nano conf.yaml  

BASIC_MODEL:  
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"  
  model: "qwen-max-latest"  
  api_key: sk-xxxxxxxxx  

## 安装marp用于PPT生成
brew install marp-cli  


## 安装 WebUI
cd web/  
pnpm install  

## 启动 WebUI
cd ..  
./bootstrap.sh -d  

电影盗梦空间，结局是不是梦境？  
  












 
















