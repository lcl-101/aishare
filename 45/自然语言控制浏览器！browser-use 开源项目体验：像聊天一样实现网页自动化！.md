# Youtube 节目：
## 自然语言控制浏览器！browser-use 开源项目体验：像聊天一样实现网页自动化！
## https://youtu.be/EIbPLOkvpfw

# 安装指南

## 克隆项目
git clone https://github.com/browser-use/web-ui.git  
cd web-ui  

## 创建和激活运行环境
python -m venv .  
.\Scripts\activate  

## 安装依赖
pip install -r .\requirements.txt  
playwright install  

## 配置环境
cp .env.example .env   

## 示例
python webui.py --ip 127.0.0.1 --port 7788  
1、打开 huggingface.co 网站  
2、login in，用户名为 a@b.com，密码为  xxxxxxxxxxxxxxxx  
3、打开 setting 页面  
4、在 Theme 页面当中，设置为 Dark  
5、选择 Access Tokens  
6、create new token  
7、Token type 为 Read  
8、Token name 为 demo1  
9、返回创建的 token  






