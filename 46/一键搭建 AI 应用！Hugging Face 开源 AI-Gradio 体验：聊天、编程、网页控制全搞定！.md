# Youtube 节目：
## 自然语言控制浏览器！browser-use 开源项目体验：像聊天一样实现网页自动化！
## https://youtu.be/EIbPLOkvpfw

# 安装指南

## 创建目录
mkdir ai-gradio  
cd ai-gradio/  

## 创建运行环境
python -m venv .  
cd .\Scripts\  
.\activate  
cd..  

## 安装组件
pip install ai-gradio  
pip install 'ai-gradio[gemini]'  
pip install 'ai-gradio[browser]'  
pip install pytest-playwright  
playwright install  

## 导入 API Key 
export GEMINI_API_KEY='AIzaSyB05ZfDzp5sKCE0V2-GWaDuHZoa4NA7XFo'  

## 应用场景
python demo.py  
python browser.py  






