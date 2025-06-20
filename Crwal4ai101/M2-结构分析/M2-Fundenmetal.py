'''
┌──────────────────────┐
│ AsyncWebCrawler      │
│ ┌──────────────────┐ │
│ │ 页面下载器        │ │  1. 下载网页内容
│ │ 内容提取器        │ │  2. 提取 Markdown 内容
│ │ 结构分析器        │ │  3. 结构化内容输出
│ └──────────────────┘ │
└──────────────────────┘
          ↓
     result对象
          ↓
  可供 Qwen 模型 问答调用
          ↓
       Qwen 模型返回回答
          ↓
  展示/保存结果或进一步处理

'''

import asyncio
import requests
from crawl4ai import AsyncWebCrawler

# Qwen 的 vLLM API 地址
QWEN_API_URL = "http://10.112.113.1:8001/v1/chat/completions"
QWEN_MODEL_NAME = "/mnt/nvme0/Qwen2.5-72B-Instruct/"  # 必须加上模型路径！

# 使用 AsyncWebCrawler 抓取页面
async def crawl_and_ask(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)  ## 页面下载器
        answer = ask_qwen(result.markdown) ## 结构分析器 result.markdown result.html result.text 等
        print("\nQwen 模型回答：\n", answer)

# 向 Qwen 模型发送请求
def ask_qwen(content):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个网页内容总结助手"},
            {"role": "user", "content": f"请用一段话中文总结以下网页内容：\n\n{content}"}
        ],
        "temperature": 0.7,
        "max_tokens": 32768
    }

    response = requests.post(QWEN_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"请求失败，状态码：{response.status_code}，响应内容：{response.text}"

# 主函数
async def main():
    url = "https://crawl4ai.com"
    await crawl_and_ask(url)

# 启动
asyncio.run(main())
