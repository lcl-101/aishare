"""
1. 使用 AsyncWebCrawler 抓取网页并输出 Markdown 格式；
2. 将 Markdown 内容提交给 Qwen 2.5-72B-Instruct 模型；
3. 模型理解内容结构，并返回课程标题 + 链接。

这是一种“结构化抓取 + LLM 智能解析”的经典范式。
"""

import asyncio
import requests
from crawl4ai import AsyncWebCrawler

# Qwen 模型部署配置（使用 vLLM 接口）
QWEN_API_URL = "http://10.112.113.1:8001/v1/chat/completions"
QWEN_MODEL_NAME = "/mnt/nvme0/Qwen2.5-72B-Instruct/"  # 本地模型路径

# 向 Qwen 模型发送 Markdown 内容，提取结构信息（标题 + 链接）
def ask_qwen(markdown_text: str) -> str:
    headers = {"Content-Type": "application/json"}
    prompt = (
        "请从以下网页内容中提取所有课程标题和链接，链接以 'https://'开头，并以清单格式输出（中英文均可）：\n\n"
        f"{markdown_text}"
    )

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个网页结构信息提取助手"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 32768
    }

    try:
        response = requests.post(QWEN_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Qwen 请求失败：{response.status_code}\n{response.text}"
    except Exception as e:
        return f"⚠️ 调用 Qwen 出错：{e}"

# 使用 AsyncWebCrawler 抓取页面并调用 Qwen 模型处理
async def crawl_and_extract_with_qwen(url: str):
    print(f"🌐 开始抓取网页：{url}")
    async with AsyncWebCrawler(headless=False) as crawler:
        result = await crawler.arun(url)
        print("✅ 抓取完成，提取 Markdown 内容并发送给 Qwen 模型...\n")

        markdown = result.markdown
        answer = ask_qwen(markdown)

        print("🤖 Qwen 模型返回结果：\n")
        print(answer)

# 程序入口
async def main():
    await crawl_and_extract_with_qwen("https://tut4it.com")

if __name__ == "__main__":
    asyncio.run(main())
