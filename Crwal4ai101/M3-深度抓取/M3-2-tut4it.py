"""
包含以下步骤：
1. 使用 AsyncWebCrawler 抓取 tut4it.com 首页；
2. 使用 Qwen 2.5-72B-Instruct 模型识别首页的总页数；
3. 遍历每一页，抓取并提取课程标题与链接；
4. 每次请求间隔 60 秒，避免触发网站防护机制（如 Cloudflare）；
"""


import asyncio
import requests
from crawl4ai import AsyncWebCrawler
import time

# Qwen 模型部署配置（使用 vLLM 接口）
QWEN_API_URL = "http://10.112.113.1:8001/v1/chat/completions"
QWEN_MODEL_NAME = "/mnt/nvme0/Qwen2.5-72B-Instruct/"  # 本地模型路径

# 向 Qwen 模型发送 Markdown 内容，提取页面总数
def ask_qwen_page(markdown_text: str) -> str:
    headers = {"Content-Type": "application/json"}
    prompt = (
        "请从以下网页内容中提取站点的页面总数，以数字的形式返回，比如 1100，只要数字就行，其它的都不需要：\n\n"
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

# 向 Qwen 模型发送 Markdown 内容，提取结构信息（标题 + 链接）
def ask_qwen(markdown_text: str) -> str:
    headers = {"Content-Type": "application/json"}
    prompt = (
        "请从以下网页内容中提取所有课程标题和链接，并以清单格式输出（中英文均可）：\n\n"
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
async def crawl_and_extract_pagenum(url: str):
    print(f"🌐 开始抓取网页：{url}")
    async with AsyncWebCrawler(headless=True) as crawler:
        result = await crawler.arun(url)
        print("✅ 抓取完成，提取 Markdown 内容并发送给 Qwen 模型...\n")
        print(result.markdown)

        markdown = result.markdown
        answer = ask_qwen_page(markdown)
        page_num = int(answer.strip())
        return page_num
        

async def crawl_and_extract(page_num: int):

    async with AsyncWebCrawler(headless=True) as crawler:
        for i in range(1, page_num + 1):
            url = f"https://tut4it.com/page/{i}/"
            print(f"抓取第 {i} 页：{url}")
            result = await crawler.arun(url)
            print("✅ 抓取完成，提取 Markdown 内容并发送给 Qwen 模型...\n")

            markdown = result.markdown
            answer = ask_qwen(markdown)

            print(f"🤖 Qwen 模型返回第 {i} 页结果：\n")
            print(answer)
            time.sleep(60)  # 避免请求过快


# 程序入口
async def main():
    page_num = await crawl_and_extract_pagenum("https://tut4it.com")
    await crawl_and_extract(page_num)

if __name__ == "__main__":
    asyncio.run(main())
