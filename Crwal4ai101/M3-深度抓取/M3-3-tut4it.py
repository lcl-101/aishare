'''
处理流程：
1. 通过 AsyncWebCrawler 下载 tut4it 首页，调用 Qwen 模型获取总页数。
2. 依次抓取每一页课程列表，调用 Qwen 模型提取课程标题和链接。
3. 对每个课程详情页，调用 Qwen 模型提取课程中文描述和下载链接。
4. 结果可用于展示、保存或后续分析。
'''

import asyncio
import requests
import time
from crawl4ai import AsyncWebCrawler

# Qwen 模型配置
QWEN_API_URL = "http://10.112.113.1:8001/v1/chat/completions"
QWEN_MODEL_NAME = "/mnt/nvme0/Qwen2.5-72B-Instruct/"

# 用于获取总页数
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

    response = requests.post(QWEN_API_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# 用于从列表页提取课程标题和链接
def ask_qwen_list(markdown_text: str) -> str:
    headers = {"Content-Type": "application/json"}
    prompt = (
        "请从以下网页内容中提取所有课程标题和链接，"
        "只返回纯文本标题和对应的链接，不要包含任何多余符号，如 '链接:', '[', ']', '(', ')' 等，"
        "格式为每行一条记录，标题和链接用逗号分隔，例如：\n"
        "The Ultimate Guide to Unity, https://tut4it.com/some-course\n\n"
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

    response = requests.post(QWEN_API_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# 从课程详情页提取中文描述 + 下载链接
def ask_qwen_course_detail(markdown_text: str) -> str:
    headers = {"Content-Type": "application/json"}
    prompt = (
        "请你作为中文语言助手，从以下网页内容中提取课程的中文简要描述（约50-100字），"
        "描述应包括该课程教授的内容、技术方向或适合人群。"
        "同时，提取该页面中所有下载链接（如百度网盘、Google Drive、OneDrive、MediaFire 等）。\n\n"
        "输出格式如下（请严格使用中文）：\n"
        "课程描述：xxxxx（用中文概括）\n"
        "下载链接：\n- 链接1\n- 链接2\n\n"
        "以下是网页内容：\n\n"
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

    response = requests.post(QWEN_API_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# 将模型结果转换为列表
def parse_qwen_answer_to_list(answer: str) -> list[tuple[str, str]]:
    lines = answer.strip().split("\n")
    result = []
    for line in lines:
        if "," in line:
            title, url = line.split(",", 1)
            if url.strip().startswith("http"):
                result.append((title.strip(), url.strip()))
    return result

# 异步获取总页数
async def crawl_and_extract_pagenum(url: str) -> int:
    async with AsyncWebCrawler(headless=True) as crawler:
        result = await crawler.arun(url)
        markdown = result.markdown
        answer = ask_qwen_page(markdown)
        return int(answer.strip())

# 异步抓取所有课程并提取详情
async def crawl_and_extract_all(page_num: int):
    async with AsyncWebCrawler(headless=True) as crawler:
        for i in range(1, page_num + 1):
            url = f"https://tut4it.com/page/{i}/"
            print(f"\n📄 抓取列表页：{url}")
            result = await crawler.arun(url)
            markdown = result.markdown
            list_answer = ask_qwen_list(markdown)
            courses = parse_qwen_answer_to_list(list_answer)

            print(f"📦 第 {i} 页发现课程数：{len(courses)}")
            for title, course_url in courses:
                print(f"\n🔗 打开课程：《{title}》\n{course_url}")
                try:
                    detail_result = await crawler.arun(course_url)
                    detail_markdown = detail_result.markdown
                    detail_answer = ask_qwen_course_detail(detail_markdown)
                    print(detail_answer)
                except Exception as e:
                    print(f"⚠️ 抓取失败：{e}")

                time.sleep(60)  # 每门课程间隔，避免过快

            time.sleep(60)  # 每页间隔

# 主函数
async def main():
    page_num = await crawl_and_extract_pagenum("https://tut4it.com")
    print(f"🌐 总页数：{page_num}")
    await crawl_and_extract_all(page_num)

if __name__ == "__main__":
    asyncio.run(main())
