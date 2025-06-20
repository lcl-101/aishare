'''
本脚本用于抓取 tut4it.com 课程标题和链接，并将其存入 MySQL 数据库（tut4it.courses）。
详细内容抓取将在 M4-3-tut4it.py 实现。
'''

import asyncio
import requests
import time
import mysql.connector
from crawl4ai import AsyncWebCrawler

QWEN_API_URL = "http://10.112.113.1:8001/v1/chat/completions"
QWEN_MODEL_NAME = "/mnt/nvme0/Qwen2.5-72B-Instruct/"

# Qwen 提取课程标题和链接
def ask_qwen_list(markdown_text: str) -> list[tuple[str, str]]:
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
    try:
        response = requests.post(QWEN_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            # 解析为 (title, url) 列表
            result = []
            for line in answer.strip().split("\n"):
                if "," in line:
                    title, url = line.split(",", 1)
                    if url.strip().startswith("http"):
                        result.append((title.strip(), url.strip()))
            return result
        else:
            print(f"❌ Qwen 请求失败：{response.status_code}\n{response.text}")
            return []
    except Exception as e:
        print(f"⚠️ 调用 Qwen 出错：{e}")
        return []

# 获取总页数
async def get_page_count():
    async with AsyncWebCrawler(headless=True) as crawler:
        result = await crawler.arun("https://tut4it.com")
        markdown = result.markdown
        headers = {"Content-Type": "application/json"}
        prompt = (
            "请从以下网页内容中提取站点的页面总数，以数字的形式返回，比如 1100，只要数字就行，其它的都不需要：\n\n"
            f"{markdown}"
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
        answer = response.json()["choices"][0]["message"]["content"]
        return int(answer.strip())

# 入库函数
def insert_course(conn, title, url):
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT IGNORE INTO courses (title, url, has_detail)
            VALUES (%s, %s, FALSE)
            """,
            (title, url)
        )
        conn.commit()
    except Exception as e:
        print(f"插入失败: {title} {url}，原因: {e}")
    cursor.close()

def query_courses(conn, limit=20):
    """查询 courses 表，默认显示前 20 条记录"""
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, url, has_detail, created_at FROM courses ORDER BY id DESC LIMIT %s", (limit,))
    rows = cursor.fetchall()
    print(f"\n最近 {limit} 条课程记录：")
    for row in rows:
        print(f"ID: {row[0]} | 标题: {row[1]} | 链接: {row[2]} | 已抓详情: {row[3]} | 入库时间: {row[4]}")
    cursor.close()

# 主抓取流程
async def main():
    # 连接数据库
    conn = mysql.connector.connect(
        host='172.30.8.246',
        port=3306,
        user='root',
        password='mysql.68.kaker',
        database='tut4it',
        charset='utf8mb4'
    )
    page_count = await get_page_count()
    print(f"共 {page_count} 页")
    async with AsyncWebCrawler(headless=True) as crawler:
        for i in range(1, page_count + 1):
            url = f"https://tut4it.com/page/{i}/"
            print(f"抓取第 {i} 页: {url}")
            result = await crawler.arun(url)
            courses = ask_qwen_list(result.markdown)
            print(f"第 {i} 页课程数: {len(courses)}")
            for title, link in courses:
                insert_course(conn, title, link)
            time.sleep(60)
    conn.close()
    print("抓取并入库完成！")
    # 查询部分课程，查看效果
    conn = mysql.connector.connect(
        host='172.30.8.246',
        port=3306,
        user='root',
        password='softice.68.kaker',
        database='tut4it',
        charset='utf8mb4'
    )
    query_courses(conn)
    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
