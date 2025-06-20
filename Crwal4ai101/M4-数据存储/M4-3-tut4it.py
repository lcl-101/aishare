'''
本脚本用于补全 tut4it.courses 表中未抓取详情的课程：
- 读取 has_detail=False 的课程 url
- 抓取详情页，提取描述和下载链接
- 更新 courses 表的 description、has_detail 字段
- 下载链接写入 course_downloads 表
'''

import asyncio
import requests
import time
import mysql.connector
from crawl4ai import AsyncWebCrawler

QWEN_API_URL = "http://10.112.113.1:8001/v1/chat/completions"
QWEN_MODEL_NAME = "/mnt/nvme0/Qwen2.5-72B-Instruct/"

# Qwen 提取课程描述和下载链接
def ask_qwen_detail(markdown_text: str) -> tuple[str, list[str]]:
    headers = {"Content-Type": "application/json"}
    prompt = (
        "请根据以下课程网页内容，提取两个信息：\n"
        "1. 用中文总结课程内容，50-100字；\n"
        "2. 课程中提到的所有下载链接（如有，支持多个）。\n"
        "返回格式：\n"
        "课程总结：...\n下载地址：...（如有多个请用换行分隔）\n\n"
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
            desc = ""
            downloads = []
            for line in answer.strip().split("\n"):
                if line.startswith("课程总结"):
                    desc = line.split("：", 1)[-1].strip()
                elif line.startswith("下载地址"):
                    # 兼容单行或多行
                    rest = line.split("：", 1)[-1].strip()
                    if rest:
                        downloads.extend([x.strip() for x in rest.split() if x.strip().startswith("http")])
            # 兼容多行下载链接
            for line in answer.strip().split("\n"):
                if line.strip().startswith("http"):
                    downloads.append(line.strip())
            return desc, list(set(downloads))
        else:
            print(f"❌ Qwen 请求失败：{response.status_code}\n{response.text}")
            return "", []
    except Exception as e:
        print(f"⚠️ 调用 Qwen 出错：{e}")
        return "", []

def get_pending_courses(conn, limit=10):
    cursor = conn.cursor()
    cursor.execute("SELECT id, url FROM courses WHERE has_detail=FALSE ORDER BY id ASC LIMIT %s", (limit,))
    rows = cursor.fetchall()
    cursor.close()
    return rows

def update_course_detail(conn, course_id, desc):
    cursor = conn.cursor()
    cursor.execute("UPDATE courses SET description=%s, has_detail=TRUE, updated_at=NOW() WHERE id=%s", (desc, course_id))
    conn.commit()
    cursor.close()

def insert_download_links(conn, course_id, links):
    cursor = conn.cursor()
    for url in links:
        cursor.execute("INSERT IGNORE INTO course_downloads (course_id, download_url) VALUES (%s, %s)", (course_id, url))
    conn.commit()
    cursor.close()

async def main():
    conn = mysql.connector.connect(
        host='172.30.8.246',
        port=3306,
        user='root',
        password='mysql.68.kaker',
        database='tut4it',
        charset='utf8mb4'
    )
    pending = get_pending_courses(conn, limit=10)
    if not pending:
        print("没有需要补全详情的课程。")
        conn.close()
        return
    async with AsyncWebCrawler(headless=True) as crawler:
        for course_id, url in pending:
            print(f"抓取详情: {url}")
            try:
                result = await crawler.arun(url)
                desc, downloads = ask_qwen_detail(result.markdown)
                update_course_detail(conn, course_id, desc)
                if downloads:
                    insert_download_links(conn, course_id, downloads)
                print(f"已补全: {url}")
            except Exception as e:
                print(f"❌ 失败: {url}，原因: {e}")
            time.sleep(60)  # 每个课程详情抓取后暂停一分钟
    conn.close()
    print("补全完成！")

if __name__ == "__main__":
    asyncio.run(main())
