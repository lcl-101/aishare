'''
简单的 Flask Web 页面，展示 tut4it.courses 表中的课程数据。

依赖：
    pip install flask
运行：
    python M5-1-tut4it.py
    # 然后在浏览器访问 http://localhost:5000/
'''

from flask import Flask, render_template_string, request
import math
import mysql.connector

app = Flask(__name__)

def get_courses(page=1, page_size=20):
    conn = mysql.connector.connect(
        host='172.30.8.246',
        port=3306,
        user='root',
        password='mysql.68.kaker',
        database='tut4it',
        charset='utf8mb4'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM courses")
    total = cursor.fetchone()[0]
    offset = (page - 1) * page_size
    cursor.execute("SELECT id, title, url, description, has_detail, created_at FROM courses ORDER BY id DESC LIMIT %s OFFSET %s", (page_size, offset))
    courses = cursor.fetchall()
    # 查询所有下载链接
    course_ids = [row[0] for row in courses]
    downloads = {}
    if course_ids:
        format_strings = ','.join(['%s'] * len(course_ids))
        cursor.execute(f"SELECT course_id, download_url FROM course_downloads WHERE course_id IN ({format_strings})", tuple(course_ids))
        for cid, durl in cursor.fetchall():
            downloads.setdefault(cid, []).append(durl)
    cursor.close()
    conn.close()
    total_pages = math.ceil(total / page_size)
    return courses, downloads, total, total_pages

@app.route('/')
def index():
    page = int(request.args.get('page', 1))
    courses, downloads, total, total_pages = get_courses(page=page, page_size=20)
    html = '''
    <html>
    <head>
        <meta charset="utf-8">
        <title>tut4it 课程数据展示</title>
        <style>
            table { border-collapse: collapse; width: 95%; margin: 20px auto; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            th { background: #f5f5f5; }
            .desc { max-width: 400px; word-break: break-all; }
            .downloads { max-width: 300px; word-break: break-all; }
            .pagination { text-align: center; margin: 20px; }
            .pagination a { margin: 0 5px; text-decoration: none; color: #007bff; }
            .pagination span { margin: 0 5px; color: #333; }
        </style>
    </head>
    <body>
        <h2 style="text-align:center;">tut4it 课程数据展示</h2>
        <table>
            <tr>
                <th>ID</th><th>标题</th><th>链接</th><th>描述</th><th>下载链接</th><th>已抓详情</th><th>入库时间</th>
            </tr>
            {% for row in courses %}
            <tr>
                <td>{{ row[0] }}</td>
                <td>{{ row[1] }}</td>
                <td><a href="{{ row[2] }}" target="_blank">课程链接</a></td>
                <td class="desc">{{ row[3] or '' }}</td>
                <td class="downloads">
                    {% for d in downloads.get(row[0], []) %}
                        <a href="{{ d }}" target="_blank">下载</a><br/>
                    {% endfor %}
                </td>
                <td>{{ '是' if row[4] else '否' }}</td>
                <td>{{ row[5] }}</td>
            </tr>
            {% endfor %}
        </table>
        <div class="pagination">
            {% if page > 1 %}
                <a href="/?page=1">首页</a>
                <a href="/?page={{ page-1 }}">上一页</a>
            {% endif %}
            <span>第 {{ page }} / {{ total_pages }} 页</span>
            {% if page < total_pages %}
                <a href="/?page={{ page+1 }}">下一页</a>
                <a href="/?page={{ total_pages }}">末页</a>
            {% endif %}
        </div>
    </body>
    </html>
    '''
    return render_template_string(html, courses=courses, downloads=downloads, page=page, total_pages=total_pages)

if __name__ == '__main__':
    app.run(debug=True)
