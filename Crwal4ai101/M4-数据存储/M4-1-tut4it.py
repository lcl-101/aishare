'''
pip install mysql-connector-python

本脚本用于连接 MySQL 服务器并进行 tut4it 课程数据存储的表结构设计。

数据库设计说明：
----------------
数据库名：tut4it

1. courses 表（课程信息表）
   - id：主键，自增。
   - title：课程名称，唯一索引，防止重复。
   - url：课程主页链接，唯一索引。
   - description：课程描述（由 LLM 生成）。
   - has_detail：是否已抓取详情（布尔型），便于分步抓取。
   - created_at：入库时间。
   - updated_at：最后更新时间。
   用途：存储每门课程的基础信息，支持断点续抓、去重和增量更新。

2. course_downloads 表（下载链接表）
   - id：主键，自增。
   - course_id：外键，关联 courses.id。
   - download_url：下载链接（支持一门课程多个下载地址）。
   - created_at：入库时间。
   用途：存储每门课程的所有下载链接，实现一对多关联。

抓取流程建议：
----------------
1. 第一步：抓取课程标题和链接，入库 courses 表（只插入新课程，has_detail 设为 False）。
2. 第二步：读取 courses 表中 has_detail=False 的课程，抓取详情和下载链接，更新 description、has_detail，并将下载链接写入 course_downloads 表。
这样设计便于断点续抓、增量更新和数据一致性维护。
'''

import mysql.connector

def show_databases(conn):
    """查询并打印所有数据库名称"""
    cursor = conn.cursor()
    cursor.execute('SHOW DATABASES;')
    databases = cursor.fetchall()
    if databases:
        print('数据库列表:')
        for db in databases:
            print(db[0])
    else:
        print('没有找到任何数据库。')
    cursor.close()

def create_database_and_tables(conn):
    """创建 tut4it 数据库和所需表结构（如未存在）"""
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS tut4it DEFAULT CHARACTER SET utf8mb4;")
    conn.database = 'tut4it'
    # 创建 courses 表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS courses (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(512) NOT NULL,
        url VARCHAR(1024) NOT NULL,
        description TEXT,
        has_detail BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uniq_title (title(191)),
        UNIQUE KEY uniq_url (url(191))
    );
    ''')
    # 创建 course_downloads 表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS course_downloads (
        id INT AUTO_INCREMENT PRIMARY KEY,
        course_id INT NOT NULL,
        download_url VARCHAR(1024) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
    );
    ''')
    print('表结构已创建。')
    cursor.close()

# 连接到 MySQL 服务器
conn = mysql.connector.connect(
    host='172.30.8.246',
    port=3306,
    user='root',
    password='mysql.68.kaker'
)

# 查询数据库
show_databases(conn)

# 创建数据库和表结构
#create_database_and_tables(conn)

conn.close()
