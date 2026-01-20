"""
FunctionGemma 工具调用演示 - Gradio Web 应用
基于 Google FunctionGemma 模型的函数调用功能
"""

import os
import re
import asyncio
import subprocess
import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from openai import OpenAI
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

# ================== 配置 ==================
MODEL_PATH = "checkpoints/functiongemma-270m-it"

# LLM API 配置
OPENAI_API_KEY = "sk-4e3e00a0b4522d6d4c119ce2ddeb1722"
API_URL = "https://api.xxx.com/v1"
MODEL_NAME = "sykjtestuqwen2-5-72b-instruct"
AI_MAX_TOKENS = 32768
AI_TEMPERATURE = 0.2
AI_TIMEOUT = 120  # 增加超时时间以支持长内容总结

# ================== 加载模型 ==================
print("正在加载 FunctionGemma 模型...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print(f"模型已加载到: {model.device}")

# ================== 初始化 OpenAI 客户端 ==================
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_URL,
    timeout=AI_TIMEOUT
)

# ================== 工具定义 ==================

# 工具1: 获取显卡配置
gpu_info_schema = {
    "type": "function",
    "function": {
        "name": "get_gpu_info",
        "description": "Gets the current GPU configuration and status using nvidia-smi command. Returns detailed information about NVIDIA GPUs including memory usage, temperature, and utilization.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
}

# 工具2: 网页抓取
web_crawler_schema = {
    "type": "function",
    "function": {
        "name": "crawl_webpage",
        "description": "Crawls and extracts content from a given URL. Returns the webpage content in markdown format.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to crawl, e.g. https://example.com",
                },
            },
            "required": ["url"],
        },
    }
}

# 工具3: 获取当前日期时间
date_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_datetime",
        "description": "Gets the current date, time, and day of week. Use this when user asks about today's date, current time, what day it is, or any time-related questions.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
}

ALL_TOOLS = [gpu_info_schema, web_crawler_schema, date_time_schema]

# ================== 工具实现 ==================

def execute_get_gpu_info():
    """执行 nvidia-smi 命令获取显卡信息"""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"错误: {result.stderr}"
    except FileNotFoundError:
        return "错误: 未找到 nvidia-smi 命令，可能未安装 NVIDIA 驱动"
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时"
    except Exception as e:
        return f"错误: {str(e)}"

def execute_get_datetime():
    """获取当前日期和时间"""
    from datetime import datetime
    import locale
    
    now = datetime.now()
    
    # 星期几的中文映射
    weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    weekday_cn = weekdays[now.weekday()]
    
    result = f"""当前日期时间信息：
- 日期: {now.strftime('%Y年%m月%d日')}
- 时间: {now.strftime('%H:%M:%S')}
- 星期: {weekday_cn}
- ISO格式: {now.isoformat()}
- 时间戳: {int(now.timestamp())}"""
    
    return result

async def execute_crawl_webpage(url: str):
    """使用 crawl4ai 抓取网页内容"""
    try:
        browser_config = BrowserConfig()
        run_config = CrawlerRunConfig()
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            return result.markdown if result.markdown else "未能提取到网页内容"
    except Exception as e:
        return f"抓取错误: {str(e)}"

def summarize_with_llm(content: str, max_length: int = 5000):
    """使用 LLM 总结内容"""
    # 截断过长的内容
    if len(content) > max_length:
        content = content[:max_length] + "\n\n[内容已截断...]"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的内容分析师。请用中文对以下网页内容进行简洁、有条理的总结，突出关键信息和要点。回复请控制在 500 字以内。"},
                {"role": "user", "content": f"请总结以下网页内容：\n\n{content}"}
            ],
            max_tokens=1024,
            temperature=AI_TEMPERATURE
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM 总结错误: {str(e)}"

# ================== 解析函数调用 ==================

def parse_function_call(output: str):
    """解析 FunctionGemma 的函数调用输出"""
    # 匹配格式: <start_function_call>call:function_name{params}<end_function_call>
    pattern = r'<start_function_call>call:(\w+)\{([^}]*)\}<end_function_call>'
    match = re.search(pattern, output)
    
    if match:
        func_name = match.group(1)
        params_str = match.group(2)
        
        # 解析参数
        params = {}
        if params_str:
            # 解析格式: key:<escape>value<escape>
            param_pattern = r'(\w+):<escape>([^<]*)<escape>'
            param_matches = re.findall(param_pattern, params_str)
            for key, value in param_matches:
                params[key] = value
        
        return func_name, params
    
    # 尝试匹配无参数的调用
    pattern_no_params = r'<start_function_call>call:(\w+)\{\}<end_function_call>'
    match = re.search(pattern_no_params, output)
    if match:
        return match.group(1), {}
    
    return None, None

# ================== FunctionGemma 推理 ==================

def generate_function_call(user_query: str, tools: list):
    """使用 FunctionGemma 生成函数调用"""
    message = [
        {
            "role": "developer",
            "content": "You are a model that can do function calling with the following functions"
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    inputs = processor.apply_chat_template(
        message,
        tools=tools,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    # 将输入移到设备上
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    # 清除 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=False,  # 使用贪婪解码避免 CUDA 采样问题
        )
    
    # 解码输出，保留特殊token以便解析函数调用
    generated_ids = out[0][len(input_ids[0]):]
    output = processor.decode(generated_ids, skip_special_tokens=False)
    
    # 清理输出中的 pad token
    output = output.replace("<pad>", "").strip()
    
    return output

# ================== 主处理函数 ==================

def process_gpu_query(user_query: str):
    """处理显卡查询请求"""
    if not user_query.strip():
        return "请输入您的问题", "", ""
    
    # 步骤1: 使用 FunctionGemma 生成函数调用
    raw_output = generate_function_call(user_query, [gpu_info_schema])
    
    # 步骤2: 解析函数调用
    func_name, params = parse_function_call(raw_output)
    
    if func_name == "get_gpu_info":
        # 步骤3: 执行工具
        tool_result = execute_get_gpu_info()
        
        tool_call_info = f"""📞 **工具调用信息**
- 识别到的函数: `{func_name}`
- 参数: `{params if params else '无参数'}`
- 原始输出: `{raw_output.strip()}`"""
        
        return tool_call_info, raw_output.strip(), tool_result
    else:
        return f"未能识别函数调用\n原始输出: {raw_output}", raw_output.strip(), "未执行工具"

async def process_web_query_async(user_query: str, url: str):
    """处理网页抓取请求（异步）"""
    if not user_query.strip():
        return "请输入您的问题", "", "", ""
    
    if not url.strip():
        return "请输入要抓取的网址", "", "", ""
    
    # 步骤1: 使用 FunctionGemma 生成函数调用
    query_with_url = f"{user_query} The URL is: {url}"
    raw_output = generate_function_call(query_with_url, [web_crawler_schema])
    
    # 步骤2: 解析函数调用
    func_name, params = parse_function_call(raw_output)
    
    if func_name == "crawl_webpage":
        # 使用用户输入的 URL（如果模型没有正确解析）
        target_url = params.get("url", url)
        if not target_url or target_url == url:
            target_url = url
        
        # 步骤3: 执行网页抓取
        crawl_result = await execute_crawl_webpage(target_url)
        
        # 步骤4: 使用 LLM 总结
        summary = summarize_with_llm(crawl_result)
        
        tool_call_info = f"""📞 **工具调用信息**
- 识别到的函数: `{func_name}`
- 目标网址: `{target_url}`
- 原始输出: `{raw_output.strip()}`"""
        
        # 截断显示的原始内容
        display_content = crawl_result[:3000] + "..." if len(crawl_result) > 3000 else crawl_result
        
        return tool_call_info, raw_output.strip(), display_content, summary
    else:
        return f"未能识别函数调用\n原始输出: {raw_output}", raw_output.strip(), "未执行抓取", "无法生成总结"

def process_web_query(user_query: str, url: str):
    """处理网页抓取请求（同步包装）"""
    return asyncio.run(process_web_query_async(user_query, url))

# ================== 智能聊天处理 ==================

def extract_url_from_text(text: str):
    """从文本中提取 URL"""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    match = re.search(url_pattern, text)
    return match.group(0) if match else None

def should_use_tools(user_message: str):
    """
    预判断用户消息是否可能需要调用工具
    使用关键词匹配来避免不必要的 FunctionGemma 调用
    返回: (需要工具, 推测的工具类型)
    """
    msg_lower = user_message.lower()
    
    # GPU 相关关键词
    gpu_keywords = ['gpu', 'cuda', 'nvidia', '显卡', '显存', '显示卡', 'graphics', '图形卡', 
                    'nvidia-smi', '算力', 'vram', '显卡配置', '显卡信息', '显卡状态']
    
    # 网页抓取相关关键词
    web_keywords = ['http://', 'https://', 'www.', '网页', '抓取', 'crawl', 'fetch', 
                    '网站', 'url', '链接', 'webpage', 'website']
    
    # 日期时间相关关键词
    datetime_keywords = ['日期', '时间', '今天', '现在', '几号', '几点', '星期', '周几', 
                         'date', 'time', 'today', 'now', 'what day', 'current time', '几日']
    
    for kw in gpu_keywords:
        if kw in msg_lower:
            return True, "gpu"
    
    for kw in web_keywords:
        if kw in msg_lower:
            return True, "web"
    
    for kw in datetime_keywords:
        if kw in msg_lower:
            return True, "datetime"
    
    return False, None

def chat_with_tools(user_message: str, history: list):
    """
    智能聊天：自动识别是否需要调用工具
    先用关键词预判断，再使用 FunctionGemma 决定调用哪个工具
    history 使用 Gradio 6.x messages 格式: [{"role": "user/assistant", "content": "..."}, ...]
    """
    if not user_message.strip():
        return history, ""
    
    # 添加用户消息到历史
    history = list(history) + [{"role": "user", "content": user_message}]
    
    # 预判断是否需要调用工具
    needs_tools, tool_hint = should_use_tools(user_message)
    
    tool_info = ""
    func_name = None
    params = {}
    raw_output = ""
    
    if needs_tools:
        # 使用 FunctionGemma 判断调用哪个工具
        raw_output = generate_function_call(user_message, ALL_TOOLS)
        func_name, params = parse_function_call(raw_output)
        
        # 备用方案：如果 FunctionGemma 没有正确识别，根据关键词提示直接调用
        if func_name is None and tool_hint:
            if tool_hint == "gpu":
                func_name = "get_gpu_info"
            elif tool_hint == "web":
                func_name = "crawl_webpage"
            elif tool_hint == "datetime":
                func_name = "get_current_datetime"
            raw_output = f"[关键词匹配备用] -> {func_name}"
    
    if func_name == "get_gpu_info":
        # 调用 GPU 信息工具
        tool_info = f"🔧 **FunctionGemma 决定调用工具**: `get_gpu_info`\n\n原始输出: `{raw_output}`"
        gpu_result = execute_get_gpu_info()
        
        # 使用 LLM 生成友好的回复
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个专业的技术助手。用户询问了 GPU 信息，你已经通过调用 nvidia-smi 命令获取了显卡信息。请根据获取到的信息，用友好、专业的中文回答用户的问题。"},
                    {"role": "user", "content": f"用户问题: {user_message}\n\nnvidia-smi 输出结果:\n{gpu_result}\n\n请根据以上信息回答用户。"}
                ],
                max_tokens=1024,
                temperature=AI_TEMPERATURE
            )
            assistant_reply = response.choices[0].message.content
        except Exception as e:
            assistant_reply = f"获取到的 GPU 信息:\n```\n{gpu_result}\n```\n\n(LLM 回复生成失败: {str(e)})"
        
        # 添加工具调用标记
        full_reply = f"🔧 *[已调用工具: get_gpu_info]*\n\n{assistant_reply}"
        history.append({"role": "assistant", "content": full_reply})
        
    elif func_name == "crawl_webpage":
        # 调用网页抓取工具
        url = params.get("url") or extract_url_from_text(user_message)
        
        if not url:
            history.append({"role": "assistant", "content": "❌ 抱歉，我需要一个有效的 URL 才能抓取网页内容。请在消息中包含完整的网址（如 https://example.com）"})
            return history, ""
        
        tool_info = f"🔧 **FunctionGemma 决定调用工具**: `crawl_webpage`\n\n目标 URL: `{url}`\n\n原始输出: `{raw_output}`"
        
        # 异步抓取网页
        try:
            crawl_result = asyncio.run(execute_crawl_webpage(url))
            
            # 使用 LLM 总结内容
            summary = summarize_with_llm(crawl_result)
            
            full_reply = f"🔧 *[已调用工具: crawl_webpage]*\n\n📄 **网页内容总结** ({url}):\n\n{summary}"
            history.append({"role": "assistant", "content": full_reply})
            
        except Exception as e:
            history.append({"role": "assistant", "content": f"❌ 抓取网页时出错: {str(e)}"})
            return history, ""
    
    elif func_name == "get_current_datetime":
        # 调用日期时间工具
        tool_info = f"🔧 **FunctionGemma 决定调用工具**: `get_current_datetime`\n\n原始输出: `{raw_output}`"
        datetime_result = execute_get_datetime()
        
        # 使用 LLM 生成友好的回复
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个专业的助手。用户询问了日期时间信息，你已经获取了当前的日期和时间。请根据获取到的信息，用友好的中文回答用户的问题。"},
                    {"role": "user", "content": f"用户问题: {user_message}\n\n获取到的日期时间信息:\n{datetime_result}\n\n请根据以上信息回答用户。"}
                ],
                max_tokens=512,
                temperature=AI_TEMPERATURE
            )
            assistant_reply = response.choices[0].message.content
        except Exception as e:
            assistant_reply = f"获取到的日期时间信息:\n{datetime_result}\n\n(LLM 回复生成失败: {str(e)})"
        
        # 添加工具调用标记
        full_reply = f"🔧 *[已调用工具: get_current_datetime]*\n\n{assistant_reply}"
        history.append({"role": "assistant", "content": full_reply})
    
    else:
        # 没有需要调用的工具，直接使用 LLM 回复
        if needs_tools and tool_hint:
            tool_info = f"ℹ️ **关键词检测到可能需要工具，但未成功调用**\n\n原始输出: `{raw_output}`\n\n将直接使用 LLM 回答..."
        else:
            tool_info = "ℹ️ **未检测到工具调用关键词**\n\n直接使用 LLM 回答..."
        
        try:
            # 构建历史消息
            messages = [
                {"role": "system", "content": """你是一个专业的 AI 助手。请用中文回答用户的问题。

你可以帮助用户完成以下任务：
1. 查询 GPU/显卡信息 - 用户可以问"我的显卡配置是什么？"或"GPU 使用情况怎样？"
2. 抓取和总结网页内容 - 用户可以说"帮我总结 https://example.com 的内容"
3. 获取当前日期时间 - 用户可以问"今天几号？"或"现在几点？"

如果用户的问题与这些功能相关，可以引导他们使用这些功能。"""}
            ]
            
            # 添加历史对话 (messages 格式)
            for h in history[:-1]:
                role = h.get("role", "user")
                content = h.get("content", "") or ""
                if content and isinstance(content, str):
                    # 移除工具调用标记用于上下文
                    clean_content = re.sub(r'🔧 \*\[已调用工具: \w+\]\*\n\n', '', content)
                    messages.append({"role": role, "content": clean_content})
            
            messages.append({"role": "user", "content": user_message})
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=2048,
                temperature=AI_TEMPERATURE
            )
            history.append({"role": "assistant", "content": response.choices[0].message.content})
        except Exception as e:
            history.append({"role": "assistant", "content": f"❌ LLM 回复生成失败: {str(e)}"})
    
    return history, tool_info

# ================== Gradio 界面 ==================

# 自定义 CSS
custom_css = """
.youtube-banner {
    background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
    color: white;
    padding: 15px 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
.youtube-banner a {
    color: white;
    text-decoration: none;
    font-weight: bold;
}
.youtube-banner a:hover {
    text-decoration: underline;
}
.tool-output {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 5px;
    font-family: monospace;
}
"""

with gr.Blocks() as demo:
    
    # YouTube 频道信息横幅
    gr.HTML("""
    <div class="youtube-banner">
        <h2>🎬 欢迎访问我的 YouTube 频道</h2>
        <p>频道名称：<strong>AI 技术分享频道</strong></p>
        <p><a href="https://www.youtube.com/@rongyikanshijie-ai" target="_blank">
            👉 点击订阅：https://www.youtube.com/@rongyikanshijie-ai
        </a></p>
    </div>
    """)
    
    gr.Markdown("""
    # 🤖 FunctionGemma 工具调用演示
    
    本应用演示了如何使用 Google 的 FunctionGemma 模型进行函数调用（Function Calling）。
    模型会根据用户的自然语言输入，自动识别需要调用的工具，并执行相应操作。
    
    ---
    """)
    
    with gr.Tabs():
        # ============ Tab 1: 显卡信息查询 ============
        with gr.TabItem("🖥️ 显卡配置查询"):
            gr.Markdown("""
            ### 功能说明
            输入关于显卡/GPU的问题，模型会自动调用 `nvidia-smi` 命令获取显卡配置信息。
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gpu_input = gr.Textbox(
                        label="输入您的问题",
                        placeholder="What is my GPU configuration?",
                        lines=2,
                        value="What is my GPU configuration?"
                    )
                    gpu_examples = gr.Examples(
                        examples=[
                            ["What is my GPU configuration?"],
                            ["Show me the current GPU status"],
                            ["Tell me about the graphics card on this machine"],
                            ["What NVIDIA GPU do I have?"],
                        ],
                        inputs=gpu_input,
                        label="示例提示词"
                    )
                    gpu_btn = gr.Button("🔍 查询显卡信息", variant="primary")
                
                with gr.Column(scale=2):
                    gpu_tool_info = gr.Markdown(label="工具调用信息")
                    gpu_raw_output = gr.Textbox(
                        label="FunctionGemma 原始输出",
                        lines=2,
                        interactive=False
                    )
                    gpu_result = gr.Textbox(
                        label="工具执行结果 (nvidia-smi 输出)",
                        lines=15,
                        interactive=False
                    )
            
            gpu_btn.click(
                fn=process_gpu_query,
                inputs=[gpu_input],
                outputs=[gpu_tool_info, gpu_raw_output, gpu_result]
            )
        
        # ============ Tab 2: 网页抓取与总结 ============
        with gr.TabItem("🌐 网页抓取与总结"):
            gr.Markdown("""
            ### 功能说明
            输入问题和目标网址，模型会自动调用网页抓取工具获取内容，然后使用 LLM 进行智能总结。
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    web_input = gr.Textbox(
                        label="输入您的问题",
                        placeholder="Please crawl this webpage and summarize the content",
                        lines=2,
                        value="Please crawl this webpage and summarize the content"
                    )
                    url_input = gr.Textbox(
                        label="目标网址",
                        placeholder="https://example.com",
                        lines=1,
                        value="https://example.com"
                    )
                    web_examples = gr.Examples(
                        examples=[
                            ["Please crawl this webpage and summarize the content", "https://example.com"],
                            ["Get the content from this URL", "https://news.ycombinator.com"],
                            ["Fetch and analyze this webpage", "https://github.com"],
                        ],
                        inputs=[web_input, url_input],
                        label="示例提示词"
                    )
                    web_btn = gr.Button("🔍 抓取并总结", variant="primary")
                
                with gr.Column(scale=2):
                    web_tool_info = gr.Markdown(label="工具调用信息")
                    web_raw_output = gr.Textbox(
                        label="FunctionGemma 原始输出",
                        lines=2,
                        interactive=False
                    )
                    web_content = gr.Textbox(
                        label="抓取的网页内容（部分）",
                        lines=8,
                        interactive=False
                    )
                    web_summary = gr.Markdown(label="📝 LLM 智能总结")
            
            web_btn.click(
                fn=process_web_query,
                inputs=[web_input, url_input],
                outputs=[web_tool_info, web_raw_output, web_content, web_summary]
            )
        
        # ============ Tab 3: 智能聊天 ============
        with gr.TabItem("💬 智能聊天"):
            gr.Markdown("""
            ### 功能说明
            在这里与 AI 自然对话。当你询问 GPU 信息、日期时间或需要抓取网页内容时，系统会自动调用相应工具！
            
            **试试这样问：**
            - "我的显卡配置是什么？"
            - "帮我总结一下 https://example.com 的内容"
            - "今天是几号？现在几点了？"
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="对话记录",
                        height=450
                    )
                    
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="输入消息",
                            placeholder="输入您的问题，例如：我的显卡是什么型号？或者：帮我总结 https://example.com",
                            lines=2,
                            scale=4
                        )
                        chat_btn = gr.Button("发送 💬", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ 清空对话")
                        
                    chat_examples = gr.Examples(
                        examples=[
                            ["你好，你是谁啊？"],
                            ["我的显卡配置是什么？"],
                            ["帮我抓取并总结 https://cj.sina.com.cn/articles/view/2290787940/888aa66401901gt2w 的内容"],
                            ["今天是几号？"],
                        ],
                        inputs=chat_input,
                        label="示例对话"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🔧 工具调用日志")
                    tool_log = gr.Markdown(
                        value="*等待用户输入...*",
                        label="FunctionGemma 工具调用日志"
                    )
                    
                    gr.Markdown("""
                    ---
                    ### 可用工具
                    
                    | 工具 | 功能 |
                    |------|------|
                    | `get_gpu_info` | 获取 GPU 配置信息 |
                    | `crawl_webpage` | 抓取网页内容 |
                    | `get_current_datetime` | 获取当前日期时间 |
                    
                    ---
                    **工作流程:**
                    1. 用户输入问题
                    2. FunctionGemma 分析是否需要调用工具
                    3. 如需调用，执行工具获取结果
                    4. LLM 根据工具结果生成回复
                    """)
            
            # 绑定事件
            chat_btn.click(
                fn=chat_with_tools,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, tool_log]
            ).then(
                fn=lambda: "",
                outputs=[chat_input]
            )
            
            chat_input.submit(
                fn=chat_with_tools,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, tool_log]
            ).then(
                fn=lambda: "",
                outputs=[chat_input]
            )
            
            clear_btn.click(
                fn=lambda: ([], "*等待用户输入...*"),
                outputs=[chatbot, tool_log]
            )
    
    gr.Markdown("""
    ---
    ### 📌 技术说明
    
    - **FunctionGemma**: Google 开发的专门用于函数调用的轻量级模型
    - **工具定义**: 使用 JSON Schema 格式定义工具接口
    - **网页抓取**: 基于 crawl4ai 库实现
    - **内容总结**: 使用 Qwen 2.5 72B 大语言模型
    
    ---
    <p style="text-align: center; color: gray;">
        Made with ❤️ | FunctionGemma 工具调用演示
    </p>
    """)

# ================== 启动应用 ==================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=custom_css
    )
