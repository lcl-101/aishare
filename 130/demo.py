from docling.document_converter import DocumentConverter
import requests
import json

# Qwen 模型部署配置（使用 vLLM 接口）
QWEN_API_URL = "http://10.112.113.1:8001/v1/chat/completions"
QWEN_MODEL_NAME = "/mnt/nvme0/Qwen2.5-72B-Instruct/"  # 本地模型路径

source = "demo.pdf"  # PDF path or URL
converter = DocumentConverter()
result = converter.convert(source)
markdown_text = result.document.export_to_markdown()
print(markdown_text)

def ask_llm_with_context(question: str, context: str) -> str:
    """
    使用提供的上下文向 Qwen LLM 提问。
    """
    system_prompt = "你是一个智能问答助手。请根据下面提供的上下文信息来回答用户的问题。如果上下文中没有相关信息，请明确告知用户你不知道答案，不要编造。"
    user_prompt = f"""
请参考以下上下文信息：
---
{context}
---
\n根据以上信息，回答问题：{question}
"""
    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 32768,
    }
    headers = {
        "Content-Type": "application/json",
    }
    print("正在向 LLM 发送请求...")
    try:
        response = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_data = response.json()
        if "choices" in response_data and len(response_data["choices"]) > 0:
            assistant_message = response_data["choices"][0].get("message", {}).get("content", "")
            return assistant_message.strip()
        else:
            print("LLM API 响应格式不正确或 choices 为空:", response_data)
            return "抱歉，无法从LLM获取有效回答。"
    except requests.exceptions.RequestException as e:
        print(f"请求 LLM API 时发生错误: {e}")
        return f"抱歉，与LLM通信时出现错误: {e}"
    except json.JSONDecodeError:
        print(f"解析 LLM API 响应时发生错误。响应内容: {response.text}")
        return "抱歉，解析LLM响应时出现错误。"

if __name__ == "__main__":
    question = "请将表格当中的流程描述转换为自然语言，并且流程节点之间使用-->连接。"
    print(f"问题: {question}\n")
    answer = ask_llm_with_context(question, markdown_text)
    print(f"LLM 回答: {answer}")