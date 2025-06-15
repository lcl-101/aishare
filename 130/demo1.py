import base64
import requests
from PIL import Image

# Qwen-VL 模型配置
QWEN_VL_API_URL = "http://10.112.113.2:8000/v1/chat/completions"
QWEN_VL_MODEL_NAME = "/model_files/Qwen2.5-VL-72B-Instruct/"

# 图像路径
IMAGE_PATH = "demo.jpeg"

def encode_image_to_base64(image_path: str) -> str:
    """
    将图像编码为 base64 字符串（用于 Qwen-VL multimodal 输入）
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        return base64.b64encode(image_bytes).decode("utf-8")

def ask_qwen_vl_with_image(question: str, image_b64: str) -> str:
    """
    向 Qwen-VL 提问，并上传图像
    """
    system_prompt = "你是一个视觉语言助手，请根据图片内容准确提取流程。"
    user_prompt = f"""
请识别图像中的流程图或表格，并将其转换为自然语言流程，流程节点之间使用 --> 连接。
例如：
开始 --> 审核 --> 批准 --> 结束
问题：{question}
"""

    # 构造图文混合 content
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        {"type": "text", "text": user_prompt}
    ]

    payload = {
        "model": QWEN_VL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    headers = {"Content-Type": "application/json"}
    print("正在向 Qwen-VL 发送请求...")

    try:
        response = requests.post(QWEN_VL_API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return "未能获取有效回答。"
    except Exception as e:
        return f"请求失败: {e}"

if __name__ == "__main__":
    question = "请将图中的流程转换为自然语言，并使用 --> 表示节点之间连接。"
    print(f"问题: {question}\n")
    image_b64 = encode_image_to_base64(IMAGE_PATH)
    answer = ask_qwen_vl_with_image(question, image_b64)
    print(f"Qwen-VL 回答:\n{answer}")
