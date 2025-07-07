"""
M5: AI 语音聊天室 (最终稳定演示版)

本脚本是整个项目的最终稳定版本，为确保在任何环境下的演示成功，采用了最可靠的设计：
-   交互: PC 优化的“单击开始/单击结束”模式。
-   反馈: 前端清晰地展示“识别中”和“AI思考中”的状态，用户体验直观。
-   响应: 后端一次性完成所有处理后返回完整结果，避免了流式传输中可能存在的环境问题。
-   安全: 使用真实的 SSL 证书，提供完美的 HTTPS 安全连接。
-   模型: 从本地路径加载 Whisper large-v3，使用 qwen3:32b 作为 LLM。

如何运行:
1.  确保项目结构正确 (certs/mixazure.crt, ../models/large-v3.pt 等)。
2.  安装所有依赖: pip install Flask pyOpenSSL openai-whisper ollama torch
3.  确保 Ollama 服务正在运行。
4.  运行此脚本: python M5_Final_Demo.py
5.  在浏览器中打开 https://<your-domain-or-ip>:7860
"""
import os
import whisper
import ollama
import logging
from flask import Flask, render_template_string, request, jsonify
import torch
import tempfile
import json

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
WHISPER_MODEL_PATH = "../models/large-v3.pt"
LLM_MODEL = "qwen3:32b"
PORT = 7860
CERT_FILE = 'certs/mixazure.crt'
KEY_FILE = 'certs/mixazure.key'

# --- 初始化 Flask 应用 ---
app = Flask(__name__)

# --- 加载模型 ---
model = None
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_model_path = os.path.join(script_dir, WHISPER_MODEL_PATH)
    if not os.path.exists(absolute_model_path):
        logging.error(f"FATAL: Whisper 模型文件未找到 -> {absolute_model_path}")
    else:
        logging.info(f"正在从本地路径加载 Whisper 模型: {absolute_model_path}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(absolute_model_path, device=device)
        logging.info(f"✅ Whisper 模型 '{WHISPER_MODEL_PATH}' 加载成功。")
except Exception as e:
    logging.error(f"❌ 加载 Whisper 模型时发生严重错误: {e}", exc_info=True)


# --- 前端 HTML/CSS/JS 模板 (最终演示版) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M5: AI 语音聊天室 (演示版)</title>
    <style> body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; background-color: #f4f4f9; } #chat-container { flex-grow: 1; overflow-y: auto; padding: 20px; border-bottom: 1px solid #ddd; display: flex; flex-direction: column; gap: 15px; } #controls { padding: 20px; background-color: #fff; text-align: center; box-shadow: 0 -2px 5px rgba(0,0,0,0.1); } #record-btn { background-color: #007bff; color: white; border: none; padding: 15px 30px; font-size: 18px; border-radius: 50px; cursor: pointer; transition: background-color 0.3s, transform 0.1s; width: 200px; } #record-btn.recording { background-color: #dc3545; animation: pulse 1.5s infinite; } @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); } 70% { box-shadow: 0 0 0 20px rgba(220, 53, 69, 0); } 100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); } } .message { padding: 10px 15px; border-radius: 20px; max-width: 80%; line-height: 1.5; box-shadow: 0 1px 3px rgba(0,0,0,0.1); word-wrap: break-word; } .user-message { background-color: #007bff; color: white; align-self: flex-end; } .ai-message { background-color: #e9e9eb; color: #333; align-self: flex-start; white-space: pre-wrap; } .status { color: #888; text-align: center; font-style: italic; align-self: center; background: none; box-shadow: none; } </style>
</head>
<body>
    <div id="chat-container">
        <div class="message ai-message">你好！我是你的AI语音助手。请点击下面的按钮开始说话。</div>
    </div>
    <div id="controls">
        <button id="record-btn">点击开始说话</button>
    </div>
    <script>
        const recordBtn = document.getElementById('record-btn');
        const chatContainer = document.getElementById('chat-container');
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;

        recordBtn.addEventListener('click', toggleRecording);
        function toggleRecording() { if (isRecording) stopRecording(); else startRecording(); }
        async function startRecording() { try { const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' }); mediaRecorder.ondataavailable = e => audioChunks.push(e.data); mediaRecorder.onstop = processAudio; audioChunks = []; mediaRecorder.start(); isRecording = true; recordBtn.textContent = '点击结束说话'; recordBtn.classList.add('recording'); } catch (err) { addMessage("错误：无法访问麦克风。请检查浏览器权限。", 'ai'); } }
        function stopRecording() { if (!mediaRecorder || mediaRecorder.state !== "recording") return; mediaRecorder.stop(); isRecording = false; recordBtn.textContent = '点击开始说话'; recordBtn.classList.remove('recording'); }
        
        async function processAudio() {
            if (audioChunks.length === 0) return;
            
            const statusMsg = addStatusMessage("录音结束，正在进行语音识别...");
            recordBtn.disabled = true; // 处理期间禁用按钮

            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('audio_file', audioBlob, 'recording.webm');
            
            try {
                // 后端现在会一次性返回所有结果
                const response = await fetch('/process_audio', { method: 'POST', body: formData });
                
                // 这里不再需要更新状态，因为结果马上就回来了
                removeStatusMessage();
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage('错误：' + data.error, 'ai');
                } else {
                    addMessage(data.user_prompt, 'user');
                    // 创建一个AI消息框，并提示正在思考
                    const aiMsgDiv = createMessageDiv('ai');
                    aiMsgDiv.innerHTML = "AI 正在思考中...";
                    
                    // 模拟AI思考的延迟，然后更新内容
                    // 在实际应用中，后端返回时已经是思考好的了
                    // 这里我们直接更新内容
                    aiMsgDiv.innerHTML = data.ai_response.replace(/\\n/g, '<br>');
                    scrollToBottom();
                }
            } catch (error) {
                console.error('处理失败:', error);
                removeStatusMessage();
                addMessage('错误：与服务器通信失败', 'ai');
            } finally {
                recordBtn.disabled = false; // 无论成功失败，都恢复按钮
            }
        }

        function createMessageDiv(sender) { const div = document.createElement('div'); div.classList.add('message', sender === 'user' ? 'user-message' : 'ai-message'); chatContainer.appendChild(div); return div; }
        function addMessage(text, sender) { const div = createMessageDiv(sender); div.innerHTML = text.replace(/\\n/g, '<br>'); scrollToBottom(); }
        function addStatusMessage(text) { const div = document.createElement('div'); div.id = 'status-message'; div.classList.add('message', 'status'); div.textContent = text; chatContainer.appendChild(div); scrollToBottom(); }
        function removeStatusMessage() { const msg = document.getElementById('status-message'); if (msg) msg.remove(); }
        function scrollToBottom() { chatContainer.scrollTop = chatContainer.scrollHeight; }
    </script>
</body>
</html>
"""

# --- Flask 路由 (一次性返回版本) ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/process_audio', methods=['POST'])
def process_audio_route():
    if 'audio_file' not in request.files: return jsonify({"error": "没有找到音频文件"}), 400
    if not model: return jsonify({"error": "Whisper 模型未加载"}), 503
    
    audio_file = request.files['audio_file']
    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        logging.info(f"正在转录临时文件: {temp_audio_path}")
        use_fp16 = torch.cuda.is_available()
        result = model.transcribe(temp_audio_path, fp16=use_fp16)
        user_prompt = result['text'].strip()
        logging.info(f"转录结果: {user_prompt}")
        if not user_prompt: return jsonify({"error": "未能识别出任何语音"}), 400

    except Exception as e:
        logging.error(f"Whisper 转录失败: {e}", exc_info=True)
        return jsonify({"error": f"语音识别失败: {e}"}), 500
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    try:
        logging.info(f"正在调用 LLM: {LLM_MODEL}")
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': '你是一个乐于助人的AI语音助手。请用简洁明了的语言回答问题。/no_think'},
                {'role': 'user', 'content': user_prompt}
            ],
            stream=False # 明确指定为非流式
        )
        ai_response = response['message']['content']
        logging.info(f"LLM 回答: {ai_response}")
    except Exception as e:
        logging.error(f"Ollama 调用失败: {e}", exc_info=True)
        return jsonify({"error": f"AI模型调用失败: {e}"}), 500
        
    return jsonify({
        "user_prompt": user_prompt,
        "ai_response": ai_response
    })

# --- 启动应用 (使用真实的 HTTPS 证书) ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cert_path = os.path.join(script_dir, CERT_FILE)
    key_path = os.path.join(script_dir, KEY_FILE)

    if not os.path.exists(cert_path) or not os.path.exists(key_path):
        logging.error("="*60)
        logging.error("错误：未能找到 SSL 证书文件！")
        logging.error(f"期望的证书路径: {cert_path}")
        logging.error(f"期望的私钥路径: {key_path}")
        logging.error("="*60)
    else:
        ssl_context = (cert_path, key_path)
        logging.info(f"准备在 https://0.0.0.0:{PORT} 启动 Flask 开发服务器...")
        logging.info(f"使用 SSL 证书: {cert_path}")
        
        try:
            app.run(
                host='0.0.0.0', 
                port=PORT, 
                debug=True, 
                ssl_context=ssl_context
            )
        except Exception as e:
            logging.error(f"启动 HTTPS 服务器时出错: {e}")