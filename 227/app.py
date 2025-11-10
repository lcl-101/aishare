# app.py (已修改以兼容旧版 Gradio 或解决版本冲突)

import gradio as gr
from PIL import Image
import rembg
import os
import numpy as np

# 设置本地模型环境变量
def setup_local_model_env():
    """设置环境变量以使用本地模型"""
    local_model_dir = os.path.abspath("models")
    if os.path.exists(local_model_dir):
        os.environ['U2NET_PATH'] = local_model_dir
        print(f"设置本地模型目录: {local_model_dir}")
        
        # 检查具体模型文件并设置对应的环境变量
        model_files = {
            'u2net.onnx': 'U2NET_PATH',
            'u2netp.onnx': 'U2NETP_PATH', 
            'isnet-general-use.onnx': 'ISNET_PATH'
        }
        
        for model_file, env_var in model_files.items():
            model_path = os.path.join(local_model_dir, model_file)
            if os.path.exists(model_path):
                os.environ[env_var] = local_model_dir
                print(f"发现本地模型: {model_file}")
    else:
        print(f"本地模型目录不存在: {local_model_dir}")

# 在导入rembg后立即设置环境变量
setup_local_model_env()

# --- 1. 模型会话缓存 (修改为支持本地模型) ---
SESSIONS = {}  # 清空缓存，重新加载
AVAILABLE_MODELS = ["u2net", "custom"]
LOCAL_U2NET_BASIC = None  # 基础抠图使用的本地U-2-Net模型

def get_local_model_path(model_name):
    """获取本地模型文件路径"""
    model_files = {
        "u2net": "models/u2net.onnx",
        "u2netp": "models/u2netp.onnx", 
        "isnet-general-use": "models/isnet-general-use.onnx",
        "isnet-anime": "models/isnet-anime.onnx",
        "custom": "models/custom.onnx"  # custom使用custom.onnx文件
    }
    return model_files.get(model_name)

def get_local_u2net_session():
    """获取本地U-2-Net ONNX模型会话"""
    global LOCAL_U2NET_BASIC
    if LOCAL_U2NET_BASIC is None:
        print(f"正在加载本地U-2-Net模型...")
        try:
            # 由于已设置环境变量，rembg会自动使用本地模型
            LOCAL_U2NET_BASIC = rembg.new_session("u2net")
            print(f"本地U-2-Net模型加载成功！")
        except Exception as e:
            print(f"加载本地模型失败: {e}")
            LOCAL_U2NET_BASIC = rembg.new_session("u2net")
    return LOCAL_U2NET_BASIC

def get_session(model_name):
    if model_name not in SESSIONS:
        print(f"正在加载模型: {model_name}...")
        
        # 检查本地模型文件
        local_path = get_local_model_path(model_name)
        print(f"模型 {model_name} 对应路径: {local_path}")
        
        if local_path and os.path.exists(local_path):
            print(f"✅ 检测到本地模型文件: {local_path}")
            
            # 对于 custom 模型，使用 u2net_custom 会话类型
            if model_name == "custom":
                print(f"🔥 使用 u2net_custom 会话类型加载: {local_path}")
                # 使用 u2net_custom 会话类型，并传递模型路径
                SESSIONS[model_name] = rembg.new_session("u2net_custom", model_path=local_path)
                print(f"🎯 Custom 模型会话创建完成!")
            else:
                # 其他模型使用标准方式加载（环境变量已设置）
                print(f"📦 使用标准方式加载模型: {model_name}")
                SESSIONS[model_name] = rembg.new_session(model_name)
        else:
            # 如果本地文件不存在，使用默认方式
            print(f"❌ 本地文件不存在: {local_path}")
            if model_name == "custom":
                print(f"⚠️ 自定义模型文件不存在: {local_path}，回退到 u2net")
                SESSIONS[model_name] = rembg.new_session("u2net")
            else:
                SESSIONS[model_name] = rembg.new_session(model_name)
        
        print(f"✅ 模型 '{model_name}' 加载成功！")
        
        # 检查会话对象的详细信息
        if model_name == "custom":
            session = SESSIONS[model_name]
            print(f"🔍 Custom 会话对象类型: {type(session)}")
            if hasattr(session, 'model_name'):
                print(f"🔍 Custom 会话模型名称: {session.model_name}")
            if hasattr(session, 'model_path'):
                print(f"🔍 Custom 会话模型路径: {session.model_path}")
    
    return SESSIONS[model_name]

# --- 2. 核心处理函数 (修改 sam_interactive) ---

def basic_remove_image(image):
    if image is None:
        gr.Warning("请先上传一张图片！")
        return None
    try:
        # 使用本地U-2-Net模型进行基础抠图
        session = get_local_u2net_session()
        return rembg.remove(image, session=session)
    except Exception as e:
        gr.Error(f"处理失败: {e}")
        return None

def advanced_remove_image(image, model_name, use_alpha_matting, bgcolor_hex, only_mask):
    if image is None:
        gr.Warning("请先上传一张图片！")
        return None
    
    # 如果只生成mask，忽略背景颜色和alpha matting设置
    if only_mask:
        try:
            session = get_session(model_name)
            # 使用 only_mask=True 参数
            mask_result = rembg.remove(
                image,
                session=session,
                only_mask=True
            )
            return mask_result
        except Exception as e:
            gr.Error(f"Mask 生成失败: {e}")
            return None
    
    # 原有的正常抠图逻辑
    bgcolor_rgba = None
    if bgcolor_hex and bgcolor_hex.strip():
        try:
            color_str = bgcolor_hex.strip()
            print(f"Color input: '{bgcolor_hex}'")
            
            # 处理 RGBA 格式：rgba(r, g, b, a)
            if color_str.startswith('rgba(') and color_str.endswith(')'):
                # 提取 rgba 值
                rgba_values = color_str[5:-1].split(',')  # 去掉 'rgba(' 和 ')'
                if len(rgba_values) == 4:
                    r = int(float(rgba_values[0].strip()))
                    g = int(float(rgba_values[1].strip()))
                    b = int(float(rgba_values[2].strip()))
                    # alpha 忽略，我们总是使用 255
                    bgcolor_rgba = (r, g, b, 255)
                else:
                    gr.Warning(f"无效的 RGBA 格式: '{bgcolor_hex}'")
                    return None
            # 处理 RGB 格式：rgb(r, g, b)
            elif color_str.startswith('rgb(') and color_str.endswith(')'):
                rgb_values = color_str[4:-1].split(',')  # 去掉 'rgb(' 和 ')'
                if len(rgb_values) == 3:
                    r = int(float(rgb_values[0].strip()))
                    g = int(float(rgb_values[1].strip()))
                    b = int(float(rgb_values[2].strip()))
                    bgcolor_rgba = (r, g, b, 255)
                else:
                    gr.Warning(f"无效的 RGB 格式: '{bgcolor_hex}'")
                    return None
            # 处理十六进制格式
            else:
                h = color_str.lstrip('#')
                if len(h) == 6 and all(c in '0123456789abcdefABCDEF' for c in h):
                    # 标准6位十六进制格式
                    bgcolor_rgba = tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (255,)
                elif len(h) == 3 and all(c in '0123456789abcdefABCDEF' for c in h):
                    # 3位十六进制格式，需要扩展
                    h = ''.join([c*2 for c in h])
                    bgcolor_rgba = tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (255,)
                else:
                    gr.Warning(f"无效的颜色格式: '{bgcolor_hex}'，请选择有效的颜色")
                    return None
                    
            print(f"Parsed color RGBA: {bgcolor_rgba}")
        except (ValueError, IndexError) as e:
            print(f"Color parsing error: {e}")
            gr.Warning(f"颜色解析错误: {str(e)}")
            return None
    try:
        session = get_session(model_name)
        result = rembg.remove(
            image,
            session=session,
            alpha_matting=use_alpha_matting,
            bgcolor=bgcolor_rgba
        )
        return result
    except Exception as e:
        gr.Error(f"处理失败: {e}")
        return None


# --- 3. Gradio 界面布局 ---

with gr.Blocks(theme=gr.themes.Soft(), title="Rembg WebUI") as demo:
    gr.Markdown("# 🖼️ Rembg 智能抠图工具箱")
    gr.Markdown("一个基于 `rembg` 库的强大WebUI，由AI驱动，轻松移除图片背景。")

    with gr.Tabs():
        # --- 基础抠图 ---
        with gr.TabItem("🚀 基础抠图"):
            gr.Markdown("**使用本地U-2-Net模型进行快速抠图**")
            with gr.Row():
                input_basic = gr.Image(type="pil", label="上传图片")
                output_basic = gr.Image(type="pil", label="抠图结果")
            btn_basic = gr.Button("一键抠图", variant="primary")
        
        # --- 高级选项 ---
        with gr.TabItem("🛠️ 高级选项"):
            with gr.Row():
                input_advanced = gr.Image(type="pil", label="上传图片")
                output_advanced = gr.Image(type="pil", label="处理结果")
            with gr.Row():
                model_select = gr.Dropdown(AVAILABLE_MODELS, value="custom", label="选择模型")
                alpha_matting_check = gr.Checkbox(label="启用 Alpha Matting (优化边缘细节)")
                bgcolor_picker = gr.ColorPicker(value="#FFFFFF", label="选择背景颜色 (留空则为透明)")
            with gr.Row():
                only_mask_check = gr.Checkbox(label="🎭 只生成 Mask (黑白遮罩)", value=False)
            btn_advanced = gr.Button("开始处理", variant="primary")

    # --- 事件绑定 ---
    btn_basic.click(fn=basic_remove_image, inputs=[input_basic], outputs=[output_basic])
    btn_advanced.click(fn=advanced_remove_image, inputs=[input_advanced, model_select, alpha_matting_check, bgcolor_picker, only_mask_check], outputs=[output_advanced])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")