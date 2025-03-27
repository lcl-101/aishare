import os
import subprocess
import gradio as gr

def run_inference(lrc_file, style_type, ref_prompt=None, wav_file=None):
    """执行推理函数：根据用户选择的标签页类型选择不同的推理方式"""
    # 检查是否上传了lrc文件
    if not lrc_file or not os.path.exists(lrc_file):
        return None
    
    # 根据选择的风格类型，选择不同的命令
    if style_type == "wav" and wav_file and os.path.exists(wav_file):
        # 使用wav参考脚本
        cmd = f"""
        cd {os.getcwd()} &&
        export PYTHONPATH=$PYTHONPATH:$PWD &&
        export CUDA_VISIBLE_DEVICES=0 &&
        python3 infer/infer.py \\
            --lrc-path {lrc_file} \\
            --ref-audio-path {wav_file} \\
            --audio-length 285 \\
            --repo_id ASLP-lab/DiffRhythm-full \\
            --output-dir infer/example/output \\
            --chunked
        """
        print("使用WAV参考推理模式")
    else:
        # 使用文本提示参考脚本
        if not ref_prompt:
            ref_prompt = "classical genres, hopeful mood, piano."
        cmd = f"""
        cd {os.getcwd()} &&
        export PYTHONPATH=$PYTHONPATH:$PWD &&
        export CUDA_VISIBLE_DEVICES=0 &&
        python3 infer/infer.py \\
            --lrc-path {lrc_file} \\
            --ref-prompt "{ref_prompt}" \\
            --audio-length 285 \\
            --repo_id ASLP-lab/DiffRhythm-full \\
            --output-dir infer/example/output \\
            --chunked
        """
        print(f"使用文本提示推理模式，提示词: {ref_prompt}")
    
    print(f"执行命令: {cmd}")
    
    try:
        # 执行命令
        subprocess.run(cmd, shell=True, check=True)
        
        # 命令执行完成后，生成的音频文件在指定位置
        output_file = "infer/example/output/output.wav"
        if os.path.exists(output_file):
            return output_file
        else:
            print(f"警告：输出文件 {output_file} 不存在")
    except subprocess.CalledProcessError as e:
        print(f"执行命令时出错: {e}")
    
    return None

# 新增：读取lrc文件内容的函数
def read_lrc_content(lrc_file):
    """读取上传的lrc文件内容并返回"""
    if not lrc_file or not os.path.exists(lrc_file):
        return ""
    try:
        with open(lrc_file, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"读取lrc文件时出错: {e}")
        return "读取文件内容失败"

# 创建Gradio界面
with gr.Blocks(title="DiffRhythm WebUI") as demo:
    # 创建状态变量来跟踪活动标签页
    active_tab = gr.State(value="prompt")  # 默认使用文本提示模式
    
    gr.Markdown("# DiffRhythm 推理 WebUI")
    
    # 上传LRC文件控件（所有标签页共用）
    lrc_input = gr.File(
        label="上传LRC文件（必选）", 
        file_types=[".lrc", ".txt"], 
        type="filepath"
    )
    lrc_preview = gr.Textbox(
        label="LRC文件内容预览",
        lines=10,
        max_lines=20,
        interactive=False
    )
    
    # 将风格选择分为两个标签页
    with gr.Tabs() as tabs:
        with gr.TabItem("使用文本提示") as tab_prompt:
            # 设置当标签页被选中时更新状态
            tab_prompt.select(fn=lambda: "prompt", outputs=active_tab)
            
            # 风格提示文本输入控件
            ref_prompt_input = gr.Textbox(
                label="风格提示文本",
                value="classical genres, hopeful mood, piano.",
                lines=2
            )
        
        with gr.TabItem("使用WAV参考") as tab_wav:
            # 设置当标签页被选中时更新状态
            tab_wav.select(fn=lambda: "wav", outputs=active_tab)
            
            # 上传参考WAV文件控件
            wav_input = gr.Audio(
                label="上传参考WAV文件", 
                type="filepath"
            )
    
    # 右侧：生成结果控件
    result_audio = gr.Audio(
        label="生成结果", 
        type="filepath"
    )
    
    # 底部：推理按钮
    infer_btn = gr.Button("开始推理")
    
    # 绑定事件：lrc文件上传后自动显示内容
    lrc_input.change(
        fn=read_lrc_content,
        inputs=[lrc_input],
        outputs=[lrc_preview]
    )
    
    # 使用状态变量而非Tab组件作为输入
    infer_btn.click(
        fn=run_inference,
        inputs=[
            lrc_input,           # lrc文件路径
            active_tab,          # 当前活动标签（状态变量）
            ref_prompt_input,    # 文本提示
            wav_input            # WAV文件路径
        ],
        outputs=[result_audio]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
