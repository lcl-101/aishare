# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import subprocess
import gradio as gr
from phantom_wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES

# 示例提示文本
EXAMPLE_PROMPT = "暖阳漫过草地，扎着双马尾、头戴绿色蝴蝶结、身穿浅绿色连衣裙的小女孩蹲在盛开的雏菊旁。她身旁一只棕白相间的狗狗吐着舌头，毛茸茸尾巴欢快摇晃。小女孩笑着举起黄红配色、带有蓝色按钮的玩具相机，将和狗狗的欢乐瞬间定格。"

def run_generation(task, size, prompt, base_seed, ref_image_path="examples/ref1.png,examples/ref2.png"):
    """调用 generate.py 执行生成任务"""
    
    # 默认参数
    ckpt_dir = "./Wan2.1-T2V-1.3B"
    phantom_ckpt = "./Phantom-Wan-1.3B/Phantom-Wan-1.3B.pth"
    
    # 构建命令
    cmd = [
        "python", "generate.py",
        "--task", task,
        "--size", size,
        "--ckpt_dir", ckpt_dir,
        "--phantom_ckpt", phantom_ckpt,
        "--ref_image", ref_image_path,
        "--prompt", prompt,
        "--base_seed", str(int(base_seed) if base_seed else 42)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 使用subprocess运行命令
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # 查找生成的输出文件
        # 假设输出文件命名包含任务名称和尺寸
        prefix = f"{task}_{size}"
        files = [f for f in os.listdir('.') if f.endswith('.mp4') or f.endswith('.png')]
        matching_files = [f for f in files if f.startswith(prefix)]
        
        if matching_files:
            # 获取最近创建的文件
            latest_file = max(matching_files, key=lambda f: os.path.getctime(f))
            return latest_file, "生成成功!"
        else:
            return None, "生成成功，但未找到输出文件。请检查控制台输出。"
    
    except subprocess.CalledProcessError as e:
        print(f"命令执行错误: {e}")
        print(f"错误输出: {e.stderr}")
        return None, f"生成失败: {e}"
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None, f"发生错误: {str(e)}"

# 创建 Gradio 界面
with gr.Blocks(title="Phantom-Wan WebUI") as demo:
    gr.Markdown("# 🎬 Phantom-Wan 视频生成系统")
    
    # 先定义输出组件，解决引用顺序问题
    with gr.Row():
        output_file = gr.Video(label="生成结果")
    status_text = gr.Textbox(label="状态", interactive=False)
    
    with gr.Row():
        with gr.Column(scale=1):
            # 基本参数
            task = gr.Dropdown(
                choices=list(WAN_CONFIGS.keys()), 
                value="s2v-1.3B", 
                label="任务类型"
            )
            
            size = gr.Dropdown(
                choices=list(SIZE_CONFIGS.keys()), 
                value="832*480", 
                label="输出尺寸"
            )
            
            prompt = gr.Textbox(
                lines=5, 
                placeholder="请输入提示词...", 
                value=EXAMPLE_PROMPT,
                label="提示词"
            )
            
            base_seed = gr.Number(
                value=42, 
                label="随机种子", 
                precision=0
            )
            
            # 生成按钮 - 移动到 examples 控件之前
            gen_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            # 输出区域已在顶部定义，所以这里使用示例区域
            gr.Markdown("### 📋 参考图示例")
            
            # 添加示例功能，包含参考图片 - 移到右侧列
            with gr.Row(visible=False) as hidden_images:  # 创建隐藏的 Image 组件供 Examples 使用
                ref_img1 = gr.Image("examples/ref1.png", visible=False)
                ref_img2 = gr.Image("examples/ref2.png", visible=False)
            
            # 使用与输出相同的宽度显示示例
            with gr.Accordion("2张参考图示例", open=True):
                # 修改示例数据的结构，使每个参考图片都是单独的一项
                examples = gr.Examples(
                    examples=[
                        # 示例1: 2张参考图片
                        ["s2v-1.3B", "832*480", "暖阳漫过草地，扎着双马尾、头戴绿色蝴蝶结、身穿浅绿色连衣裙的小女孩蹲在盛开的雏菊旁。她身旁一只棕白相间的狗狗吐着舌头，毛茸茸尾巴欢快摇晃。小女孩笑着举起黄红配色、带有蓝色按钮的玩具相机，将和狗狗的欢乐瞬间定格。", 42, "examples/ref1.png", "examples/ref2.png"],
                        
                        # 示例2: 2张参考图片
                        ["s2v-1.3B", "832*480", "夕阳下，一位有着小麦色肌肤、留着乌黑长发的女人穿上有着大朵立体花朵装饰、肩袖处带有飘逸纱带的红色纱裙，漫步在金色的海滩上，海风轻拂她的长发，画面唯美动人。", 42, "examples/ref3.png", "examples/ref4.png"],
                    ],
                    inputs=[task, size, prompt, base_seed, ref_img1, ref_img2],
                    label="参考图: ref1 + ref2"
                )
            
            # 为3张参考图创建新的示例组
            with gr.Accordion("3张参考图示例", open=False):
                ref_img3 = gr.Image(visible=False)  # 添加第三个图片组件
                examples3 = gr.Examples(
                    examples=[
                        # 示例3: 3张参考图片
                        ["s2v-1.3B", "832*480", "在被冰雪覆盖，周围盛开着粉色花朵，有蝴蝶飞舞，屋内透出暖黄色灯光的梦幻小屋场景下，一位头发灰白、穿着深绿色上衣的老人牵着梳着双丸子头、身着中式传统服饰、外披白色毛绒衣物的小女孩的手，缓缓前行，画面温馨宁静。", 42, "examples/ref5.png", "examples/ref6.png", "examples/ref7.png"],
                    ],
                    inputs=[task, size, prompt, base_seed, ref_img1, ref_img2, ref_img3],
                    label="参考图: ref5 + ref6 + ref7",
                    fn=lambda t, s, p, b, r1, r2, r3: run_generation(t, s, p, b, f"{r1},{r2},{r3}"),
                    outputs=[output_file, status_text]
                )
            
            # 为4张参考图创建新的示例组
            with gr.Accordion("4张参考图示例", open=False):
                ref_img4 = gr.Image(visible=False)  # 添加第四个图片组件
                examples4 = gr.Examples(
                    examples=[
                        # 示例4: 4张参考图片
                        ["s2v-1.3B", "832*480", "一位金色长发的女人身穿棕色带波点网纱长袖、胸前系带设计的泳衣，手持一杯有橙色切片和草莓装饰、插着绿色吸管的分层鸡尾酒，坐在有着棕榈树、铺有蓝白条纹毯子和灰色垫子、摆放着躺椅的沙滩上晒日光浴的慢镜头，捕捉她享受阳光的微笑与海浪轻抚沙滩的美景。", 42, "examples/ref8.png", "examples/ref9.png", "examples/ref10.png", "examples/ref11.png"],
                    ],
                    inputs=[task, size, prompt, base_seed, ref_img1, ref_img2, ref_img3, ref_img4],
                    label="参考图: ref8 + ref9 + ref10 + ref11",
                    fn=lambda t, s, p, b, r1, r2, r3, r4: run_generation(t, s, p, b, f"{r1},{r2},{r3},{r4}"),
                    outputs=[output_file, status_text]
                )
    
    # 事件绑定
    gen_btn.click(
        fn=run_generation,
        inputs=[task, size, prompt, base_seed],
        outputs=[output_file, status_text]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
