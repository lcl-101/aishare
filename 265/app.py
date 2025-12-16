import os
import torch
import gradio as gr
from functools import partial
from torchvision.utils import save_image
from diffusers import QwenImagePipeline
import numpy as np
from PIL import Image

from diffusers_patch.modeling_qwen_image import QwenImage
from unified_sampler import UnifiedSampler

# 初始化设备和模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/TwinFlow/TwinFlow-Qwen-Image-v1.0/TwinFlow-Qwen-Image/"

# 全局变量存储模型
model = None

def load_model():
    """加载模型"""
    global model
    if model is None:
        print("正在加载模型...")
        model = QwenImage(model_path, aux_time_embed=True, device=device)
        print("模型加载完成！")
    return model

def generate_image(
    prompt,
    height,
    width,
    seed,
    sampling_steps,
    stochast_ratio,
    extrapol_ratio,
    rfba_gap_start,
    rfba_gap_end
):
    """
    生成图像的主函数
    """
    try:
        # 加载模型
        model = load_model()
        
        # 设置随机种子
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        torch.manual_seed(seed)
        
        # 配置采样器
        sampler_config = {
            "sampling_steps": sampling_steps,
            "stochast_ratio": stochast_ratio,
            "extrapol_ratio": extrapol_ratio,
            "sampling_order": 1,
            "time_dist_ctrl": [1.0, 1.0, 1.0],
            "rfba_gap_steps": [rfba_gap_start, rfba_gap_end],
        }
        
        sampler = partial(UnifiedSampler().sampling_loop, **sampler_config)
        
        # 生成图像
        print(f"开始生成图像，提示词: {prompt[:50]}...")
        demox = model.sample(
            [prompt],
            cfg_scale=0.0,  # should be zero
            seed=seed,
            height=height,
            width=width,
            sampler=sampler,
            return_traj=False,
        )
        
        # 处理输出
        demox = demox.squeeze(0)  # [C, H, W]
        
        # 转换为 float32（模型输出是 bfloat16，numpy 不支持）
        demox = demox.float()
        
        # 转换为 PIL 图像
        image_tensor = (demox + 1) / 2  # 归一化到 [0, 1]
        image_tensor = torch.clamp(image_tensor, 0, 1)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        image_np = (image_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        
        print("图像生成完成！")
        return image_pil, f"生成成功！使用的种子: {seed}"
        
    except Exception as e:
        print(f"生成图像时出错: {str(e)}")
        return None, f"错误: {str(e)}"

# 示例提示词
examples = [
    [
        '一张逼真的年轻东亚女性肖像，位于画面中心偏左的位置，带着浅浅的微笑直视观者。她身着以浓郁的红色和金色为主的传统中式服装。她的头发被精心盘起，饰有精致的红色和金色花卉和叶形发饰。她的眉心之间额头上绘有一个小巧、华丽的红色花卉图案。她左手持一把仿古扇子，扇面上绘有一位身着传统服饰的女性、一棵树和一只鸟的场景。她的右手向前伸出，手掌向上，托着一个悬浮的发光的霓虹黄色灯牌，上面写着"TwinFlow So Fast"，这是画面中最亮的元素。背景是模糊的夜景，带有暖色调的人工灯光，一场户外文化活动或庆典。在远处的背景中，她头部的左侧略偏，是一座高大、多层、被暖光照亮的西安大雁塔。中景可见其他模糊的建筑和灯光，暗示着一个繁华的城市或文化背景。光线是低调的，灯牌为她的脸部和手部提供了显著的照明。整体氛围神秘而迷人。人物的头部、手部和上半身完全可见，下半身被画面底部边缘截断。图像具有中等景深，主体清晰聚焦，背景柔和模糊。色彩方案温暖，以红色、金色和闪电的亮黄色为主。',
        1024, 768, 42, 2, 1.0, 0.0, 0.001, 0.6
    ],
    [
        "一只可爱的橘色小猫坐在窗台上，阳光透过窗户洒在它身上，背景是模糊的城市景观。小猫眼睛明亮，毛发蓬松，充满好奇地看着窗外。画面温馨，光线柔和，细节丰富。",
        768, 768, 123, 4, 1.0, 0.0, 0.001, 0.5
    ],
    [
        "一幅宁静的山水画，远处是连绵的雪山，中景是碧绿的湖泊，近景是盛开的樱花树。天空中有几朵白云，阳光明媚。整体色彩鲜明，构图优美，充满诗意。",
        1024, 1024, 456, 4, 1.0, 0.0, 0.001, 0.5
    ],
    [
        "一座未来主义风格的城市，高耸的摩天大楼，霓虹灯闪烁，飞行汽车在空中穿梭。街道上人来人往，充满科技感和赛博朋克氛围。夜晚，光影交错，色彩绚丽。",
        768, 1024, 789, 2, 1.0, 0.0, 0.001, 0.6
    ],
    [
        "一个精致的咖啡杯，装满拉花艺术的卡布奇诺，放在木质桌面上。旁边有一本打开的书和一副眼镜。窗外的阳光洒进来，营造出温暖舒适的氛围。特写镜头，景深效果明显。",
        768, 768, 321, 4, 1.0, 0.0, 0.001, 0.5
    ],
]

# 创建 Gradio 界面
with gr.Blocks(title="TwinFlow 图像生成") as demo:
    gr.Markdown(
        """
        # 🎨 TwinFlow 图像生成系统
        
        基于 TwinFlow-Qwen-Image 模型的快速图像生成系统。支持 2-4 NFE（神经网络函数评估）的快速生成。
        
        ## 使用说明：
        1. 输入详细的中文或英文提示词
        2. 调整图像尺寸和采样参数
        3. 点击"生成图像"按钮
        4. 也可以直接选择下方的示例快速开始
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="提示词 (Prompt)",
                placeholder="请输入详细的图像描述...",
                lines=6,
                value=examples[0][0]
            )
            
            with gr.Row():
                height_slider = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    step=64,
                    value=1024,
                    label="高度 (Height)"
                )
                width_slider = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    step=64,
                    value=768,
                    label="宽度 (Width)"
                )
            
            with gr.Row():
                seed_input = gr.Number(
                    label="随机种子 (Seed)",
                    value=42,
                    precision=0,
                    info="-1 表示随机种子"
                )
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=2,
                    label="采样步数 (Sampling Steps)",
                    info="推荐 2 或 4 步"
                )
            
            with gr.Accordion("高级设置", open=False):
                stochast_ratio = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="随机比例 (Stochastic Ratio)"
                )
                extrapol_ratio = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.0,
                    label="外推比例 (Extrapolation Ratio)"
                )
                with gr.Row():
                    rfba_gap_start = gr.Number(
                        label="RFBA Gap Start",
                        value=0.001,
                        precision=3
                    )
                    rfba_gap_end = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.6,
                        label="RFBA Gap End"
                    )
            
            generate_btn = gr.Button("🚀 生成图像", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="生成的图像", type="pil", height=600)
            output_info = gr.Textbox(label="状态信息", lines=2)
    
    # 示例区域
    gr.Markdown("## 📚 示例提示词")
    gr.Examples(
        examples=examples,
        inputs=[
            prompt_input,
            height_slider,
            width_slider,
            seed_input,
            steps_slider,
            stochast_ratio,
            extrapol_ratio,
            rfba_gap_start,
            rfba_gap_end
        ],
        outputs=[output_image, output_info],
        fn=generate_image,
        cache_examples=False,
        label="点击示例快速生成"
    )
    
    # 绑定生成按钮
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            height_slider,
            width_slider,
            seed_input,
            steps_slider,
            stochast_ratio,
            extrapol_ratio,
            rfba_gap_start,
            rfba_gap_end
        ],
        outputs=[output_image, output_info]
    )
    
    gr.Markdown(
        """
        ---
        ### 💡 提示：
        - **2 NFE 配置**: 最快速度，适合快速预览（推荐 RFBA Gap End = 0.6）
        - **4 NFE 配置**: 更高质量，适合最终输出（推荐 RFBA Gap End = 0.5）
        - 提示词越详细，生成的图像质量越好
        - 建议图像尺寸为 64 的倍数
        """
    )

# 启动应用
if __name__ == "__main__":
    print("正在启动 TwinFlow Web 应用...")
    print("正在加载模型，请稍候...")
    
    # 预先加载模型
    load_model()
    
    print("模型加载完成！启动 Web 界面...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )
