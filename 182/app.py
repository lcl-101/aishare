#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

# 首先设置环境变量，必须在导入其他模块之前
JOBS_DIR = Path(__file__).parent.absolute()
MODEL_BASE = str(JOBS_DIR / "checkpoints" / "Hunyuan-GameCraft-1.0" / "stdmodels")
os.environ["MODEL_BASE"] = MODEL_BASE
os.environ["PYTHONPATH"] = f"{JOBS_DIR}:{os.environ.get('PYTHONPATH', '')}"
print(f"🔧 预设置MODEL_BASE环境变量: {MODEL_BASE}")
print(f"🔧 预设置PYTHONPATH环境变量: {os.environ['PYTHONPATH']}")

# 现在才导入其他模块
import gradio as gr
import threading
import time
import tempfile
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import random

# 添加项目路径到Python路径
sys.path.insert(0, str(JOBS_DIR))

# 导入项目模块
from hymm_sp.config import parse_args
from hymm_sp.sample_inference import HunyuanVideoSampler
from hymm_sp.data_kits.data_tools import save_videos_grid


class CropResize:
    """
    Custom transform to resize and crop images to a target size while preserving aspect ratio.
    """
    def __init__(self, size=(704, 1216)):
        self.target_h, self.target_w = size  

    def __call__(self, img):
        w, h = img.size
        scale = max(self.target_w / w, self.target_h / h)
        new_size = (int(h * scale), int(w * scale))
        resize_transform = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BILINEAR)
        resized_img = resize_transform(img)
        crop_transform = transforms.CenterCrop((self.target_h, self.target_w))
        return crop_transform(resized_img)


class HunyuanGameCraftWebUI:
    def __init__(self):
        self.output_dir = "./results"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型路径配置
        self.checkpoint_dir = JOBS_DIR / "checkpoints" / "Hunyuan-GameCraft-1.0"
        self.model_base = str(self.checkpoint_dir / "stdmodels")
        
        # 模型配置选项
        self.model_configs = {
            "标准模型": {
                "checkpoint_path": str(self.checkpoint_dir / "gamecraft_models" / "mp_rank_00_model_states.pt"),
                "infer_steps": 30,
                "cfg_scale": 2.0,
                "use_fp8": False,
                "description": "标准质量模型，推理较慢但质量较高"
            },
            "加速模型": {
                "checkpoint_path": str(self.checkpoint_dir / "gamecraft_models" / "mp_rank_00_model_states_distill.pt"),
                "infer_steps": 8,
                "cfg_scale": 1.0,
                "use_fp8": True,
                "description": "蒸馏加速模型，推理快速但质量略低"
            }
        }
        
        # 默认使用标准模型
        self.current_model = "标准模型"
        self.checkpoint_path = self.model_configs[self.current_model]["checkpoint_path"]
        
        print(f"📁 模型路径: {self.model_base}")
        
        # 检查模型文件
        self.model_available = self.check_model_files()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化模型为None，延迟加载
        self.hunyuan_video_sampler = None
        self.model_loaded = False
        
        # 动作列表选项
        self.action_options = {
            "w": "向前移动 (W)",
            "s": "向后移动 (S)", 
            "a": "向左移动 (A)",
            "d": "向右移动 (D)",
            "wa": "向前+向左 (W+A)",
            "wd": "向前+向右 (W+D)",
            "sa": "向后+向左 (S+A)",
            "sd": "向后+向右 (S+D)"
        }
        
        # 反向映射：从显示文本到动作代码
        self.action_reverse_map = {v: k for k, v in self.action_options.items()}
        
        # 预设的prompt模板
        self.prompt_templates = {
            "中世纪村庄": "A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky.",
            "自定义": ""
        }

    def check_model_files(self):
        """检查模型文件是否存在"""
        model_base_exists = os.path.exists(self.model_base)
        
        print(f"模型检查:")
        print(f"  - 基础模型目录: {'✅' if model_base_exists else '❌'} {self.model_base}")
        
        # 检查所有模型文件
        all_models_available = True
        for model_name, config in self.model_configs.items():
            checkpoint_exists = os.path.exists(config["checkpoint_path"])
            print(f"  - {model_name}: {'✅' if checkpoint_exists else '❌'} {config['checkpoint_path']}")
            if not checkpoint_exists:
                all_models_available = False
        
        return model_base_exists and all_models_available

    def switch_model(self, model_name):
        """切换模型配置"""
        if model_name in self.model_configs:
            self.current_model = model_name
            config = self.model_configs[model_name]
            self.checkpoint_path = config["checkpoint_path"]
            
            # 如果已经加载了模型，需要重新加载
            if self.model_loaded:
                self.model_loaded = False
                self.hunyuan_video_sampler = None
                print(f"🔄 切换到{model_name}，模型将在下次生成时重新加载")
            
            # 返回推荐的参数设置
            return (
                config["infer_steps"],
                config["cfg_scale"],
                f"✅ 已切换到{model_name} - {config['description']}"
            )
        else:
            return None, None, f"❌ 未知的模型: {model_name}"

    def load_model(self, progress_callback=None):
        """加载模型"""
        if self.model_loaded:
            return True
            
        try:
            if progress_callback:
                progress_callback(0.1, "准备加载模型...")
            
            # 设置单GPU分布式环境变量
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29605"
            
            # 创建默认参数
            args = self.create_default_args()
            print(f"✅ 参数创建成功，latent_channels: {args.latent_channels}")
            
            if progress_callback:
                progress_callback(0.3, "初始化分布式环境...")
            
            # 初始化分布式环境（单GPU模式）
            from hymm_sp.modules.parallel_states import initialize_distributed
            print("🔄 初始化分布式环境...")
            initialize_distributed(args.seed)
            print("✅ 分布式环境初始化完成")
            
            if progress_callback:
                progress_callback(0.5, "加载Hunyuan视频生成器...")
            
            # 加载模型
            self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
                self.checkpoint_path, 
                args=args, 
                device=self.device
            )
            
            if progress_callback:
                progress_callback(0.9, "模型加载完成...")
            
            self.model_loaded = True
            print("✅ 模型加载成功!")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_default_args(self):
        """创建默认参数"""
        # 使用parse_args创建默认参数，但不从命令行解析
        import argparse
        parser = argparse.ArgumentParser()
        from hymm_sp.config import add_extra_args, sanity_check_args
        parser = add_extra_args(parser)
        
        # 获取当前模型配置
        config = self.model_configs[self.current_model]
        
        # 创建默认参数，包含所有必要的模型参数
        base_args = [
            "--ckpt", self.checkpoint_path,
            "--seed", "250160",
            "--infer-steps", str(config["infer_steps"]),
            "--cfg-scale", str(config["cfg_scale"]),
            "--flow-shift-eval-video", "5.0",
            "--video-size", "704", "1216",
            "--sample-n-frames", "33",
            "--model", "HYVideo-T/2",
            "--vae", "884-16c-hy0801",
            "--text-encoder", "llava-llama-3-8b",
            "--text-encoder-2", "clipL",
            "--tokenizer", "llava-llama-3-8b",
            "--tokenizer-2", "clipL",
            "--rope-theta", "256",
            "--precision", "bf16",
            "--reproduce",
            "--use-deepcache", "1",  # 启用DeepCache加速
            "--use-sage"  # 启用SAGE注意力优化
        ]
        
        # 如果是加速模型，添加FP8优化
        if config["use_fp8"]:
            base_args.append("--use-fp8")
        
        args = parser.parse_args(base_args)
        
        # 执行参数检查，这会自动设置latent_channels
        args = sanity_check_args(args)
        
        return args

    def validate_inputs(self, image, prompt, action_list, action_speeds):
        """验证输入参数"""
        if image is None:
            return False, "请上传一张参考图片"
        
        if not prompt.strip():
            return False, "请输入描述提示词"
            
        if not action_list:
            return False, "请至少选择一个动作"
            
        if len(action_list) != len(action_speeds):
            return False, "动作列表和速度列表长度必须一致"
            
        for speed in action_speeds:
            if not (0 <= speed <= 3):
                return False, "动作速度必须在0-3之间"
                
        return True, "验证通过"

    def generate_video(self, image, prompt_template, custom_prompt, add_pos_prompt, add_neg_prompt,
                      action_list_display, action_speeds_text, video_width, video_height, 
                      cfg_scale, seed, infer_steps, flow_shift, progress=gr.Progress()):
        """生成视频的主函数"""
        
        try:
            # 确定最终的prompt
            if prompt_template == "自定义":
                final_prompt = custom_prompt
            else:
                final_prompt = self.prompt_templates.get(prompt_template, custom_prompt)
            
            # 转换动作列表
            action_list = [self.action_reverse_map.get(action, action) for action in action_list_display]
            
            # 解析动作速度
            try:
                action_speeds = [float(x.strip()) for x in action_speeds_text.split()]
            except:
                return None, "❌ 动作速度格式错误", ""
            
            # 验证输入
            is_valid, msg = self.validate_inputs(image, final_prompt, action_list, action_speeds)
            if not is_valid:
                return None, f"❌ {msg}", ""
            
            # 加载模型
            progress(0.1, desc="检查模型状态...")
            if not self.model_loaded:
                if not self.load_model(lambda p, desc: progress(p * 0.4, desc=desc)):
                    return None, "❌ 模型加载失败", ""
            
            progress(0.5, desc="准备输入数据...")
            
            # 保存临时图片
            temp_image_path = os.path.join(tempfile.gettempdir(), f"input_image_{int(time.time())}.png")
            image.save(temp_image_path)
            
            # 准备图像变换
            closest_size = (video_height, video_width)
            ref_image_transform = transforms.Compose([
                CropResize(closest_size),
                transforms.CenterCrop(closest_size),
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])
            ])
            
            # 处理参考图像
            raw_ref_images = [image.convert('RGB')]
            ref_images_pixel_values = [ref_image_transform(ref_image) for ref_image in raw_ref_images]
            ref_images_pixel_values = torch.cat(ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(self.device)
            
            progress(0.6, desc="编码参考图像...")
            
            # 编码参考图像
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                self.hunyuan_video_sampler.pipeline.vae.enable_tiling()
                
                raw_last_latents = self.hunyuan_video_sampler.vae.encode(
                    ref_images_pixel_values
                ).latent_dist.sample().to(dtype=torch.float16)
                raw_last_latents.mul_(self.hunyuan_video_sampler.vae.config.scaling_factor)
                raw_ref_latents = raw_last_latents.clone()
                
                self.hunyuan_video_sampler.pipeline.vae.disable_tiling()
            
            # 生成视频
            ref_images = raw_ref_images
            last_latents = raw_last_latents
            ref_latents = raw_ref_latents
            
            progress(0.7, desc="开始生成视频...")
            
            out_cat = None
            for idx, action_id in enumerate(action_list):
                is_image = (idx == 0)  # 第一个动作使用图像
                
                progress(0.7 + 0.2 * (idx / len(action_list)), desc=f"生成动作 {idx+1}/{len(action_list)}: {action_id}")
                
                outputs = self.hunyuan_video_sampler.predict(
                    prompt=final_prompt,
                    action_id=action_id,
                    action_speed=action_speeds[idx],                    
                    is_image=is_image,
                    size=(video_height, video_width),
                    seed=seed,
                    last_latents=last_latents,
                    ref_latents=ref_latents,
                    video_length=33,  # 固定33帧
                    guidance_scale=cfg_scale,
                    num_images_per_prompt=1,
                    negative_prompt=add_neg_prompt,
                    infer_steps=infer_steps,
                    flow_shift=flow_shift,
                    ref_images=ref_images,
                    output_dir=self.output_dir,
                    return_latents=True,
                )
                
                # 更新latents
                ref_latents = outputs["ref_latents"]
                last_latents = outputs["last_latents"]
                
                # 拼接视频
                sub_samples = outputs['samples'][0]
                if idx == 0:
                    out_cat = sub_samples
                else:
                    out_cat = torch.cat([out_cat, sub_samples], dim=2)
            
            progress(0.95, desc="保存视频...")
            
            # 保存视频
            timestamp = int(time.time())
            save_path_mp4 = os.path.join(self.output_dir, f"generated_video_{timestamp}.mp4")
            save_videos_grid(out_cat, save_path_mp4, n_rows=1, fps=24)
            
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            progress(1.0, desc="生成完成!")
            return save_path_mp4, "✅ 视频生成成功!", f"视频已保存到: {save_path_mp4}"
                
        except Exception as e:
            return None, f"❌ 生成过程中出现错误: {str(e)}", str(e)

    def update_prompt(self, template_choice):
        """根据模板选择更新prompt"""
        if template_choice == "自定义":
            return gr.update(visible=True, value="")
        else:
            return gr.update(visible=False, value=self.prompt_templates[template_choice])

    def load_sample(self, sample_name):
        """加载示例数据"""
        samples = {
            "中世纪村庄": {
                "image": "asset/village.png",
                "prompt": "A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky.",
                "add_pos_prompt": "Realistic, High-quality.",
                "add_neg_prompt": "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
                "action_list": ["向前移动 (W)", "向后移动 (S)", "向右移动 (D)", "向左移动 (A)"],
                "action_speeds": "0.2 0.2 0.2 0.2",
                "seed": 250160
            }
        }
        
        if sample_name in samples:
            sample = samples[sample_name]
            # 加载图片
            image_path = sample["image"]
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                image = None
                
            return (
                image,
                sample_name,  # 直接使用sample_name作为prompt_template，自动选择对应模板
                sample["prompt"],  # custom_prompt
                sample["add_pos_prompt"],
                sample["add_neg_prompt"],
                sample["action_list"],
                sample["action_speeds"],
                sample["seed"]
            )
        else:
            return None, "自定义", "", "", "", [], "", 250160

    def create_interface(self):
        """创建Gradio界面"""
        
        # 检测GPU信息
        gpu_info = "CPU模式"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info = f"{gpu_name} ({gpu_memory:.1f}GB)"
        
        with gr.Blocks(title="Hunyuan GameCraft 视频生成器", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎮 Hunyuan GameCraft 视频生成器")
            gr.Markdown("基于腾讯混元GameCraft模型的交互式视频生成工具")
            
            # 显示系统状态
            status_info = f"📊 系统状态: {'✅ 模型已就绪' if self.model_available else '⚠️ 模型文件缺失'} | GPU: {gpu_info}"
            gr.Markdown(status_info)
            
            # 添加示例区域
            with gr.Row():
                gr.Markdown("## 🎯 快速开始")
            with gr.Row():
                with gr.Column(scale=2):
                    sample_dropdown = gr.Dropdown(
                        choices=["中世纪村庄"],
                        label="选择官方示例",
                        value=None,
                        interactive=True
                    )
                with gr.Column(scale=1):
                    load_sample_btn = gr.Button("📥 加载示例", variant="secondary")
            
            # 模型选择区域
            with gr.Row():
                gr.Markdown("## 🤖 模型选择")
            with gr.Row():
                with gr.Column(scale=2):
                    model_dropdown = gr.Dropdown(
                        choices=list(self.model_configs.keys()),
                        value=self.current_model,
                        label="选择模型版本",
                        interactive=True
                    )
                    model_info = gr.Textbox(
                        value=f"当前: {self.model_configs[self.current_model]['description']}",
                        label="模型信息",
                        interactive=False,
                        lines=1
                    )
                with gr.Column(scale=1):
                    switch_model_btn = gr.Button("🔄 切换模型", variant="secondary")
            
            gr.Markdown("---")  # 分隔线
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 📸 输入设置")
                    
                    # 图片上传
                    image_input = gr.Image(
                        label="参考图片", 
                        type="pil",
                        height=300
                    )
                    
                    # Prompt设置
                    with gr.Group():
                        gr.Markdown("### 📝 描述提示词")
                        prompt_template = gr.Dropdown(
                            choices=list(self.prompt_templates.keys()),
                            value="中世纪村庄",
                            label="选择模板"
                        )
                        custom_prompt = gr.Textbox(
                            label="自定义描述",
                            placeholder="输入您的自定义描述...",
                            lines=3,
                            visible=False
                        )
                        
                        add_pos_prompt = gr.Textbox(
                            label="附加正面提示词",
                            value="Realistic, High-quality.",
                            lines=2
                        )
                        add_neg_prompt = gr.Textbox(
                            label="负面提示词",
                            value="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
                            lines=3
                        )
                    
                    # 动作设置
                    with gr.Group():
                        gr.Markdown("### 🎮 动作控制")
                        gr.Markdown("每个动作对应33帧（25FPS），可以组合多个动作生成长视频")
                        
                        action_checkboxes = gr.CheckboxGroup(
                            choices=list(self.action_options.values()),
                            label="选择动作序列",
                            value=["向前移动 (W)", "向后移动 (S)", "向右移动 (D)", "向左移动 (A)"]
                        )
                        
                        action_speeds = gr.Textbox(
                            label="动作速度 (0-3之间，用空格分隔)",
                            value="0.2 0.2 0.2 0.2",
                            placeholder="例如: 0.2 0.5 1.0 0.8"
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("## ⚙️ 生成参数")
                    
                    with gr.Group():
                        gr.Markdown("### 🎥 视频设置")
                        with gr.Row():
                            video_width = gr.Slider(
                                minimum=512, maximum=1920, step=64, value=1216,
                                label="视频宽度"
                            )
                            video_height = gr.Slider(
                                minimum=512, maximum=1080, step=64, value=704,
                                label="视频高度"
                            )
                    
                    with gr.Group():
                        gr.Markdown("### 🎛️ 生成控制")
                        cfg_scale = gr.Slider(
                            minimum=1.0, maximum=10.0, step=0.1, value=2.0,
                            label="CFG Scale (控制提示词遵循度)"
                        )
                        seed = gr.Number(
                            label="随机种子", 
                            value=250160,
                            precision=0
                        )
                        infer_steps = gr.Slider(
                            minimum=5, maximum=100, step=1, value=30,
                            label="推理步数 (加速模型推荐8步，标准模型推荐30-50步)"
                        )
                        flow_shift = gr.Slider(
                            minimum=1.0, maximum=10.0, step=0.1, value=5.0,
                            label="Flow Shift"
                        )
                    
                    # 生成按钮
                    generate_btn = gr.Button(
                        "🚀 开始生成视频", 
                        variant="primary",
                        size="lg"
                    )
            
            # 输出区域
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📹 生成结果")
                    status_output = gr.Textbox(
                        label="状态",
                        value="等待开始生成...",
                        lines=2
                    )
                    video_output = gr.Video(
                        label="生成的视频",
                        height=400
                    )
            
            # 日志输出
            with gr.Accordion("📋 详细日志", open=False):
                log_output = gr.Textbox(
                    label="生成日志",
                    lines=10,
                    max_lines=20
                )
            
            # 事件绑定
            prompt_template.change(
                fn=self.update_prompt,
                inputs=[prompt_template],
                outputs=[custom_prompt]
            )
            
            # 示例加载事件
            load_sample_btn.click(
                fn=self.load_sample,
                inputs=[sample_dropdown],
                outputs=[
                    image_input, prompt_template, custom_prompt,
                    add_pos_prompt, add_neg_prompt, 
                    action_checkboxes, action_speeds, seed
                ]
            )
            
            # 模型切换事件
            switch_model_btn.click(
                fn=self.switch_model,
                inputs=[model_dropdown],
                outputs=[infer_steps, cfg_scale, model_info]
            )
            
            generate_btn.click(
                fn=self.generate_video,
                inputs=[
                    image_input, prompt_template, custom_prompt, 
                    add_pos_prompt, add_neg_prompt,
                    action_checkboxes, action_speeds,
                    video_width, video_height, cfg_scale, 
                    seed, infer_steps, flow_shift
                ],
                outputs=[video_output, status_output, log_output]
            )
            
            # 添加使用说明
            gr.Markdown("## 💡 使用提示")
            gr.Markdown("""
            1. **快速开始**: 点击"📥 加载示例"按钮，自动加载中世纪村庄示例（包括图片、描述和参数）
            2. **模型选择**: 
               - **标准模型**: 质量更高，推理步数30步，速度较慢，适合最终作品
               - **加速模型**: 推理步数仅8步，速度快4倍，质量略低，适合快速预览
            3. **自定义使用**: 
               - 上传图片：选择一张高质量的参考图片作为视频的起始帧
               - 设置描述：选择"自定义"模板，输入您的描述文字
            4. **配置动作**: 选择移动方向和对应的速度值
            5. **调整参数**: 根据需要调整视频尺寸和生成参数
            6. **开始生成**: 点击生成按钮，等待视频生成完成
            
            **动作说明**:
            - W: 向前移动 | S: 向后移动 | A: 向左移动 | D: 向右移动
            - 可以组合使用，如WA表示向前向左移动
            - 速度值范围0-3，数值越大移动距离越大
            
            **性能建议**:
            - 首次使用建议选择"加速模型"进行快速测试
            - 满意效果后再使用"标准模型"生成最终视频
            - 加速模型推理时间约为标准模型的1/4
            
            **硬件要求**:
            - 建议使用NVIDIA GPU，至少8GB显存
            - 推理时间与动作序列长度成正比
            """)
        
        return interface

def main():
    """主函数"""
    print("🚀 启动 Hunyuan GameCraft WebUI...")
    
    webui = HunyuanGameCraftWebUI()
    interface = webui.create_interface()
    
    # 启动界面
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )

if __name__ == "__main__":
    main()
