import torch
import gradio as gr
from transformers import AutoModel, AutoTokenizer
from utils import load_image, split_model
import os
import tempfile
from PIL import Image

class SkyworkWebUI:
    def __init__(self, model_path='checkpoints/Skywork-R1V3-38B'):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """加载模型和分词器"""
        try:
            device_map = split_model(self.model_path)
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True, 
                use_fast=False
            )
            return "模型加载成功！"
        except Exception as e:
            return f"模型加载失败: {str(e)}"
    
    def process_images(self, image_files):
        """处理上传的图片文件"""
        if not image_files:
            return None, []
        
        print(f"Debug: image_files type: {type(image_files)}")
        print(f"Debug: image_files content: {image_files}")
        
        # 处理Gallery组件返回的图片列表
        pixel_values_list = []
        image_paths = []
        
        # Gallery组件返回的格式可能是不同类型的对象
        if isinstance(image_files, list):
            for img_item in image_files:
                print(f"Debug: processing item type: {type(img_item)}, content: {img_item}")
                if isinstance(img_item, str):
                    # 直接是文件路径
                    image_paths.append(img_item)
                elif hasattr(img_item, 'name'):
                    # 文件对象，获取路径
                    image_paths.append(img_item.name)
                elif isinstance(img_item, tuple) and len(img_item) > 0:
                    # 元组格式，通常第一个元素是路径
                    image_paths.append(img_item[0] if isinstance(img_item[0], str) else str(img_item[0]))
                else:
                    # 其他格式，尝试转换为字符串
                    image_paths.append(str(img_item))
        else:
            # 单个图片
            print(f"Debug: processing single item type: {type(image_files)}, content: {image_files}")
            if isinstance(image_files, str):
                image_paths = [image_files]
            elif hasattr(image_files, 'name'):
                image_paths = [image_files.name]
            elif isinstance(image_files, tuple) and len(image_files) > 0:
                image_paths = [image_files[0] if isinstance(image_files[0], str) else str(image_files[0])]
            else:
                image_paths = [str(image_files)]
        
        # 过滤掉空路径
        image_paths = [path for path in image_paths if path and path.strip() and path != 'None']
        
        print(f"Debug: final image_paths: {image_paths}")
        
        if not image_paths:
            return None, []
        
        # 加载图片
        pixel_values = [load_image(img_path, max_num=12).to(torch.bfloat16).cuda() for img_path in image_paths]
        
        if len(pixel_values) > 1:
            num_patches_list = [img.size(0) for img in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = pixel_values[0]
            num_patches_list = None
            
        return pixel_values, num_patches_list
    
    def chat(self, images, question, max_tokens=64000, temperature=0.6, top_p=0.95, repetition_penalty=1.05):
        """与模型对话"""
        if not question.strip():
            return "请输入问题"
        
        if not images:
            return "请上传至少一张图片"
        
        if self.model is None or self.tokenizer is None:
            return "模型尚未加载，请稍候再试"
        
        try:
            # 处理图片
            pixel_values, num_patches_list = self.process_images(images)
            
            if pixel_values is None:
                return "图片处理失败"
            
            # 计算实际的图片数量
            actual_image_count = len(images) if isinstance(images, list) else 1
            
            # 构建prompt，明确要求中文回答
            prompt = "<image>\n" * actual_image_count + f"请用中文回答以下问题：{question}"
            
            # 生成配置
            generation_config = dict(
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # 生成回答
            response = self.model.chat(
                self.tokenizer, 
                pixel_values, 
                prompt, 
                generation_config, 
                num_patches_list=num_patches_list
            )
            
            return response
            
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

# 创建WebUI实例
webui = SkyworkWebUI()

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="Skywork-R1V3 WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Skywork-R1V3 多模态大模型 WebUI")
        gr.Markdown("上传图片并提问，让AI为您解答！")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 图片上传和预览（合并为一个组件）
                images_input = gr.Gallery(
                    label="上传图片 📷 (支持多张图片，点击+号上传)",
                    show_label=True,
                    columns=3,
                    rows=2,
                    height=400,
                    allow_preview=True,
                    show_share_button=False,
                    show_download_button=False,
                    interactive=True,
                    type="filepath"
                )
                
                # 参数设置
                with gr.Accordion("高级设置", open=False):
                    max_tokens = gr.Slider(
                        minimum=100,
                        maximum=64000,
                        value=64000,
                        step=100,
                        label="最大生成tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.6,
                        step=0.1,
                        label="温度 (Temperature)"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        label="Top-p"
                    )
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.05,
                        step=0.05,
                        label="重复惩罚"
                    )
            
            with gr.Column(scale=1):
                # 问题输入
                question_input = gr.Textbox(
                    label="请输入您的问题 💬",
                    placeholder="例如：这张图片中有什么？请详细描述。",
                    lines=3
                )
                
                # 提交按钮
                submit_btn = gr.Button("🤖 生成回答", variant="primary", size="lg")
                
                # 回答显示
                answer_output = gr.Textbox(
                    label="AI回答 🎯",
                    lines=15,
                    max_lines=30,
                    show_copy_button=True
                )
                
                # 清除按钮
                clear_btn = gr.Button("🗑️ 清除", variant="secondary")
        
        # 示例
        gr.Examples(
            examples=[
                [None, "请用中文详细描述这张图片中的内容"],
                [None, "请用中文回答：图片中有哪些物体？"],
                [None, "请用中文分析图片中的场景和环境"],
                [None, "请用中文识别并描述图片中的文字内容"],
            ],
            inputs=[images_input, question_input],
            label="💡 示例问题"
        )
        
        # 事件处理
        def process_chat(images, question, max_tokens, temperature, top_p, repetition_penalty):
            if not images:
                return "请先上传图片"
            return webui.chat(images, question, max_tokens, temperature, top_p, repetition_penalty)
        
        def clear_all():
            return None, "", ""
        
        # 绑定事件
        submit_btn.click(
            fn=process_chat,
            inputs=[images_input, question_input, max_tokens, temperature, top_p, repetition_penalty],
            outputs=[answer_output]
        )
        
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[images_input, question_input, answer_output]
        )
        
        # 回车提交
        question_input.submit(
            fn=process_chat,
            inputs=[images_input, question_input, max_tokens, temperature, top_p, repetition_penalty],
            outputs=[answer_output]
        )
    
    return demo

def main():
    """主函数"""
    print("🚀 启动 Skywork-R1V3 WebUI...")
    print("📍 模型路径:", webui.model_path)
    
    demo = create_interface()
    
    # 启动界面
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

if __name__ == "__main__":
    main()
