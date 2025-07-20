import gradio as gr
from vllm import LLM, SamplingParams
import gc
import torch
import time

class SeedXTranslator:
    def __init__(self):
        self.models = {}
        self.model_paths = {
            "Seed-X-Instruct-7B": "checkpoints/Seed-X-Instruct-7B",
            "Seed-X-PPO-7B": "checkpoints/Seed-X-PPO-7B"
        }
        self.model_descriptions = {
            "Seed-X-Instruct-7B": "指令微调模型 (Instruction-tuned)",
            "Seed-X-PPO-7B": "PPO强化学习模型 (PPO Reinforcement Learning)"
        }
        self.loading_status = "🚀 正在初始化系统..."
        self.ready = False
        
        # 启动时自动加载所有模型
        self.load_all_models()
    
    def clear_memory(self):
        """清理GPU内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_all_models(self):
        """启动时加载所有模型"""
        print("🚀 开始加载 Seed-X 模型...")
        self.loading_status = "⏳ 正在加载模型，请稍候..."
        
        try:
            # 为同时加载两个模型优化内存配置
            memory_per_model = 0.35  # 每个模型使用35%显存，总共70%，留30%缓冲
            
            for i, (model_name, model_path) in enumerate(self.model_paths.items()):
                print(f"📦 [{i+1}/{len(self.model_paths)}] 加载模型: {model_name}")
                self.loading_status = f"⏳ 正在加载 {model_name}... ({i+1}/{len(self.model_paths)})"
                
                # 在加载第二个模型前清理内存
                if i > 0:
                    print("🧹 清理GPU内存...")
                    self.clear_memory()
                
                model = LLM(
                    model=model_path,
                    max_num_seqs=64,  # 进一步降低并发数
                    tensor_parallel_size=1,
                    enable_prefix_caching=True,
                    gpu_memory_utilization=memory_per_model,  # 保守的内存使用
                    trust_remote_code=True,
                    enforce_eager=True,  # 禁用CUDA图以节省内存
                    disable_log_stats=True  # 禁用统计日志
                )
                
                self.models[model_name] = model
                print(f"✅ {model_name} 加载完成")
            
            self.loading_status = f"✅ 所有模型已就绪！共加载 {len(self.models)} 个模型"
            self.ready = True
            print("🎉 所有模型加载完成，系统就绪！")
            
        except Exception as e:
            error_msg = f"❌ 模型加载失败: {str(e)}"
            self.loading_status = error_msg
            print(error_msg)
            self.ready = False
    
    def translate_single(self, text, model_name, task_type, temperature=0, max_tokens=512):
        """单个模型翻译"""
        try:
            if not self.ready:
                return "⏳ 系统正在初始化，请稍候..."
                
            if model_name not in self.models:
                return f"❌ 模型 {model_name} 未加载"
            
            # 构建提示词
            if task_type == "简单翻译":
                prompt = f"Translate the following English sentence into Chinese:\n{text} <zh>"
            elif task_type == "详细解释翻译":
                prompt = f"Translate the following English sentence into Chinese and explain it in detail:\n{text} <zh>"
            elif task_type == "技术术语翻译":
                prompt = f"Translate the following technical English sentence into Chinese:\n{text} <zh>"
            elif task_type == "诗意表达翻译":
                prompt = f"Translate the following poetic English sentence into Chinese, preserving the artistic beauty:\n{text} <zh>"
            else:
                prompt = f"Translate the following English sentence into Chinese:\n{text} <zh>"
            
            # 设置生成参数
            decoding_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                skip_special_tokens=True,
                top_p=0.9,
                frequency_penalty=0.1
            )
            
            # 生成翻译
            model = self.models[model_name]
            results = model.generate([prompt], decoding_params)
            response = results[0].outputs[0].text.strip()
            
            return response
            
        except Exception as e:
            return f"❌ 翻译失败: {str(e)}"
    
    def compare_models(self, text, task_type, temperature=0, max_tokens=512, progress=gr.Progress()):
        """比较所有模型的翻译结果"""
        if not text.strip():
            return "请输入要翻译的英文文本"
        
        if not self.ready:
            return "⏳ 系统正在初始化中，请稍候片刻..."
        
        results = {}
        model_names = list(self.model_paths.keys())
        total_models = len(model_names)
        
        # 并行翻译所有模型
        for i, model_name in enumerate(model_names):
            progress((i + 1) / total_models, desc=f"翻译中... ({model_name})")
            result = self.translate_single(text, model_name, task_type, temperature, max_tokens)
            results[model_name] = result
        
        # 格式化输出
        output = f"📝 **原文**: {text}\n"
        output += f"🎯 **任务类型**: {task_type}\n"
        output += f"⚙️ **参数**: Temperature={temperature}, Max_tokens={max_tokens}\n"
        output += "=" * 80 + "\n\n"
        
        for model_name, result in results.items():
            output += f"## 🤖 {model_name}\n"
            output += f"**描述**: {self.model_descriptions[model_name]}\n\n"
            output += f"**翻译结果**:\n```\n{result}\n```\n\n"
            output += "-" * 60 + "\n\n"
        
        return output
    
    def get_system_status(self):
        """获取系统状态"""
        if self.ready:
            return f"🟢 系统就绪 | 已加载 {len(self.models)} 个模型"
        else:
            return self.loading_status

# 初始化翻译器
translator = SeedXTranslator()

# 创建 Gradio 界面
def create_interface():
    with gr.Blocks(title="Seed-X 模型翻译对比系统", theme=gr.themes.Soft()) as iface:
        gr.Markdown("""
        # 🤖 Seed-X 模型翻译对比系统
        
        基于 Seed-X-Instruct-7B 和 Seed-X-PPO-7B 的实时翻译对比系统
        
        ## � 特点：
        - **预加载模型**: 启动时自动加载所有模型，翻译响应更快
        - **实时对比**: 同时使用两个模型翻译，直观比较效果差异
        - **多种任务**: 支持简单翻译、详细解释、技术术语、诗意表达等
        - **参数调节**: 可调节温度和输出长度等生成参数
        """)
        
        # 系统状态显示
        with gr.Row():
            system_status = gr.Textbox(
                label="🔧 系统状态",
                value=translator.get_system_status(),
                interactive=False,
                show_copy_button=False
            )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 翻译参数")
                
                task_type = gr.Dropdown(
                    choices=["简单翻译", "详细解释翻译", "技术术语翻译", "诗意表达翻译"],
                    value="简单翻译",
                    label="任务类型",
                    info="选择翻译任务的类型"
                )
                
                temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.1,
                    label="Temperature",
                    info="控制翻译的随机性，0=最确定，1=最随机"
                )
                
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="最大输出长度",
                    info="限制翻译结果的最大长度"
                )
                
                gr.Markdown("### 📊 模型信息")
                gr.Markdown("""
                **Seed-X-Instruct-7B**: 指令微调模型，擅长遵循翻译指令
                
                **Seed-X-PPO-7B**: PPO强化学习模型，提供更详细的推理过程
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 📝 翻译对比")
                
                input_text = gr.Textbox(
                    label="输入英文文本",
                    placeholder="请输入要翻译的英文文本，例如：May the force be with you",
                    lines=4
                )
                
                with gr.Row():
                    translate_btn = gr.Button("🚀 开始翻译对比", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                
                output_text = gr.Markdown(
                    value="准备就绪，请输入文本开始翻译...",
                    show_copy_button=True
                )
        
        # 预设示例
        gr.Markdown("### 💡 快速测试示例")
        with gr.Row():
            examples_data = [
                ("May the force be with you", "简单翻译"),
                ("Machine learning algorithms can process vast amounts of data", "技术术语翻译"), 
                ("The stars shine brightest in the darkest night", "诗意表达翻译"),
                ("Life is like a box of chocolates", "诗意表达翻译"),
                ("Artificial intelligence is transforming our world", "技术术语翻译"),
                ("Please explain the meaning behind this quote", "详细解释翻译")
            ]
            
            # 创建示例按钮
            for i in range(0, len(examples_data), 2):
                with gr.Row():
                    for j in range(2):
                        if i + j < len(examples_data):
                            text, task = examples_data[i + j]
                            btn = gr.Button(f"📝 {text[:35]}...", size="sm")
                            btn.click(
                                fn=lambda t=text, ta=task: (t, ta),
                                outputs=[input_text, task_type]
                            )
        
        # 绑定事件
        translate_btn.click(
            fn=translator.compare_models,
            inputs=[input_text, task_type, temperature, max_tokens],
            outputs=[output_text]
        )
        
        clear_btn.click(
            fn=lambda: ("", "准备就绪，请输入文本开始翻译..."),
            outputs=[input_text, output_text]
        )
        
        # 定期更新系统状态
        def update_status():
            return translator.get_system_status()
        
        # 在页面加载时更新状态
        iface.load(fn=update_status, outputs=[system_status])
    
    return iface

# 启动界面
if __name__ == "__main__":
    print("🚀 启动 Seed-X 翻译对比系统...")
    print("� 正在预加载模型，请稍候...")
    
    # 创建界面（此时会自动加载模型）
    demo = create_interface()
    
    print("🌐 启动 Web 服务...")
    print("📝 请在浏览器中访问显示的地址")
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 默认端口
        share=False,            # 不创建公开链接
        show_error=True,        # 显示错误信息
        quiet=False             # 显示启动信息
    )
