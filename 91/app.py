import gradio as gr
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# 模型路径映射
MODEL_OPTIONS = {
    "Instruct": "checkpoints/Kimi-VL-A3B-Instruct",
    "Think": "checkpoints/Kimi-VL-A3B-Thinking"
}

# 示例数据
EXAMPLES = [
    [
        "Instruct",
        ["./figures/demo.png"],
        "图片中那个圆顶建筑是什么？一步一步地想。"
    ],
    [
        "Think",
        ["./figures/demo1.png", "./figures/demo2.png"],
        "请一步一步推断这份手稿属于谁，以及它记录了什么内容。"
    ]
]

def infer(model_choice, image_files, user_input):
    model_path = MODEL_OPTIONS[model_choice]
    # 加载模型和处理器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 兼容 gr.Files 返回的多文件路径和单文件路径
    if not isinstance(image_files, list):
        image_files = [image_files]
    # 过滤 None
    image_files = [img for img in image_files if img]
    images = [Image.open(img) if not isinstance(img, Image.Image) else img for img in image_files]

    # 构建 messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img.name if hasattr(img, "name") else img} for img in image_files
            ] + [{"type": "text", "text": user_input}],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(images=images, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    max_new_tokens = 2048
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response

with gr.Blocks() as demo:
    gr.Markdown("# Kimi-VL WebUI Demo")
    with gr.Row():
        # 左侧：输入区（选择模型、问题描述、上传按钮、推理按钮、输出）
        with gr.Column(scale=7):
            model_choice = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value="Instruct",
                label="选择模型"
            )
            user_input = gr.Textbox(
                label="请输入问题描述",
                value="图片中那个圆顶建筑是什么？一步一步地想。"
            )
            with gr.Row():
                upload_btn = gr.Button("选择图片")
                run_btn = gr.Button("推理")
            output = gr.Textbox(label="模型输出", lines=10)
        # 右侧：图片预览区
        with gr.Column(scale=5):
            image_gallery = gr.Gallery(
                label="图片预览",
                show_label=True,
                elem_id="image_preview",
                height=360
            )
    # 隐藏的文件上传控件
    image_input = gr.Files(
        label="上传图片（可多选）",
        file_types=["image"],
        show_label=False,
        visible=False
    )
    # 上传图片时，更新预览
    def update_gallery(files):
        if not files:
            return []
        return files
    image_input.change(
        fn=update_gallery,
        inputs=image_input,
        outputs=image_gallery
    )
    # Examples 区域单独放在下方
    gr.Markdown("### 示例")
    examples_data = [
        [
            "Instruct",
            ["./figures/demo.png"],
            "图片中那个圆顶建筑是什么？一步一步地想。"
        ],
        [
            "Think",
            ["./figures/demo1.png", "./figures/demo2.png"],
            "请一步一步推断这份手稿属于谁，以及它记录了什么内容。"
        ]
    ]
    examples = gr.Examples(
        examples=examples_data,
        inputs=[model_choice, image_input, user_input],
        outputs=[model_choice, image_input, user_input],
        cache_examples=False
    )

    # 上传按钮逻辑
    upload_btn.click(
        lambda: gr.update(visible=True),
        None,
        [image_input]
    )
    run_btn.click(
        fn=infer,
        inputs=[model_choice, image_input, user_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
