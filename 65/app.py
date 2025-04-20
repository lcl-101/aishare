import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor


device = "cuda:0"
model_path = "checkpoints/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def inference(video, question):
    if video is None:
        return "请上传视频文件"
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": video,
                        "fps": 1,
                        "max_frames": 180
                    }
                },
                {"type": "text", "text": question},
            ]
        },
    ]
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

def inference_image(image, question):
    if image is None:
        return "请上传图片文件"
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": image}},
                {"type": "text", "text": question},
            ]
        },
    ]
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

with gr.Blocks() as demo:
    gr.Markdown("<div align='center'>VideoLLaMA3 测试</div>")
    with gr.Tabs():
        with gr.TabItem("视频识别"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="上传视频")
                    question_input = gr.Textbox(label="输入问题", placeholder="请输入问题，例如：猫在做什么？")
                    submit_btn = gr.Button("提交")
                    examples = gr.Examples(
                        examples=[["./assets/cat_and_chicken.mp4", "请详细描述一下视频当中的内容。"]],
                        inputs=[video_input, question_input],
                        label="示例 (assets目录)"
                    )
                with gr.Column():
                    response_output = gr.Textbox(label="结果", interactive=False)
            submit_btn.click(fn=inference, inputs=[video_input, question_input], outputs=response_output)
        with gr.TabItem("图片识别"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="上传图片", type="filepath")
                    question_input_img = gr.Textbox(label="输入问题", placeholder="请输入问题，例如：图片中有什么？")
                    submit_btn_img = gr.Button("提交")
                    examples_img = gr.Examples(
                        examples=[
                            ["./assets/desert.jpg", "请详细描述图片当中的景观，请讲中文。"],
                            ["./assets/logo.png", "这个logo属于哪个公司？"],
                            ["./assets/performance.png", "请详细解释图中有哪些模型，以及性能指标的对比。"],
                            ["./assets/pipeline.jpg", "请详细描述一下图中，模型处理过程。"],
                            ["./assets/sora.png", "请详细描述一下这张图当中的内容？"]
                        ],
                        inputs=[image_input, question_input_img],
                        label="示例 (assets目录)"
                    )
                with gr.Column():
                    response_output_img = gr.Textbox(label="结果", interactive=False)
            submit_btn_img.click(fn=inference_image, inputs=[image_input, question_input_img], outputs=response_output_img)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')