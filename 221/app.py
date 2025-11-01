from transformers import AutoProcessor, ShieldGemma2ForImageClassification
from PIL import Image
import gradio as gr
import torch

MODEL_ID = "checkpoints/shieldgemma-2-4b-it"

# 加载模型
model = ShieldGemma2ForImageClassification.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else "cpu"
).eval()

# 修复权重绑定问题（transformers 库的已知 bug）
if hasattr(model.model, 'tie_weights'):
    model.model.tie_weights()

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)

# 策略定义
POLICIES = {
    "Dangerous Content": 'The image shall not contain content that facilitates or encourages activities that could cause real-world harm (e.g., building firearms and explosive devices, promotion of terrorism, instructions for suicide).',
    "Sexually Explicit": 'The image shall not contain content that depicts explicit or graphic sexual acts (e.g., pornography, erotic nudity, depictions of rape or sexual assault).',
    "Violence/Gore": 'The image shall not contain content that depicts shocking, sensational, or gratuitous violence (e.g., excessive blood and gore, gratuitous violence against animals, extreme injury or moment of death).'
}

def format_policy_details():
    """生成策略说明的 Markdown"""
    lines = ["### 审核策略说明"]
    for name, desc in POLICIES.items():
        lines.append(f"**{name}**\n{desc}")
    lines.append("*模型会对所有策略依次执行审核。*")
    return "\n\n".join(lines)

def predict(image):
    """对图片进行安全审核"""
    if image is None:
        return "请上传图片。"

    # 预处理图片
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    # 模型推理
    inputs = processor(images=[image], return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model(**inputs)
    
    # 解析结果
    probs = outputs.probabilities.cpu()
    results = ["### 审核结果\n"]
    
    for idx, (policy_name, policy_desc) in enumerate(POLICIES.items()):
        prob_yes = probs[idx, 0].item()  # 违规概率
        prob_no = probs[idx, 1].item()   # 不违规概率
        conclusion = "违规 (Violated)" if prob_yes >= 0.5 else "安全 (Safe)"
        
        results.append(
            f"**{policy_name}**\n"
            f"- 违规概率：{prob_yes:.4f}\n"
            f"- 安全概率：{prob_no:.4f}\n"
            f"- **结论**：{conclusion}"
        )
    
    return "\n\n".join(results)

# 构建 Gradio 界面
with gr.Blocks(title="ShieldGemma 图像审查", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ShieldGemma 图像安全审查")
    gr.Markdown(format_policy_details())
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传图片")
            submit_btn = gr.Button("开始审核", variant="primary", size="lg")
        with gr.Column():
            result_output = gr.Markdown(value="*上传图片后点击'开始审核'*")
    
    submit_btn.click(predict, inputs=image_input, outputs=result_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
