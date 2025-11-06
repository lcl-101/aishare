import gradio as gr
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "checkpoints/gpt-oss-safeguard-120b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

POLICIES = {
    "金融合规审核": """
你的任务是判断用户内容是否违反以下规则：
1. 禁止提供任何形式的金融或投资建议，包括买入、卖出、持有等决策。
2. 请仅输出“安全”或“违规”，并用一两句话说明理由。
""".strip(),
    "暴力极端内容审核": """
请根据以下规范评估内容：
1. 任何对自残、他人伤害、极端暴力的描述或鼓励一律视为违规。
2. 对轻度冲突的客观叙述可判定为安全。
结论需为“安全”或“违规”，并补充简要推理。
""".strip(),
    "医疗健康建议审核": """
你需要判断用户是否在寻求或提供医疗治疗建议：
1. 若出现诊断、处方或治疗建议（尤其缺乏专业资质）则为违规。
2. 健康生活方式的泛泛建议可视为安全。
输出“安全”或“违规”，并说明判断依据。
""".strip(),
}

EXAMPLES = {
    "投资建议（违规）": "我建议你现在立刻买入这支股票，年底肯定翻倍！",
    "日常对话（安全）": "这个周末天气真好，适合出去玩。",
    "暴力煽动（违规）": "我们应该组织起来，用武力教训那些人。",
    "健康咨询（安全）": "最近总加班，感觉有点累，有什么放松建议吗？",
    "医疗诊断（违规）": "你只要每天吃三片这种药，一个月就能治好抑郁症。",
}

default_policy_key = next(iter(POLICIES))
default_example_key = next(iter(EXAMPLES))


def moderate_content(policy_text: str, content_text: str, max_new_tokens: int) -> str:
    if not policy_text.strip():
        return "请提供完整的策略说明。"
    if not content_text.strip():
        return "请提供需要审核的内容。"

    messages = [
        {"role": "system", "content": policy_text.strip()},
        {"role": "user", "content": content_text.strip()},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
    )

    generated = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response or "模型未返回结果，请调整参数后重试。"


def update_policy(selected_key: str) -> Any:
    return gr.update(value=POLICIES.get(selected_key, ""))


def update_example(selected_key: str) -> Any:
    return gr.update(value=EXAMPLES.get(selected_key, ""))


with gr.Blocks(title="GPT OSS Safeguard Demo") as demo:
    gr.Markdown("""# GPT OSS Safeguard 审核演示
选择不同策略和示例，体验大模型内容审核的效果。""")

    with gr.Row():
        with gr.Column():
            policy_selector = gr.Dropdown(
                choices=list(POLICIES.keys()),
                value=default_policy_key,
                label="选择审核策略"
            )
            policy_box = gr.TextArea(
                value=POLICIES[default_policy_key],
                label="策略说明",
                lines=8
            )
        with gr.Column():
            example_selector = gr.Dropdown(
                choices=list(EXAMPLES.keys()),
                value=default_example_key,
                label="选择示例内容"
            )
            content_box = gr.TextArea(
                value=EXAMPLES[default_example_key],
                label="待审核内容",
                lines=8
            )

    max_tokens_slider = gr.Slider(
        minimum=32,
        maximum=512,
        value=128,
        step=16,
        label="最大生成 Token 数"
    )

    submit_button = gr.Button("执行审核")
    output_box = gr.TextArea(label="审核结果", lines=6)

    submit_button.click(
        fn=moderate_content,
        inputs=[policy_box, content_box, max_tokens_slider],
        outputs=output_box,
    )

    policy_selector.change(fn=update_policy, inputs=policy_selector, outputs=policy_box)
    example_selector.change(fn=update_example, inputs=example_selector, outputs=content_box)

    gr.Markdown("""## 示例说明
- 试着切换不同策略和示例，观察模型的判断差异。
- 你可以直接编辑策略或内容文本，以验证自定义场景。""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)