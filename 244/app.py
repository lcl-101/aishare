import gradio as gr
import torch
from diffusers import ZImagePipeline

PIPELINE_PATH = "checkpoints/Z-Image-Turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_PIPE = None


def _preferred_dtype() -> torch.dtype:
    """Return the best dtype for the current device."""
    if DEVICE == "cuda":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if supports_bf16:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _load_pipeline() -> ZImagePipeline:
    """Load and cache the Z-Image pipeline once."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    dtype = _preferred_dtype()
    _PIPE = ZImagePipeline.from_pretrained(
        PIPELINE_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    _PIPE.to(DEVICE)

    # Flash attention provides a nice speed-up on supported GPUs, but silently skip otherwise.
    try:
        _PIPE.transformer.set_attention_backend("flash")
    except Exception:
        pass

    return _PIPE


def generate_image(prompt: str, height: int, width: int, steps: int, guidance: float, seed: int):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt must not be empty.")

    pipe = _load_pipeline()

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    result = pipe(
        prompt=prompt.strip(),
        height=int(height),
        width=int(width),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    ).images[0]

    return result


FULL_PROMPT = (
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
    "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
    "red flowers, beads. Holds round folding fan with lady, trees, bird. Neon "
    "lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. "
    "Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), "
    "blurred colorful distant lights."
)


EXAMPLES = [
    [
        FULL_PROMPT,
        1024,
        1024,
        9,
        0.0,
        -1,
    ],
    [
        "Given that chickens and rabbits are in the same cage, there are a total of 35 heads and 94 feet. Find the number of chickens and rabbits.",
        1024,
        1024,
        12,
        1.0,
        -1,
    ],
    [
        "帮我给《登科后》配图，最出名的两句",
        1024,
        1024,
        10,
        0.5,
        -1,
    ],
    [
        "一幅中国水墨画，画面表现‘孤舟蓑笠翁，独钓寒江雪’的意境。",
        1024,
        1024,
        12,
        0.8,
        -1,
    ],
    [
        "一张电影海报，画面是一个穿着宇航服的猫，背景是火星，上方写着标题‘喵星人登陆’，字体是红色的。",
        1024,
        1024,
        11,
        1.0,
        -1,
    ],
]


_load_pipeline()  # eager load so weights are ready at startup


with gr.Blocks() as demo:
    gr.Markdown("""
    # Z-Image Turbo Playground
    Generate high-quality images locally with the Z-Image Turbo checkpoint.
    """)

    with gr.Row():
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Describe what you would like to see...",
            lines=4,
            value=FULL_PROMPT,
        )

    with gr.Row():
        height_slider = gr.Slider(512, 1280, value=1024, step=64, label="Height")
        width_slider = gr.Slider(512, 1280, value=1024, step=64, label="Width")

    with gr.Accordion("Advanced", open=False):
        steps_slider = gr.Slider(4, 30, value=9, step=1, label="Inference Steps")
        guidance_slider = gr.Slider(0.0, 5.0, value=0.0, step=0.1, label="Guidance Scale")
        seed_input = gr.Number(value=-1, label="Seed (-1 for random)")

    generate_button = gr.Button("Generate", variant="primary")
    gallery = gr.Image(type="pil", label="Result", show_label=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt_input, height_slider, width_slider, steps_slider, guidance_slider, seed_input],
        examples_per_page=3,
    )

    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, height_slider, width_slider, steps_slider, guidance_slider, seed_input],
        outputs=gallery,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")