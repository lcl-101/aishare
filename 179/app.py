import os
import time
import logging
from pathlib import Path

import torch
from torchvision.io import read_image
import torchvision.transforms.v2 as transforms
from torchvision.utils import make_grid

import gradio as gr
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import SiglipImageProcessor, SiglipVisionModel
from huggingface_hub import hf_hub_download

from esrgan_model import UpscalerESRGAN
from model import create_model

device = "cuda"
# Custom timer logger only
timer_logger = logging.getLogger("TIMER")
timer_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()  # Attach a stream handler with formatter
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))
timer_logger.addHandler(handler)
timer_logger.propagate = False  # Avoid duplicate logs

# Custom transform to pad images to square
class PadToSquare:
    def __call__(self, img):
        _, h, w = img.shape
        max_side = max(h, w)
        pad_h = (max_side - h) // 2
        pad_w = (max_side - w) // 2
        padding = (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h)
        return transforms.functional.pad(img, padding, padding_mode="edge")

# Timer decorator
def timer_func(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        timer_logger.info(f"{func.__name__} took {time.time() - t0:.2f} seconds")
        return result
    return wrapper

@timer_func
def load_model(model_class_name, model_filename, repo_id: str = "rizavelioglu/tryoffdiff"):
    path_model = hf_hub_download(repo_id=repo_id, filename=model_filename, force_download=False)
    state_dict = torch.load(path_model, weights_only=True, map_location=device)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model = create_model(model_class_name).to(device)
    # model = torch.compile(model)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()

def validate_garment_selection(garment_types):
    """Validate garment type selection and return selected types and label indices."""
    label_map = {"Upper-Body": 0, "Lower-Body": 1, "Dress": 2}
    valid_single = ["Upper-Body", "Lower-Body", "Dress"]
    valid_tuple = ["Upper-Body", "Lower-Body"]

    if not garment_types:
        raise gr.Error("Please select at least one garment type.")
    if len(garment_types) == 1 and garment_types[0] in valid_single:
        selected, label_indices = garment_types, [label_map[garment_types[0]]]
    elif sorted(garment_types) == sorted(valid_tuple):
        selected, label_indices = valid_tuple, [label_map[t] for t in valid_tuple]
    else:
        raise gr.Error("Invalid selection. Choose one garment type or Upper-Body and Lower-Body together.")
    
    return selected, label_indices

def generate_multi_image_wrapper(input_image, garment_types, seed=42, guidance_scale=2.0, num_inference_steps=50, is_upscale=False):
    """Wrapper function that validates input before calling the GPU function."""
    # Validate selection before entering GPU context
    selected, label_indices = validate_garment_selection(garment_types)
    return generate_multi_image(input_image, selected, label_indices, seed, guidance_scale, num_inference_steps, is_upscale)

@torch.no_grad()
@timer_func
def generate_multi_image(input_image, selected, label_indices, seed=42, guidance_scale=2.0, num_inference_steps=50, is_upscale=False):
    batch_size = len(selected)
    scheduler.set_timesteps(num_inference_steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch_size, 4, 64, 64, generator=generator, device=device)

    # Process inputs
    cond_image = img_enc_transform(read_image(input_image))
    inputs = {k: v.to(device) for k, v in img_processor(images=cond_image, return_tensors="pt").items()}
    cond_emb = img_enc(**inputs).last_hidden_state.to(device)
    cond_emb = cond_emb.expand(batch_size, *cond_emb.shape[1:])
    uncond_emb = torch.zeros_like(cond_emb) if guidance_scale > 1 else None
    label = torch.tensor(label_indices, device=device, dtype=torch.int64)
    model = models["multi"]

    with torch.autocast(device):
        for t in scheduler.timesteps:
            t = t.to(device)  # Ensure t is on the correct device
            if guidance_scale > 1:
                noise_pred = model(torch.cat([x] * 2), t, torch.cat([uncond_emb, cond_emb]), torch.cat([label, label])).chunk(2)
                noise_pred = noise_pred[0] + guidance_scale * (noise_pred[1] - noise_pred[0])  # Classifier-free guidance
            else:
                noise_pred = model(x, t, cond_emb, label)  # Standard prediction

            # Scheduler step
            scheduler_output = scheduler.step(noise_pred, t, x)
            x = scheduler_output.prev_sample

    # Decode predictions from latent space
    decoded = vae.decode(1 / vae.config.scaling_factor * scheduler_output.pred_original_sample).sample
    images = (decoded / 2 + 0.5).cpu()
    grid = make_grid(images, nrow=len(images), normalize=True, scale_each=True)
    output_image = transforms.ToPILImage()(grid)
    return upscaler(output_image) if is_upscale else output_image  # Optionally upscale the output image

@torch.no_grad()
@timer_func
def generate_upper_image(input_image, seed=42, guidance_scale=2.0, num_inference_steps=50, is_upscale=False):
    model = models["upper"]
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, 4, 64, 64, generator=generator, device=device)

    # Process input image
    cond_image = img_enc_transform(read_image(input_image))
    inputs = {k: v.to(device) for k, v in img_processor(images=cond_image, return_tensors="pt").items()}
    cond_emb = img_enc(**inputs).last_hidden_state.to(device)
    uncond_emb = torch.zeros_like(cond_emb) if guidance_scale > 1 else None

    with torch.autocast(device):
        for t in scheduler.timesteps:
            t = t.to(device)  # Ensure t is on the correct device
            if guidance_scale > 1:  # Classifier-free guidance
                noise_pred = model(torch.cat([x] * 2), t, torch.cat([uncond_emb, cond_emb])).chunk(2)
                noise_pred = noise_pred[0] + guidance_scale * (noise_pred[1] - noise_pred[0])
            else:  # Standard prediction
                noise_pred = model(x, t, cond_emb)

            # Scheduler step
            scheduler_output = scheduler.step(noise_pred, t, x)
            x = scheduler_output.prev_sample

    # Decode predictions from latent space
    decoded = vae.decode(1 / vae.config.scaling_factor * scheduler_output.pred_original_sample).sample
    images = (decoded / 2 + 0.5).cpu()
    grid = make_grid(images, nrow=len(images), normalize=True, scale_each=True)
    output_image = transforms.ToPILImage()(grid)
    return upscaler(output_image) if is_upscale else output_image  # Optionally upscale the output image

@torch.no_grad()
@timer_func
def generate_lower_image(input_image, seed=42, guidance_scale=2.0, num_inference_steps=50, is_upscale=False):
    model = models["lower"]
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, 4, 64, 64, generator=generator, device=device)

    # Process input image
    cond_image = img_enc_transform(read_image(input_image))
    inputs = {k: v.to(device) for k, v in img_processor(images=cond_image, return_tensors="pt").items()}
    cond_emb = img_enc(**inputs).last_hidden_state.to(device)
    uncond_emb = torch.zeros_like(cond_emb) if guidance_scale > 1 else None

    with torch.autocast(device):
        for t in scheduler.timesteps:
            t = t.to(device)  # Ensure t is on the correct device
            if guidance_scale > 1:  # Classifier-free guidance
                noise_pred = model(torch.cat([x] * 2), t, torch.cat([uncond_emb, cond_emb])).chunk(2)
                noise_pred = noise_pred[0] + guidance_scale * (noise_pred[1] - noise_pred[0])
            else:  # Standard prediction
                noise_pred = model(x, t, cond_emb)

            # Scheduler step
            scheduler_output = scheduler.step(noise_pred, t, x)
            x = scheduler_output.prev_sample

    # Decode predictions from latent space
    decoded = vae.decode(1 / vae.config.scaling_factor * scheduler_output.pred_original_sample).sample
    images = (decoded / 2 + 0.5).cpu()
    grid = make_grid(images, nrow=len(images), normalize=True, scale_each=True)
    output_image = transforms.ToPILImage()(grid)
    return upscaler(output_image) if is_upscale else output_image  # Optionally upscale the output image

@torch.no_grad()
@timer_func
def generate_dress_image(input_image, seed=42, guidance_scale=2.0, num_inference_steps=50, is_upscale=False):
    model = models["dress"]
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, 4, 64, 64, generator=generator, device=device)

    # Process input image
    cond_image = img_enc_transform(read_image(input_image))
    inputs = {k: v.to(device) for k, v in img_processor(images=cond_image, return_tensors="pt").items()}
    cond_emb = img_enc(**inputs).last_hidden_state.to(device)
    uncond_emb = torch.zeros_like(cond_emb) if guidance_scale > 1 else None

    with torch.autocast(device):
        for t in scheduler.timesteps:
            t = t.to(device)  # Ensure t is on the correct device
            if guidance_scale > 1:  # Classifier-free guidance
                noise_pred = model(torch.cat([x] * 2), t, torch.cat([uncond_emb, cond_emb])).chunk(2)
                noise_pred = noise_pred[0] + guidance_scale * (noise_pred[1] - noise_pred[0])
            else:  # Standard prediction
                noise_pred = model(x, t, cond_emb)

            # Scheduler step
            scheduler_output = scheduler.step(noise_pred, t, x)
            x = scheduler_output.prev_sample

    # Decode predictions from latent space
    decoded = vae.decode(1 / vae.config.scaling_factor * scheduler_output.pred_original_sample).sample
    images = (decoded / 2 + 0.5).cpu()
    grid = make_grid(images, nrow=len(images), normalize=True, scale_each=True)
    output_image = transforms.ToPILImage()(grid)
    return upscaler(output_image) if is_upscale else output_image  # Optionally upscale the output image

def create_multi_tab():
    description = r"""
    <table class="description-table">
      <tr>
        <td width="50%">
          In total, 4 models are available for generating garments (one in each tab):<br>
          - <b>Multi-Garment</b>: Generate multiple garments (e.g., upper-body and lower-body) sequentially.<br>
          - <b>Upper-Body</b>: Generate upper-body garments (e.g., tops, jackets, etc.).<br>
          - <b>Lower-Body</b>: Generate lower-body garments (e.g., pants, skirts, etc.).<br>
          - <b>Dress</b>: Generate dresses.<br>
        </td>
        <td width="50%">
          <b>How to use:</b><br>
          1. Upload a reference image,<br>
          2. Adjust the parameters as needed,<br>
          3. Click "Generate" to create the garment(s).<br>
          &#128161; Individual models perform slightly better than the multi-garment model, but the latter is more versatile.
        </td>
      </tr>
    </table>
    """
    examples = [
        ["examples/048851_0.jpg", ["Upper-Body", "Lower-Body"], 42, 2.0, 20, False],
        ["examples/048851_0.jpg", ["Upper-Body"], 42, 2.0, 20, False],
        ["examples/048588_0.jpg", ["Upper-Body", "Lower-Body"], 42, 2.0, 20, False],
        ["examples/048588_0.jpg", ["Upper-Body"], 42, 2.0, 20, False],
        ["examples/048643_0.jpg", ["Upper-Body", "Lower-Body"], 42, 2.0, 20, False],
        ["examples/048643_0.jpg", ["Lower-Body"], 42, 2.0, 20, False],
        ["examples/048737_0.jpg", ["Dress"], 42, 2.0, 20, False],
        ["examples/048737_0.jpg", ["Upper-Body", "Lower-Body"], 42, 2.0, 20, False],
        ["examples/048690_0.jpg", ["Upper-Body", "Lower-Body"], 42, 2.0, 20, False],
        ["examples/048690_0.jpg", ["Lower-Body"], 42, 2.0, 20, False],
        ["examples/048691_0.jpg", ["Upper-Body", "Lower-Body"], 42, 2.0, 20, False],
        ["examples/048691_0.jpg", ["Upper-Body"], 42, 2.0, 20, False],
        ["examples/048732_0.jpg", ["Upper-Body", "Lower-Body"], 42, 2.0, 20, False],
        ["examples/048754_0.jpg", ["Upper-Body", "Lower-Body"], 42, 2.0, 20, False],
        ["examples/048799_0.jpg", ["Upper-Body", "Lower-Body"], 42, 2.0, 20, False],
        ["examples/048811_0.jpg", ["Upper-Body", "Lower-Body"], 42, 2.0, 20, False],
        ["examples/048821_0.jpg", ["Upper-Body", "Lower-Body"], 42, 2.0, 20, False],
        ["examples/048821_0.jpg", ["Upper-Body"], 42, 2.0, 20, False],
    ]

    with gr.Blocks() as tab:
        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Reference Image", height=384, width=384)
            with gr.Column(min_width=250):
                garment_type = gr.CheckboxGroup(["Upper-Body", "Lower-Body", "Dress"], label="Select Garment Type", value=["Upper-Body", "Lower-Body"])
                seed = gr.Slider(value=42, minimum=0, maximum=1e6, step=1, label="Seed")
                guidance_scale = gr.Slider(value=2.0, minimum=1, maximum=5, step=0.5, label="Guidance Scale(s)", info="No guidance at s=1.")
                inference_steps = gr.Slider(value=20, minimum=5, maximum=1000, step=10, label="# of Inference Steps")
                upscale = gr.Checkbox(value=False, label="Upscale Output", info="Upscale output by 4x (2048x2048) using an off-the-shelf model.")
                submit_btn = gr.Button("Generate")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Generated Garment", height=384, width=384)
        gr.Examples(examples=examples, inputs=[input_image, garment_type, seed, guidance_scale, inference_steps, upscale], outputs=output_image, fn=generate_multi_image_wrapper, cache_examples=False, examples_per_page=2)
        gr.Markdown(article)
        submit_btn.click(
            fn=generate_multi_image_wrapper,
            inputs=[input_image, garment_type, seed, guidance_scale, inference_steps, upscale],
            outputs=output_image
        )
    return tab

def create_upper_tab():
    examples = [[f"examples/{img_filename}", 42, 2.0, 20, False] for img_filename in os.listdir("examples/") if img_filename.endswith("_0.jpg")]
    examples += [
        ["examples/00084_00.jpg", 42, 2.0, 20, False],
        ["examples/00254_00.jpg", 42, 2.0, 20, False],
        ["examples/00397_00.jpg", 42, 2.0, 20, False],
        ["examples/01320_00.jpg", 42, 2.0, 20, False],
        ["examples/02390_00.jpg", 42, 2.0, 20, False],
        ["examples/14227_00.jpg", 42, 2.0, 20, False],
    ]
    with gr.Blocks() as tab:
        gr.Markdown(title)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Reference Image", height=384, width=384)
            with gr.Column(min_width=250):
                seed = gr.Slider(value=42, minimum=0, maximum=1e6, step=1, label="Seed")
                guidance_scale = gr.Slider(value=2.0, minimum=1, maximum=5, step=0.5, label="Guidance Scale(s)", info="No guidance at s=1.")
                inference_steps = gr.Slider(value=20, minimum=5, maximum=1000, step=10, label="# of Inference Steps")
                upscale = gr.Checkbox(value=False, label="Upscale Output", info="Upscale output by 4x (2048x2048) using an off-the-shelf model.")
                submit_btn = gr.Button("Generate")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Generated Garment", height=384, width=384)
        gr.Examples(examples=examples, inputs=[input_image, seed, guidance_scale, inference_steps, upscale], outputs=output_image, fn=generate_upper_image, cache_examples=False, examples_per_page=2)
        gr.Markdown(article)
        submit_btn.click(
            fn=generate_upper_image,
            inputs=[input_image, seed, guidance_scale, inference_steps, upscale],
            outputs=output_image
        )
    return tab

def create_lower_tab():
    examples = [[f"examples/{img_filename}", 42, 2.0, 20, False] for img_filename in os.listdir("examples/") if img_filename.endswith("_0.jpg")]
    with gr.Blocks() as tab:
        gr.Markdown(title)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Reference Image", height=384, width=384)
            with gr.Column(min_width=250):
                seed = gr.Slider(value=42, minimum=0, maximum=1e6, step=1, label="Seed")
                guidance_scale = gr.Slider(value=2.0, minimum=1, maximum=5, step=0.5, label="Guidance Scale(s)", info="No guidance at s=1.")
                inference_steps = gr.Slider(value=20, minimum=5, maximum=1000, step=10, label="# of Inference Steps")
                upscale = gr.Checkbox(value=False, label="Upscale Output", info="Upscale output by 4x (2048x2048) using an off-the-shelf model.")
                submit_btn = gr.Button("Generate")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Generated Garment", height=384, width=384)
        gr.Examples(examples=examples, inputs=[input_image, seed, guidance_scale, inference_steps, upscale], outputs=output_image, fn=generate_lower_image, cache_examples=False, examples_per_page=2)
        gr.Markdown(article)
        submit_btn.click(
            fn=generate_lower_image,
            inputs=[input_image, seed, guidance_scale, inference_steps, upscale],
            outputs=output_image
        )
    return tab

def create_dress_tab():
    examples = [
        ["examples/053480_0.jpg", 42, 2.0, 20, False],
        ["examples/048737_0.jpg", 42, 2.0, 20, False],
        ["examples/048811_0.jpg", 42, 2.0, 20, False],
        ["examples/053733_0.jpg", 42, 2.0, 20, False],
        ["examples/052606_0.jpg", 42, 2.0, 20, False],
        ["examples/053682_0.jpg", 42, 2.0, 20, False],
        ["examples/052036_0.jpg", 42, 2.0, 20, False],
        ["examples/052644_0.jpg", 42, 2.0, 20, False],
    ]
    with gr.Blocks() as tab:
        gr.Markdown(title)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Reference Image", height=384, width=384)
            with gr.Column(min_width=250):
                seed = gr.Slider(value=42, minimum=0, maximum=1e6, step=1, label="Seed")
                guidance_scale = gr.Slider(value=2.0, minimum=1, maximum=5, step=0.5, label="Guidance Scale(s)", info="No guidance at s=1.")
                inference_steps = gr.Slider(value=20, minimum=5, maximum=1000, step=10, label="# of Inference Steps")
                upscale = gr.Checkbox(value=False, label="Upscale Output", info="Upscale output by 4x (2048x2048) using an off-the-shelf model.")
                submit_btn = gr.Button("Generate")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Generated Garment", height=384, width=384)
        gr.Examples(examples=examples, inputs=[input_image, seed, guidance_scale, inference_steps, upscale], outputs=output_image, fn=generate_dress_image, cache_examples=False, examples_per_page=2)
        gr.Markdown(article)
        submit_btn.click(
            fn=generate_dress_image,
            inputs=[input_image, seed, guidance_scale, inference_steps, upscale],
            outputs=output_image
        )
    return tab

# UI elements
title = f"""
<div class='center-header' style="flex-direction: row; gap: 1.5em;">
    <h1 style="font-size:2.2em; margin-bottom:0.1em;">Virtual Try-Off Generator</h1>
    <a href='https://rizavelioglu.github.io/tryoffdiff' style="align-self:center;">
        <button style="background-color:#1976d2; color:white; font-weight:bold; border:none; border-radius:4px; padding:4px 10px; font-size:1.1em; cursor:pointer;">
            &#128279; Project page
        </button>
    </a>
</div>
"""
article = r"""
**Citation**<br>If you use this work, please give a star ⭐ and a citation:
```
@inproceedings{velioglu2025tryoffdiff,
  title     = {TryOffDiff: Virtual-Try-Off via High-Fidelity Garment Reconstruction using Diffusion Models},
  author    = {Velioglu, Riza and Bevandic, Petra and Chan, Robin and Hammer, Barbara},
  booktitle = {BMVC},
  year      = {2025},
  note      = {\url{https://doi.org/nt3n}}
}
@inproceedings{velioglu2025mgt,
  title     = {MGT: Extending Virtual Try-Off to Multi-Garment Scenarios},
  author    = {Velioglu, Riza and Bevandic, Petra and Chan, Robin and Hammer, Barbara},
  booktitle = {ICCVW},
  year      = {2025},
  note      = {\url{https://doi.org/pn67}}
}
```
"""
# Custom CSS for proper styling
custom_css = """
.center-header {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 0 20px 0;
}
.center-header h1 {
    margin: 0;
    text-align: center;
}
.description-table {
    width: 100%;
    border-collapse: collapse;
}
.description-table td {
    padding: 10px;
    vertical-align: top;
}
"""

if __name__ == "__main__":
    # Image Encoder and transforms
    img_enc_transform = transforms.Compose(
        [
            PadToSquare(),  # Custom transform to pad the image to a square
            transforms.Resize((512, 512)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    ckpt = "google/siglip-base-patch16-512"
    img_processor = SiglipImageProcessor.from_pretrained(ckpt, do_resize=False, do_rescale=False, do_normalize=False)
    img_enc = SiglipVisionModel.from_pretrained(ckpt).eval().to(device)

    # Initialize VAE (only Decoder will be used) & Noise Scheduler
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to(device)
    scheduler = EulerDiscreteScheduler.from_pretrained(
        hf_hub_download(repo_id="rizavelioglu/tryoffdiff", filename="scheduler/scheduler_config_v2.json", force_download=False)
    )
    scheduler.is_scale_input_called = True  # suppress warning

    # Upscaler model
    upscaler = UpscalerESRGAN(
        model_path=Path(hf_hub_download(repo_id="philz1337x/upscaler", filename="4x-UltraSharp.pth")),
        device=torch.device(device),
        dtype=torch.float32,
    )

    # Model configurations and loading
    models = {}
    model_paths = {
        "upper": {"class_name": "TryOffDiffv2Single", "path": "tryoffdiffv2_upper.pth"},  # internal code: model_20250213_134430
        "lower": {"class_name": "TryOffDiffv2Single", "path": "tryoffdiffv2_lower.pth"},  # internal code: model_20250213_134130
        "dress": {"class_name": "TryOffDiffv2Single", "path": "tryoffdiffv2_dress.pth"},  # internal code: model_20250213_133554
        "multi": {"class_name": "TryOffDiffv2", "path": "tryoffdiffv2_multi.pth"},  # internal code: model_20250310_155608
    }
    for name, cfg in model_paths.items():
        models[name] = load_model(cfg["class_name"], cfg["path"])
        torch.cuda.empty_cache()

    # Create tabbed interface
    demo = gr.TabbedInterface(
        [create_multi_tab(), create_upper_tab(), create_lower_tab(), create_dress_tab()],
        ["Multi-Garment", "Upper-Body", "Lower-Body", "Dress"],
        css=custom_css,
    )

    demo.launch(server_name="0.0.0.0")
