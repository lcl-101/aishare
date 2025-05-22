# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import gradio as gr
import torch
from PIL import Image
import random
import numpy as np

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from inferencer import InterleaveInferencer

# Model loading
model_path = "checkpoints/BAGEL-7B-MoT"

llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config,
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

max_mem_per_gpu = "40GiB"
device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]
for k in same_device_modules:
    device_map[k] = device_map[same_device_modules[0]]

if not os.path.exists("offload"):
    os.makedirs("offload")

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    offload_folder="offload",
    dtype=torch.bfloat16,
)
model = model.eval()

inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids
)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 图5：文本生成图像
def t2i_fn(prompt):
    inference_hyper = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
    )
    output_dict = inferencer(text=prompt, **inference_hyper)
    return output_dict['image']

# 图6：文本生成图像（带think）
def t2i_think_fn(prompt):
    inference_hyper = dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
    )
    output_dict = inferencer(text=prompt, think=True, **inference_hyper)
    return output_dict['text'], output_dict['image']

# 图7：图像编辑
def edit_fn(image, prompt):
    inference_hyper = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    )
    output_dict = inferencer(image=image, text=prompt, **inference_hyper)
    return output_dict['image']

# 图8：图像编辑（带think）
def edit_think_fn(image, prompt):
    inference_hyper = dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )
    output_dict = inferencer(image=image, text=prompt, think=True, **inference_hyper)
    return output_dict['text'], output_dict['image']

# 图9：图像理解
def understand_fn(image, prompt):
    inference_hyper = dict(
        max_think_token_n=1000,
        do_sample=False,
    )
    output_dict = inferencer(image=image, text=prompt, understanding_output=True, **inference_hyper)
    return output_dict['text']

with gr.Blocks() as demo:
    with gr.Tab("Text2Image"):
        gr.Markdown("## 文本生成图像（图5）")
        prompt = gr.Textbox(label="Prompt", value="A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere.")
        btn = gr.Button("生成")
        output = gr.Image(label="生成图像")
        btn.click(t2i_fn, inputs=prompt, outputs=output)
    with gr.Tab("Text2Image-Think"):
        gr.Markdown("## 文本生成图像-带Think（图6）")
        prompt = gr.Textbox(label="Prompt", value="a car made of small cars")
        btn = gr.Button("生成")
        text_out = gr.Textbox(label="中间推理文本")
        img_out = gr.Image(label="生成图像")
        btn.click(t2i_think_fn, inputs=prompt, outputs=[text_out, img_out])
    with gr.Tab("Edit"):
        gr.Markdown("## 图像编辑（图7）")
        image = gr.Image(type="pil", label="输入图像")
        prompt = gr.Textbox(label="编辑描述", value="She wear a glass, wearing the same clothes.")
        btn = gr.Button("编辑")
        output = gr.Image(label="编辑后图像")
        btn.click(edit_fn, inputs=[image, prompt], outputs=output)
    with gr.Tab("Edit-Think"):
        gr.Markdown("## 图像编辑-带Think（图8）")
        image = gr.Image(type="pil", label="输入图像")
        prompt = gr.Textbox(label="编辑描述", value="Could you display the sculpture that takes after this design?")
        btn = gr.Button("编辑")
        text_out = gr.Textbox(label="中间推理文本")
        img_out = gr.Image(label="编辑后图像")
        btn.click(edit_think_fn, inputs=[image, prompt], outputs=[text_out, img_out])
    with gr.Tab("Understand"):
        gr.Markdown("## 图像理解（图9）")
        image = gr.Image(type="pil", label="输入图像")
        prompt = gr.Textbox(label="问题", value="请详细地描述一下图片的内容。")
        btn = gr.Button("理解")
        output = gr.Textbox(label="理解结果")
        btn.click(understand_fn, inputs=[image, prompt], outputs=output)

demo.launch(server_name="0.0.0.0")
