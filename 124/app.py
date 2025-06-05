# Chain-of-Zoom WebUI (app.py)
# This file now contains a full copy of inference_coz.py for independent WebUI development.
# You may freely modify this file without affecting the CLI pipeline.

import os
import sys
sys.path.append(os.getcwd())
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
vlm_model_name = "checkpoints/Qwen2.5-VL-3B-Instruct"
print("[DEBUG] [GLOBAL] Loading Qwen2.5-VL-3B-Instruct...", flush=True)
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    vlm_model_name, torch_dtype="auto", device_map="cuda:0"
)
vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
print("[DEBUG] [GLOBAL] Qwen2.5-VL-3B-Instruct loaded.", flush=True)

def call_qwen_vl_for_bbox(image_path, prompt, vlm_model, vlm_processor, process_vision_info):
    # 让Qwen2.5-VL-3B-Instruct输出bbox字符串
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [{"type": "image", "image": image_path}]}
    ]
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=32)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]

def gradio_infer(input_image, prompt, efficient_memory):
    print("[DEBUG] Entered gradio_infer", flush=True)
    with open("debug_gradio_infer.log", "a") as f:
        f.write("[DEBUG] Entered gradio_infer\n")
    # 保存上传图片到临时文件
    input_path = "tmp_input.png"
    input_image.save(input_path)

    # 1. 使用全局Qwen2.5-VL-3B-Instruct
    print("[DEBUG] Using global Qwen2.5-VL-3B-Instruct", flush=True)
    # 2. 让Qwen输出bbox
    bbox_prompt = prompt + "，请只返回该区域的左上和右下坐标(x1,y1,x2,y2)数字，用英文逗号分隔。"
    bbox_str = call_qwen_vl_for_bbox(input_path, bbox_prompt, vlm_model, vlm_processor, process_vision_info)
    print(f"Qwen返回bbox: {bbox_str}")
    # 修正：只取前4个数字作为bbox
    try:
        import re
        nums = re.findall(r"-?\d+\.?\d*", bbox_str)
        bbox = [int(float(x)) for x in nums[:4]]
        print(f"[DEBUG] Parsed bbox: {bbox}", flush=True)
        if len(bbox) != 4:
            raise ValueError
    except Exception:
        print(f"[ERROR] bbox解析失败，原始Qwen输出: {bbox_str}", flush=True)
        return f"Qwen输出无法解析为bbox，终止。原始输出: {bbox_str}", None

    # 3. 裁剪图片
    image = Image.open(input_path).convert('RGB')
    # 扩展bbox区域，防止裁掉内容，但限制最大扩展比例
    w, h = image.size
    x1, y1, x2, y2 = bbox
    orig_w, orig_h = x2 - x1, y2 - y1
    pad_x = int(orig_w * 0.1)
    pad_y = int(orig_h * 0.1)
    # 限制最大扩展比例为1.2倍
    max_w = int(orig_w * 1.2)
    max_h = int(orig_h * 1.2)
    x1_new = max(0, x1 - pad_x)
    y1_new = max(0, y1 - pad_y)
    x2_new = min(w, x2 + pad_x)
    y2_new = min(h, y2 + pad_y)
    # 如果扩展后超过最大宽高，则收缩
    if x2_new - x1_new > max_w:
        extra = (x2_new - x1_new) - max_w
        x1_new += extra // 2
        x2_new -= extra - extra // 2
    if y2_new - y1_new > max_h:
        extra = (y2_new - y1_new) - max_h
        y1_new += extra // 2
        y2_new -= extra - extra // 2
    bbox_expanded = [x1_new, y1_new, x2_new, y2_new]
    print(f"[DEBUG] Original bbox: {bbox}, Expanded bbox: {bbox_expanded}", flush=True)
    cropped = image.crop(bbox_expanded)
    # 3.5. resize到512x512，保证和CLI一致
    cropped = cropped.resize((512, 512), Image.BICUBIC)

    # 4. 加载SR模型
    print("[DEBUG] About to load SR model", flush=True)
    with open("debug_gradio_infer.log", "a") as f:
        f.write("[DEBUG] About to load SR model\n")
    # 强制抛出异常测试日志
    # raise Exception("[DEBUG] Forced exception to test logging!")
    try:
        from osediff_sd3 import OSEDiff_SD3_TEST, OSEDiff_SD3_TEST_efficient, SD3Euler
        print("[DEBUG] Loading SD3Euler base model...", flush=True)
        weight_dtype = torch.float16
        model = SD3Euler()
        print("[DEBUG] SD3Euler loaded.", flush=True)
        # 确保所有SR模型组件在CUDA上
        model.text_enc_1.to('cuda:0')
        model.text_enc_2.to('cuda:0')
        model.text_enc_3.to('cuda:0')
        model.transformer.to('cuda:0', dtype=torch.float32)
        model.vae.to('cuda:0', dtype=torch.float32)
        for p in [model.text_enc_1, model.text_enc_2, model.text_enc_3, model.transformer, model.vae]:
            p.requires_grad_(False)
        class Args: pass
        args = Args()
        args.lora_path = "ckpt/SR_LoRA/model_20001.pkl"
        args.vae_path = "ckpt/SR_VAE/vae_encoder_20001.pt"
        args.lora_rank = 4
        args.mixed_precision = 'fp16'
        args.efficient_memory = efficient_memory
        print("[DEBUG] Injecting LoRA and VAE...", flush=True)
        if efficient_memory:
            model_test = OSEDiff_SD3_TEST_efficient(args, model)
        else:
            model_test = OSEDiff_SD3_TEST(args, model)
        print("[DEBUG] LoRA/encoder injected.", flush=True)
        # 关键：加载VAE encoder权重（必须在LoRA注入后，和CLI一致）
        encoder_state_dict_fp16 = torch.load(args.vae_path, map_location="cpu")
        model.vae.encoder.load_state_dict(encoder_state_dict_fp16)
        print("[DEBUG] VAE encoder state dict loaded.", flush=True)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("[ERROR] SR模型加载失败:", e, flush=True)
        print(tb, flush=True)
        return f"SR模型加载失败: {e}\n{tb}", None

    # 5. 超分
    try:
        print("[DEBUG] Preparing input tensor for SR...", flush=True)
        tensor_transforms = transforms.Compose([transforms.ToTensor()])
        lq = tensor_transforms(cropped).unsqueeze(0).to('cuda:0')
        lq = lq * 2 - 1
        # Efficient memory: ensure all needed modules are on CUDA for SR
        if efficient_memory:
            print("[DEBUG] Ensuring SR model components are on CUDA for SR inference...", flush=True)
            if not isinstance(model_test, OSEDiff_SD3_TEST_efficient):
                model.text_enc_1.to('cuda:0')
                model.text_enc_2.to('cuda:0')
                model.text_enc_3.to('cuda:0')
            model.transformer.to('cuda:0', dtype=torch.float32)
            model.vae.to('cuda:0', dtype=torch.float32)
        print("[DEBUG] Running SR inference...", flush=True)
        with torch.no_grad():
            output_image = model_test(lq, prompt=prompt)
            # 兼容 OSEDiff_SD3_TEST/efficient 返回 torch.Tensor 或 PIL.Image
            if isinstance(output_image, torch.Tensor):
                if output_image.dim() == 4:
                    output_image = output_image[0]
                output_image = torch.clamp(output_image.cpu(), -1.0, 1.0)
                output_pil = transforms.ToPILImage()(output_image * 0.5 + 0.5)
            elif isinstance(output_image, Image.Image):
                output_pil = output_image
            else:
                print("[ERROR] SR模型输出类型异常", flush=True)
                return "SR模型输出类型异常", None
        print("[DEBUG] SR inference complete.", flush=True)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("[ERROR] SR推理失败:", e, flush=True)
        print(tb, flush=True)
        return f"SR推理失败: {e}\n{tb}", None
    return output_pil, str(bbox)

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Chain-of-Zoom: 区域智能超分 WebUI (Qwen2.5-VL驱动)")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="上传图片")
                prompt = gr.Textbox(label="区域描述 (如: 衣服上的字)", value="衣服上的字")
                efficient_memory = gr.Checkbox(label="高效显存模式 (单卡建议勾选)", value=True)
                run_btn = gr.Button("超分！")
            with gr.Column():
                output_image = gr.Image(type="pil", label="超分结果")
                bbox_txt = gr.Textbox(label="Qwen返回的bbox坐标")
        run_btn.click(fn=gradio_infer, inputs=[input_image, prompt, efficient_memory], outputs=[output_image, bbox_txt])

    demo.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main()