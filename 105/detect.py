VIDEO_DIR = "video/"
TMP_DIR = "tmp/"
import os
import cv2
from transformers import AutoProcessor, ShieldGemma2ForImageClassification
from PIL import Image
import torch
import time

# 加载模型和处理器
model_id = "checkpoints/shieldgemma-2-4b-it"
model = ShieldGemma2ForImageClassification.from_pretrained(model_id).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

# 只检测色情内容
POLICY = "Sexually Explicit content"
POLICY_DESC = "The image shall not contain content that depicts explicit or graphic sexual acts."

def check_porn(image_path):
    image = Image.open(image_path).convert("RGB")
    policies = [POLICY]
    policy_descriptions = [POLICY_DESC]
    custom_policies = dict(zip(policies, policy_descriptions))
    inputs = processor(
        images=[image],
        custom_policies=custom_policies,
        policies=policies,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output = model(**inputs)
    yes_prob = output.probabilities.cpu()[0][0].item()
    return yes_prob > 0.5

def extract_frames():
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    for filename in os.listdir(VIDEO_DIR):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(VIDEO_DIR, filename)
            video_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(TMP_DIR, video_name)
            os.makedirs(output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(output_dir, f"{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                # 检查色情内容
                if check_porn(frame_path):
                    print(f"Porn detected: {frame_path}")
                frame_idx += 1
            cap.release()

if __name__ == "__main__":
    start_time = time.time()
    extract_frames()
    print("Frames extracted successfully.")
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")