import gradio as gr
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image, ImageDraw, ImageFont, ImageColor
import json
import ast
import os
import warnings
import requests
from io import BytesIO
from bs4 import BeautifulSoup, Tag
import re
import math
import numpy as np
import cv2
import random
import sys
import hashlib
import base64
from pdf2image import convert_from_path
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu

# Add cookbooks/utils to path for agent_function_call
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cookbooks'))
from utils.agent_function_call import ComputerUse, MobileUse
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*not used when initializing.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

# Model configuration
# You can switch between different models here:
# Option 1: Instruct model (recommended for OCR tasks)
# Using the smaller 30B model you downloaded
MODEL_PATH = "./checkpoints/Qwen3-VL-30B-A3B-Instruct"
# Option 2: (previously used large FP8 models)
# MODEL_PATH = "./checkpoints/Qwen3-VL-235B-A22B-Instruct-FP8/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"

# Global variables for model and processor
model = None
processor = None

def load_model():
    """Load the model and processor"""
    global model, processor
    
    # Load model with bfloat16 precision
    print(f"Loading model from {MODEL_PATH} with device_map='cuda'...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map='cuda',  # Use 'cuda' instead of 'auto' to put everything on GPU 0
    )
    print(f"Model loaded successfully to GPU: {torch.cuda.get_device_name(0)}")
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Model and processor loaded successfully!")

def parse_json(json_output):
    """Parse JSON output from markdown fencing"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def plot_text_bounding_boxes(image, bounding_boxes):
    """Plot bounding boxes on an image"""
    img = image.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Parse the JSON output
    bounding_boxes = parse_json(bounding_boxes)
    
    try:
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size=12)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", size=12)
            except:
                font = ImageFont.load_default()
        
        # Parse bounding boxes
        bbox_list = ast.literal_eval(bounding_boxes)
        
        for i, bounding_box in enumerate(bbox_list):
            color = 'green'
            
            # Convert normalized coordinates to absolute coordinates
            abs_y1 = int(bounding_box["bbox_2d"][1]/999 * height)
            abs_x1 = int(bounding_box["bbox_2d"][0]/999 * width)
            abs_y2 = int(bounding_box["bbox_2d"][3]/999 * height)
            abs_x2 = int(bounding_box["bbox_2d"][2]/999 * width)
            
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1
            
            # Draw the bounding box
            draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=2
            )
            
            # Draw the text
            if "text_content" in bounding_box:
                draw.text((abs_x1, abs_y2), bounding_box["text_content"], fill=color, font=font)
    except Exception as e:
        print(f"Error plotting bounding boxes: {e}")
    
    return img

def inference(image, prompt, max_new_tokens=4096):
    """Run inference on the image with the given prompt"""
    if model is None or processor is None:
        return "Error: Model not loaded. Please wait for model initialization.", None
    
    # Prepare messages following official format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def ocr_full_page(image):
    """Full-page OCR for text extraction"""
    prompt = "Read all the text in the image."
    response = inference(image, prompt)
    return response, None

def ocr_multilingual(image):
    """Full-page OCR for multilingual text"""
    prompt = "Please output only the text content from the image without any additional descriptions or formatting."
    response = inference(image, prompt)
    return response, None

def text_spotting_line(image):
    """Text spotting with line-level localization"""
    prompt = "Spotting all the text in the image with line-level, and output in JSON format as [{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...]."
    response = inference(image, prompt, max_new_tokens=8192)
    
    # Plot bounding boxes
    try:
        annotated_image = plot_text_bounding_boxes(image, response)
        return response, annotated_image
    except Exception as e:
        return response, None

def text_spotting_word(image):
    """Text spotting with word-level localization"""
    prompt = "Spotting all the text in the image with word-level, and output in JSON format as [{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...]."
    response = inference(image, prompt, max_new_tokens=8192)
    
    # Plot bounding boxes
    try:
        annotated_image = plot_text_bounding_boxes(image, response)
        return response, annotated_image
    except Exception as e:
        return response, None

def visual_info_extraction(image, keys):
    """Visual information extraction with given keys"""
    prompt = f"Extract the key-value information in the format:{keys}"
    response = inference(image, prompt)
    return response, None

def custom_ocr(image, custom_prompt):
    """Custom OCR with user-defined prompt"""
    response = inference(image, custom_prompt)
    return response, None


# ---------- Document parsing helpers (from cookbooks/document_parsing.ipynb) ----------
def clean_and_format_html(full_predict):
    soup = BeautifulSoup(full_predict, 'html.parser')
    
    # Regular expression pattern to match 'color' styles in style attributes
    color_pattern = re.compile(r'\bcolor:[^;]+;?')

    # Find all tags with style attributes and remove 'color' styles
    for tag in soup.find_all(style=True):
        original_style = tag.get('style', '')
        new_style = color_pattern.sub('', original_style)
        if not new_style.strip():
            del tag['style']
        else:
            new_style = new_style.rstrip(';')
            tag['style'] = new_style
            
    # Remove 'data-bbox' and 'data-polygon' attributes from all tags
    for attr in ["data-bbox", "data-polygon"]:
        for tag in soup.find_all(attrs={attr: True}):
            del tag[attr]

    classes_to_update = ['formula.machine_printed', 'formula.handwritten']
    # Update specific class names in div tags
    for tag in soup.find_all(class_=True):
        if isinstance(tag, Tag) and 'class' in tag.attrs:
            new_classes = [cls if cls not in classes_to_update else 'formula' for cls in tag.get('class', [])]
            tag['class'] = list(dict.fromkeys(new_classes))  # Deduplicate and update class names

    # Clear contents of divs with specific class names and rename their classes
    for div in soup.find_all('div', class_='image caption'):
        div.clear()
        div['class'] = ['image']

    classes_to_clean = ['music sheet', 'chemical formula', 'chart']
    # Clear contents and remove 'format' attributes of tags with specific class names
    for class_name in classes_to_clean:
        for tag in soup.find_all(class_=class_name):
            if isinstance(tag, Tag):
                tag.clear()
                if 'format' in tag.attrs:
                    del tag['format']

    # Manually build the output string
    output = []
    for child in soup.body.children:
        if isinstance(child, Tag):
            output.append(str(child))
            output.append('\n')  # Add newline after each top-level element
        elif isinstance(child, str) and not child.strip():
            continue  # Ignore whitespace text nodes
    complete_html = f"""```html\n<html><body>\n{" ".join(output)}</body></html>\n```"""
    return complete_html


def draw_bbox_html(image_input, full_predict):
    """Draw bounding boxes from Qwenvl HTML's data-bbox attributes and return annotated PIL image."""
    # load image
    if isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    elif isinstance(image_input, str) and image_input.startswith('http'):
        response = requests.get(image_input)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        # treat as path
        image = Image.open(image_input).convert('RGB')

    width, height = image.size
    soup = BeautifulSoup(full_predict, 'html.parser')
    elements_with_bbox = soup.find_all(attrs={'data-bbox': True})

    # Filtering logic similar to notebook
    filtered_elements = []
    for el in elements_with_bbox:
        if el.name == 'ol':
            continue
        elif el.name == 'li' and el.parent.name == 'ol':
            filtered_elements.append(el)
        else:
            filtered_elements.append(el)

    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 10)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)
    for element in filtered_elements:
        bbox_str = element.get('data-bbox', '')
        text = element.get_text(strip=True)
        try:
            x1, y1, x2, y2 = map(int, bbox_str.split())
        except Exception:
            continue

        bx1 = int(x1 / 1000 * width)
        by1 = int(y1 / 1000 * height)
        bx2 = int(x2 / 1000 * width)
        by2 = int(y2 / 1000 * height)
        if bx1 > bx2:
            bx1, bx2 = bx2, bx1
        if by1 > by2:
            by1, by2 = by2, by1

        draw.rectangle([bx1, by1, bx2, by2], outline='red', width=2)
        draw.text((bx1, by2), text, fill='black', font=font)

    return image


def draw_bbox_markdown(image_input, md_content):
    """Visualize coordinates embedded in markdown comments like <!-- Image/Table (x1, y1, x2, y2) -->"""
    if isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    elif isinstance(image_input, str) and image_input.startswith('http'):
        response = requests.get(image_input)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_input).convert('RGB')

    width = image.width
    height = image.height
    pattern = r"<!-- (Image|Table) \(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\) -->"
    matches = re.findall(pattern, md_content)
    draw = ImageDraw.Draw(image)
    for item in matches:
        typ, x1, y1, x2, y2 = item
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        bx1 = int(x1 / 1000 * width)
        by1 = int(y1 / 1000 * height)
        bx2 = int(x2 / 1000 * width)
        by2 = int(y2 / 1000 * height)
        if bx1 > bx2:
            bx1, bx2 = bx2, bx1
        if by1 > by2:
            by1, by2 = by2, by1
        color = 'blue' if typ == "Image" else 'red'
        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=6)

    return image


# ========== 2D Grounding Helper Functions ==========

def plot_2d_bounding_boxes(image, bounding_boxes_text):
    """Plot 2D bounding boxes from JSON output on image"""
    img = image.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Color list
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 
              'brown', 'gray', 'cyan', 'magenta', 'lime', 'navy', 'maroon']
    
    # Parse JSON
    bounding_boxes = parse_json(bounding_boxes_text)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size=14)
    except:
        font = ImageFont.load_default()
    
    try:
        json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
        # Try to truncate at last valid JSON object
        end_idx = bounding_boxes.rfind('"}') + len('"}')
        truncated_text = bounding_boxes[:end_idx] + "]"
        try:
            json_output = ast.literal_eval(truncated_text)
        except:
            return img
    
    if not isinstance(json_output, list):
        json_output = [json_output]
    
    # Draw each bounding box
    for i, bbox in enumerate(json_output):
        color = colors[i % len(colors)]
        
        # Convert normalized coordinates (0-1000) to absolute
        abs_x1 = int(bbox["bbox_2d"][0] / 1000 * width)
        abs_y1 = int(bbox["bbox_2d"][1] / 1000 * height)
        abs_x2 = int(bbox["bbox_2d"][2] / 1000 * width)
        abs_y2 = int(bbox["bbox_2d"][3] / 1000 * height)
        
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        
        # Draw rectangle
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3)
        
        # Draw label
        if "label" in bbox:
            draw.text((abs_x1 + 8, abs_y1 + 6), bbox["label"], fill=color, font=font)
    
    return img


def plot_2d_points(image, points_text):
    """Plot 2D points from JSON output on image"""
    img = image.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 
              'brown', 'gray', 'cyan', 'magenta', 'lime', 'navy', 'maroon']
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size=14)
    except:
        font = ImageFont.load_default()
    
    # Parse JSON
    points_data = parse_json(points_text)
    
    try:
        json_output = json.loads(points_data)
    except:
        return img
    
    if not isinstance(json_output, list):
        json_output = [json_output]
    
    # Draw each point
    for i, item in enumerate(json_output):
        if "point_2d" in item:
            color = colors[i % len(colors)]
            x, y = item["point_2d"]
            abs_x = int(x / 1000 * width)
            abs_y = int(y / 1000 * height)
            radius = 5
            
            # Draw circle
            draw.ellipse([(abs_x - radius, abs_y - radius), 
                         (abs_x + radius, abs_y + radius)], fill=color)
            
            # Draw label
            if "label" in item:
                draw.text((abs_x + 2*radius, abs_y + 2*radius), 
                         item["label"], fill=color, font=font)
    
    return img


# ========== 3D Grounding Helper Functions ==========

def parse_bbox_3d_from_text(text):
    """Parse 3D bounding box information from assistant response"""
    try:
        # Find JSON content
        if "```json" in text:
            start_idx = text.find("```json")
            end_idx = text.find("```", start_idx + 7)
            if end_idx != -1:
                json_str = text[start_idx + 7:end_idx].strip()
            else:
                json_str = text[start_idx + 7:].strip()
        else:
            # Find first [ and last ]
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx + 1]
            else:
                return []
        
        # Parse JSON
        bbox_data = json.loads(json_str)
        
        # Normalize to list format
        if isinstance(bbox_data, list):
            return bbox_data
        elif isinstance(bbox_data, dict):
            return [bbox_data]
        else:
            return []
            
    except (json.JSONDecodeError, IndexError, KeyError):
        return []


def convert_3dbbox(point, cam_params):
    """Convert 3D bounding box to 2D image coordinates"""
    x, y, z, x_size, y_size, z_size, pitch, yaw, roll = point
    hx, hy, hz = x_size / 2, y_size / 2, z_size / 2
    local_corners = [
        [ hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy,  hz],
        [ hx, -hy, -hz],
        [-hx,  hy,  hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [-hx, -hy, -hz]
    ]

    def rotate_xyz(_point, _pitch, _yaw, _roll):
        x0, y0, z0 = _point
        x1 = x0
        y1 = y0 * math.cos(_pitch) - z0 * math.sin(_pitch)
        z1 = y0 * math.sin(_pitch) + z0 * math.cos(_pitch)

        x2 = x1 * math.cos(_yaw) + z1 * math.sin(_yaw)
        y2 = y1
        z2 = -x1 * math.sin(_yaw) + z1 * math.cos(_yaw)

        x3 = x2 * math.cos(_roll) - y2 * math.sin(_roll)
        y3 = x2 * math.sin(_roll) + y2 * math.cos(_roll)
        z3 = z2

        return [x3, y3, z3]
    
    img_corners = []
    for corner in local_corners:
        rotated = rotate_xyz(corner, np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll))
        X, Y, Z = rotated[0] + x, rotated[1] + y, rotated[2] + z
        if Z > 0:
            x_2d = cam_params['fx'] * (X / Z) + cam_params['cx']
            y_2d = cam_params['fy'] * (Y / Z) + cam_params['cy']
            img_corners.append([x_2d, y_2d])

    return img_corners


def draw_3dbboxes_on_image(image, cam_params, bbox_3d_list):
    """Draw multiple 3D bounding boxes on PIL image and return annotated PIL image"""
    # Convert PIL to cv2
    if isinstance(image, Image.Image):
        annotated_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        return image
    
    edges = [
        [0,1], [2,3], [4,5], [6,7],
        [0,2], [1,3], [4,6], [5,7],
        [0,4], [1,5], [2,6], [3,7]
    ]
    
    # Draw 3D box for each bbox
    for bbox_data in bbox_3d_list:
        # Extract bbox_3d from the dictionary
        if isinstance(bbox_data, dict) and 'bbox_3d' in bbox_data:
            bbox_3d = bbox_data['bbox_3d']
        else:
            bbox_3d = bbox_data
        
        # Convert angles multiplied by 180 to degrees
        bbox_3d = list(bbox_3d)
        bbox_3d[-3:] = [_x * 180 for _x in bbox_3d[-3:]]
        bbox_2d = convert_3dbbox(bbox_3d, cam_params)

        if len(bbox_2d) >= 8:
            # Generate random color for each box
            box_color = [random.randint(0, 255) for _ in range(3)]
            for start, end in edges:
                try:
                    pt1 = tuple([int(_pt) for _pt in bbox_2d[start]])
                    pt2 = tuple([int(_pt) for _pt in bbox_2d[end]])
                    cv2.line(annotated_image, pt1, pt2, box_color, 2)
                except:
                    continue

    # Convert BGR back to RGB and then to PIL
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_image_rgb)


def load_camera_params(image_name):
    """Load camera parameters for a specific image from cam_infos.json"""
    cam_file = "cookbooks/assets/spatial_understanding/cam_infos.json"
    if os.path.exists(cam_file):
        with open(cam_file, 'r') as f:
            cam_infos = json.load(f)
        return cam_infos.get(image_name, None)
    return None


def generate_camera_params(image, fov=60):
    """Generate camera parameters for 3D visualization with default FOV=60°"""
    if isinstance(image, Image.Image):
        w, h = image.size
    else:
        return None
    
    # Generate pseudo camera params
    fx = round(w / (2 * np.tan(np.deg2rad(fov) / 2)), 2)
    fy = round(h / (2 * np.tan(np.deg2rad(fov) / 2)), 2)
    cx = round(w / 2, 2)
    cy = round(h / 2, 2)
    
    cam_params = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    return cam_params


# ========== Computer Use Helper Functions ==========

def draw_point(image, point, color=None):
    """Draw a point on image for Computer Use visualization"""
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)  
        except ValueError:
            color = (255, 0, 0, 128)  
    else:
        color = (255, 0, 0, 128)  

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    x, y = point 

    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color
    )
    
    center_radius = radius * 0.1
    overlay_draw.ellipse(
        [(x - center_radius, y - center_radius), 
         (x + center_radius, y + center_radius)],
        fill=(0, 255, 0, 255)
    )

    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)

    return combined.convert('RGB')


def perform_computer_use_grounding(image, user_query):
    """Perform GUI grounding using Qwen model to interpret user query on a screenshot"""
    import tempfile
    
    # Get image dimensions
    width, height = image.size
    
    # Save image to temporary file (required by ContentItem)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        temp_path = tmp.name
        image.save(temp_path)
    
    try:
        # Initialize computer use function
        computer_use = ComputerUse(
            cfg={"display_width_px": 1000, "display_height_px": 1000}
        )

        # Build messages using qwen_agent
        message = NousFnCallPrompt().preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=user_query),
                    ContentItem(image=f"file://{temp_path}")
                ]),
            ],
            functions=[computer_use.function],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]

        # Process input
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        # Generate output
        output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # Parse action and visualize
        try:
            if '<tool_call>' in output_text and '</tool_call>' in output_text:
                action_str = output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
                action = json.loads(action_str)
                
                if 'arguments' in action and 'coordinate' in action['arguments']:
                    coordinate_relative = action['arguments']['coordinate']
                    coordinate_absolute = [
                        coordinate_relative[0] / 1000 * width, 
                        coordinate_relative[1] / 1000 * height
                    ]
                    
                    display_image = draw_point(image, coordinate_absolute, color='green')
                    return output_text, display_image
            
            # If parsing failed, return original image
            return output_text, image
        except Exception as e:
            return f"{output_text}\n\n解析错误: {str(e)}", image
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ----------------------------------------------------------------------------------------
# Long Document Understanding Functions
# ----------------------------------------------------------------------------------------

def download_file(url, dest_path):
    """Download a file from URL to local path."""
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    return dest_path


def get_pdf_images(pdf_path, dpi=144, cache_dir='cache'):
    """Convert PDF to images with caching support."""
    os.makedirs(cache_dir, exist_ok=True)

    # Create a hash for the PDF path to use in cache filenames
    pdf_hash = hashlib.md5(pdf_path.encode('utf-8')).hexdigest()
    
    # Handle URL
    if pdf_path.startswith('http://') or pdf_path.startswith('https://'):
        pdf_file_path = os.path.join(cache_dir, f'{pdf_hash}.pdf')
        if not os.path.exists(pdf_file_path):
            download_file(pdf_path, pdf_file_path)
    else:
        pdf_file_path = pdf_path

    # Check for cached images
    images_cache_file = os.path.join(cache_dir, f'{pdf_hash}_{dpi}_images.npy')
    if os.path.exists(images_cache_file):
        images = np.load(images_cache_file, allow_pickle=True)
        pil_images = [Image.fromarray(image) for image in images]
        return pdf_file_path, pil_images

    # Convert PDF to images if not cached
    pil_images = convert_from_path(pdf_file_path, dpi=dpi)
    
    # Image size control - resize if too large
    resize_pil_images = []
    for img in pil_images:
        width, height = img.size
        max_side = max(width, height)
        max_side_value = 1500
        if max_side > max_side_value:
            img = img.resize((width * max_side_value // max_side, height * max_side_value // max_side))
        resize_pil_images.append(img)
    pil_images = resize_pil_images
    
    images = [np.array(img) for img in pil_images]
    
    # Save to cache
    np.save(images_cache_file, images)
    
    return pdf_file_path, pil_images


def create_image_grid(pil_images, num_columns=8):
    """Create a grid image from list of PIL images."""
    if not pil_images:
        return None
    
    num_rows = math.ceil(len(pil_images) / num_columns)
    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image


def image_to_base64(img, format="PNG"):
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64


def long_document_inference(pdf_file, prompt, dpi=144, max_new_tokens=4096, show_grid=True):
    """
    Process a long PDF document with multiple pages.
    Args:
        pdf_file: Path to PDF file (can be gradio File object)
        prompt: User query about the document
        dpi: DPI for PDF to image conversion
        max_new_tokens: Max tokens to generate
        show_grid: Whether to return the document grid visualization
    Returns:
        (response_text, grid_image or None)
    """
    if pdf_file is None:
        return "请上传PDF文件", None
    
    # Handle gradio File object
    if hasattr(pdf_file, 'name'):
        pdf_path = pdf_file.name
    else:
        pdf_path = pdf_file
    
    # Convert PDF to images
    _, pil_images = get_pdf_images(pdf_path, dpi=dpi)
    
    if not pil_images:
        return "PDF转换失败，未生成图片", None
    
    # Create content list with all page images
    content_list = []
    for image in pil_images:
        base64_image = image_to_base64(image)
        content_list.append({
            "type": "image",
            "image": f"data:image/png;base64,{base64_image}",
            "min_pixels": 512*32*32,
            "max_pixels": 730*32*32,
        })
    content_list.append({"type": "text", "text": prompt})
    
    messages = [{
        "role": "user",
        "content": content_list
    }]
    
    # Use existing model and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    # Create grid visualization if requested
    grid_image = None
    if show_grid:
        grid_image = create_image_grid(pil_images, num_columns=8)
        # Resize for display
        if grid_image:
            grid_image = grid_image.resize((1200, int(1200 * grid_image.height / grid_image.width)))
    
    page_info = f"处理了 {len(pil_images)} 页PDF文档 (DPI={dpi})\n\n"
    return page_info + output_text, grid_image

# ----------------------------------------------------------------------------------------
# Multimodal Coding Functions
# ----------------------------------------------------------------------------------------

def extract_last_code_block(text):
    """Extract the last named markdown code block from the text"""
    import re
    code_blocks = re.findall(r"```(?:python|html)(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    return None


def image_to_html(image, custom_prompt=None):
    """
    Convert screenshot/sketch to HTML code.
    Args:
        image: PIL Image
        custom_prompt: Optional custom prompt, otherwise use default
    Returns:
        (html_code_string, message)
    """
    if image is None:
        return "", "请上传图片"
    
    # Default prompt
    if not custom_prompt or not custom_prompt.strip():
        prompt = "Analyze this screenshot and convert it to clean, functional and modern HTML code."
    else:
        prompt = custom_prompt.strip()
    
    # Use the existing inference function
    try:
        response = inference(image, prompt, max_new_tokens=8192)
        
        # Extract code block
        code = extract_last_code_block(response)
        if code is None:
            return response, "模型返回中未找到代码块，显示原始输出"
        
        return code, f"成功生成 {len(code)} 字符的HTML代码"
    except Exception as e:
        return "", f"错误: {str(e)}"


def chart_to_matplotlib(image, custom_prompt=None):
    """
    Convert chart image to matplotlib code.
    Args:
        image: PIL Image
        custom_prompt: Optional custom prompt
    Returns:
        (python_code_string, message)
    """
    if image is None:
        return "", "请上传图片"
    
    # Default prompt
    if not custom_prompt or not custom_prompt.strip():
        prompt = "Convert this chart image to Python matplotlib code that can reproduce the chart."
    else:
        prompt = custom_prompt.strip()
    
    # Use the existing inference function
    try:
        response = inference(image, prompt, max_new_tokens=8192)
        
        # Extract code block
        code = extract_last_code_block(response)
        if code is None:
            return response, "模型返回中未找到代码块，显示原始输出"
        
        return code, f"成功生成 {len(code)} 字符的Python代码"
    except Exception as e:
        return "", f"错误: {str(e)}"

# ----------------------------------------------------------------------------------------
# Mobile Agent Functions
# ----------------------------------------------------------------------------------------

def rescale_coordinates(point, width, height):
    """Rescale coordinates from 999x999 to actual image dimensions."""
    point = [round(point[0]/999*width), round(point[1]/999*height)]
    return point


def mobile_agent_inference(image, instruction, history_text=""):
    """
    Mobile device agent with function calling.
    Args:
        image: Mobile screenshot (PIL Image)
        instruction: User query/instruction
        history_text: Operation history (e.g., "Step 1: xxx; Step 2: yyy;")
    Returns:
        (action_output_text, visualized_image)
    """
    if image is None:
        return "请上传移动设备截图", None
    
    import tempfile
    width, height = image.size
    
    # Save to temp file (ContentItem needs file path, not PIL object)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        temp_path = tmp.name
        image.save(temp_path)
    
    try:
        # Initialize MobileUse with screen resolution
        mobile_use = MobileUse(cfg={"display_width_px": 999, "display_height_px": 999})
        
        # Build system prompt with function description
        system_prompt = (
            "\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f'{{"type": "function", "function": {json.dumps(mobile_use.function)}}}\n'
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Thought: one concise sentence explaining the next move (no multi-step reasoning).\n"
            "2) Action: a short imperative describing what to do in the UI.\n"
            '3) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.\n\n'
            "Rules:\n"
            "- Output exactly in the order: Thought, Action, <tool_call>.\n"
            "- Be brief: one sentence for Thought, one for Action.\n"
            "- Do not output anything else outside those three parts.\n"
            "- If finishing, use action=terminate in the tool call."
        )
        
        # Build user query with history
        if history_text and history_text.strip():
            user_query = f"The user query: {instruction}.\nTask progress (You have done the following operation on the current device): {history_text}\n"
        else:
            user_query = f"The user query: {instruction}.\n"
        
        # Prepare messages using NousFnCallPrompt
        message = NousFnCallPrompt().preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text=system_prompt)]),
                Message(role="user", content=[
                    ContentItem(text=user_query),
                    ContentItem(image=f"file://{temp_path}")
                ]),
            ],
            functions=[mobile_use.function],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]
        
        # Inference
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        
        output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        # Parse action and visualize
        try:
            if '<tool_call>' in output_text and '</tool_call>' in output_text:
                action_str = output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
                action = json.loads(action_str)
                
                display_image = image.copy()
                
                # Visualize click actions
                if action.get('arguments', {}).get('action') == 'click':
                    coordinate = action['arguments'].get('coordinate', [])
                    if coordinate and len(coordinate) == 2:
                        # Rescale from 999x999 to actual dimensions
                        coordinate_absolute = rescale_coordinates(coordinate, width, height)
                        display_image = draw_point(display_image, coordinate_absolute, color='green')
                        
                        action_summary = f"动作: 点击坐标 {coordinate} (原始) -> {coordinate_absolute} (实际)\n\n完整输出:\n{output_text}"
                        return action_summary, display_image
                
                # For other actions, just return the output
                return output_text, display_image
            
            # If parsing failed, return original image
            return output_text, image
        except Exception as e:
            return f"{output_text}\n\n解析错误: {str(e)}", image
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ----------------------------------------------------------------------------------------
# Video Understanding Functions
# ----------------------------------------------------------------------------------------

def video_inference(video_input, prompt, max_new_tokens=2048, num_frames=64):
    """
    Perform video understanding inference.
    Args:
        video_input: Video file path (supports local files or URLs)
        prompt: User query about the video
        max_new_tokens: Maximum tokens to generate
        num_frames: Number of frames to sample from video
    Returns:
        (response_text, preview_grid_image)
    """
    if video_input is None or not video_input:
        return "请上传视频文件或输入视频URL", None
    
    # Handle gradio File object or URL string
    if hasattr(video_input, 'name'):
        video_path = video_input.name
    else:
        video_path = video_input.strip()
    
    try:
        # Download video if URL
        if video_path.startswith('http://') or video_path.startswith('https://'):
            video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
            cache_dir = 'cache'
            os.makedirs(cache_dir, exist_ok=True)
            local_path = os.path.join(cache_dir, f'{video_hash}.mp4')
            
            if not os.path.exists(local_path):
                response = requests.get(video_path, stream=True)
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8096):
                        f.write(chunk)
            video_path = local_path
        
        # Read video and sample frames
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        
        # Create preview grid
        pil_frames = [Image.fromarray(frame) for frame in frames]
        grid_image = create_image_grid(pil_frames, num_columns=8)
        # Resize for display
        if grid_image:
            grid_image = grid_image.resize((1200, int(1200 * grid_image.height / grid_image.width)))
        
        # Prepare messages for inference
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "video": video_path,
                        "total_pixels": 20480 * 32 * 32,
                        "min_pixels": 64 * 32 * 32,
                        "max_frames": num_frames,
                        "sample_fps": 2
                    },
                    {"type": "text", "text": prompt},
                ]
            },
        ]
        
        # Process with model
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True
        )
        
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # Generate
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        info = f"视频信息：总帧数 {total_frames}，采样 {num_frames} 帧\n\n"
        return info + output_text, grid_image
        
    except Exception as e:
        import traceback
        return f"错误: {str(e)}\n\n{traceback.format_exc()}", None

# ----------------------------------------------------------------------------------------

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Qwen3-VL OCR & 文档解析演示") as demo:
        gr.Markdown("# Qwen3-VL OCR & 文档解析演示")
        gr.Markdown("此演示展示了Qwen3-VL的OCR和文档解析能力。选择顶层任务和子任务来运行。")
        # example image paths (relative to repo root)
        examples_all = [
            "cookbooks/assets/ocr/ocr_example1.jpg",
            "cookbooks/assets/ocr/ocr_example2.jpg",
            "cookbooks/assets/ocr/ocr_example3.jpg",
            "cookbooks/assets/ocr/ocr_example4.jpg",
            "cookbooks/assets/ocr/ocr_example5.jpg",
            "cookbooks/assets/ocr/ocr_example6.jpg",
        ]

        # Task-specific example subsets (based on `cookbooks/ocr.ipynb` usage)
        examples_fullpage = [
            "cookbooks/assets/ocr/ocr_example2.jpg",
            "cookbooks/assets/ocr/ocr_example1.jpg",
        ]
        examples_multilingual = [
            "cookbooks/assets/ocr/ocr_example6.jpg",
            "cookbooks/assets/ocr/ocr_example5.jpg",
        ]
        examples_textspotting = [
            "cookbooks/assets/ocr/ocr_example3.jpg",
            "cookbooks/assets/ocr/ocr_example2.jpg",
        ]
        examples_vie = [
            # For VIE we include the image + a default keys JSON that will populate the keys_input
            ["cookbooks/assets/ocr/ocr_example3.jpg", '{"company": "", "date": "", "address": "", "total": ""}'],
            ["cookbooks/assets/ocr/ocr_example4.jpg", '{"invoice_code":"", "invoice_no":"", "date":"", "total":""}'],
        ]
        examples_custom = [
            ["cookbooks/assets/ocr/ocr_example1.jpg", "Describe the image and list all text contents."],
            ["cookbooks/assets/ocr/ocr_example5.jpg", "Recognize all Chinese text in the image."],
        ]
        
        # Document Parsing examples (from cookbooks/assets/document_parsing/)
        examples_doc_html = [
            "cookbooks/assets/document_parsing/docparsing_example1.jpg",
            "cookbooks/assets/document_parsing/docparsing_example2.jpg",
            "cookbooks/assets/document_parsing/docparsing_example3.jpg",
            "cookbooks/assets/document_parsing/docparsing_example4.jpg",
        ]
        examples_doc_markdown = [
            "cookbooks/assets/document_parsing/docparsing_example5.png",
            "cookbooks/assets/document_parsing/docparsing_example6.png",
            "cookbooks/assets/document_parsing/docparsing_example7.jpg",
            "cookbooks/assets/document_parsing/docparsing_example8.png",
        ]
        
        # 2D Grounding examples (from cookbooks/assets/spatial_understanding/)
        examples_multi_target = [
            "cookbooks/assets/spatial_understanding/drone_cars2.png",
        ]
        examples_spatial_reasoning = [
            "cookbooks/assets/spatial_understanding/football_field.jpg",
        ]
        
        # 3D Grounding examples (from cookbooks/assets/spatial_understanding/)
        examples_3d_detection = [
            "cookbooks/assets/spatial_understanding/autonomous_driving.jpg",
            "cookbooks/assets/spatial_understanding/office.jpg",
            "cookbooks/assets/spatial_understanding/lounge.jpg",
        ]
        
        # Computer Use examples (from cookbooks/assets/computer_use/)
        examples_computer_use = [
            ["cookbooks/assets/computer_use/computer_use1.jpeg", "Reload cache"],
            ["cookbooks/assets/computer_use/computer_use2.jpeg", "open the first issue"],
        ]
        
        with gr.Tabs():
            # Top-level: OCR
            with gr.Tab("文字识别"):
                with gr.Tabs():
                    # Tab 1: Full-page OCR
                    with gr.Tab("全页面OCR"):
                        with gr.Row():
                            with gr.Column():
                                input_image_1 = gr.Image(type="pil", label="上传图片")
                                gr.Markdown("**示例（全页面/文档类图片）**")
                                gr.Examples(examples_fullpage, inputs=[input_image_1], label="示例")
                                btn_1 = gr.Button("运行全页面OCR")
                            with gr.Column():
                                output_text_1 = gr.Textbox(label="提取的文本", lines=10)
                                output_image_1 = gr.Image(type="pil", label="标注图片")

                        btn_1.click(fn=ocr_full_page, inputs=[input_image_1], outputs=[output_text_1, output_image_1])

                    # Tab 2: Multilingual OCR
                    with gr.Tab("多语言OCR"):
                        with gr.Row():
                            with gr.Column():
                                input_image_2 = gr.Image(type="pil", label="上传图片")
                                gr.Markdown("**示例（多语言内容）**")
                                gr.Examples(examples_multilingual, inputs=[input_image_2], label="示例")
                                btn_2 = gr.Button("运行多语言OCR")
                            with gr.Column():
                                output_text_2 = gr.Textbox(label="提取的文本", lines=10)
                                output_image_2 = gr.Image(type="pil", label="标注图片")

                        btn_2.click(fn=ocr_multilingual, inputs=[input_image_2], outputs=[output_text_2, output_image_2])

                    # Tab 3: Text Spotting (Line-level)
                    with gr.Tab("文本定位（行级别）"):
                        with gr.Row():
                            with gr.Column():
                                input_image_3 = gr.Image(type="pil", label="上传图片")
                                gr.Markdown("**示例（文本定位/坐标识别）**")
                                gr.Examples(examples_textspotting, inputs=[input_image_3], label="示例")
                                btn_3 = gr.Button("运行文本定位（行级别）")
                            with gr.Column():
                                output_text_3 = gr.Textbox(label="检测到的文本及坐标", lines=10)
                                output_image_3 = gr.Image(type="pil", label="标注图片")

                        btn_3.click(fn=text_spotting_line, inputs=[input_image_3], outputs=[output_text_3, output_image_3])

                    # Tab 4: Text Spotting (Word-level)
                    with gr.Tab("文本定位（词级别）"):
                        with gr.Row():
                            with gr.Column():
                                input_image_4 = gr.Image(type="pil", label="上传图片")
                                gr.Markdown("**示例（词级定位/细粒度）**")
                                gr.Examples(examples_textspotting, inputs=[input_image_4], label="示例")
                                btn_4 = gr.Button("运行文本定位（词级别）")
                            with gr.Column():
                                output_text_4 = gr.Textbox(label="检测到的文本及坐标", lines=10)
                                output_image_4 = gr.Image(type="pil", label="标注图片")

                        btn_4.click(fn=text_spotting_word, inputs=[input_image_4], outputs=[output_text_4, output_image_4])

                    # Tab 5: Visual Information Extraction
                    with gr.Tab("视觉信息提取"):
                        with gr.Row():
                            with gr.Column():
                                input_image_5 = gr.Image(type="pil", label="上传图片")
                                keys_input = gr.Textbox(
                                    label="要提取的字段（JSON格式）", 
                                    value='{"company": "", "date": "", "address": "", "total": ""}',
                                    lines=3
                                )
                                gr.Markdown("**示例（发票/票据的键值对提取）**")
                                # examples_vie contains tuples of (image, default_keys_json)
                                gr.Examples(examples_vie, inputs=[input_image_5, keys_input], label="示例")
                                btn_5 = gr.Button("提取信息")
                            with gr.Column():
                                output_text_5 = gr.Textbox(label="提取的信息", lines=10)
                                output_image_5 = gr.Image(type="pil", label="标注图片")

                        btn_5.click(fn=visual_info_extraction, inputs=[input_image_5, keys_input], outputs=[output_text_5, output_image_5])

                    # Tab 6: Custom Prompt
                    with gr.Tab("自定义提示词"):
                        with gr.Row():
                            with gr.Column():
                                input_image_6 = gr.Image(type="pil", label="上传图片")
                                custom_prompt = gr.Textbox(
                                    label="自定义提示词", 
                                    placeholder="Enter your custom prompt here...",
                                    lines=3
                                )
                                gr.Markdown("**示例（点击图片填充示例提示词）**")
                                gr.Examples(examples_custom, inputs=[input_image_6, custom_prompt], label="示例")
                                btn_6 = gr.Button("运行自定义OCR")
                            with gr.Column():
                                output_text_6 = gr.Textbox(label="响应结果", lines=10)
                                output_image_6 = gr.Image(type="pil", label="标注图片")

                        btn_6.click(fn=custom_ocr, inputs=[input_image_6, custom_prompt], outputs=[output_text_6, output_image_6])
        
            # Top-level: Document Parsing
            with gr.Tab("文档解析"):
                with gr.Tabs():
                    with gr.Tab("Qwenvl HTML"):
                        with gr.Row():
                            with gr.Column():
                                doc_image_1 = gr.Image(type="pil", label="上传图片进行HTML解析")
                                gr.Markdown("**生成带位置信息的Qwenvl HTML**")
                                gr.Examples(examples_doc_html, inputs=[doc_image_1], label="示例")
                                btn_doc_html = gr.Button("解析为Qwenvl HTML")
                            with gr.Column():
                                output_html = gr.Textbox(label="解析的HTML（已清理）", lines=20)
                                annotated_html_image = gr.Image(type="pil", label="标注图片")

                        def run_doc_html(img):
                            prompt = "qwenvl html"
                            out = inference(img, prompt)
                            cleaned = clean_and_format_html(out)
                            ann = draw_bbox_html(img, out)
                            return cleaned, ann

                        btn_doc_html.click(fn=run_doc_html, inputs=[doc_image_1], outputs=[output_html, annotated_html_image])

                    with gr.Tab("Qwenvl Markdown"):
                        with gr.Row():
                            with gr.Column():
                                doc_image_2 = gr.Image(type="pil", label="上传图片进行Markdown解析")
                                gr.Markdown("**生成带位置信息的Qwenvl Markdown**")
                                gr.Examples(examples_doc_markdown, inputs=[doc_image_2], label="示例")
                                btn_doc_markdown = gr.Button("解析为Qwenvl Markdown")
                            with gr.Column():
                                output_md = gr.Textbox(label="解析的Markdown", lines=20)
                                annotated_md_image = gr.Image(type="pil", label="标注图片")

                        def run_doc_md(img):
                            prompt = "qwenvl markdown"
                            out = inference(img, prompt)
                            ann = draw_bbox_markdown(img, out)
                            return out, ann

                        btn_doc_markdown.click(fn=run_doc_md, inputs=[doc_image_2], outputs=[output_md, annotated_md_image])

            # Top-level: 2D Spatial Grounding
            with gr.Tab("2D空间定位"):
                with gr.Tabs():
                    # Task 1: Multi-Target Object Detection
                    with gr.Tab("多目标检测"):
                        with gr.Row():
                            with gr.Column():
                                grounding_image_1 = gr.Image(type="pil", label="上传图片")
                                categories_input = gr.Textbox(
                                    label="目标类别（用逗号分隔）",
                                    placeholder='e.g., "car, bus, bicycle, pedestrian"',
                                    value="car, bus, bicycle, pedestrian",
                                    lines=2
                                )
                                gr.Markdown("**示例（多目标检测场景）**")
                                gr.Examples(examples_multi_target, inputs=[grounding_image_1], label="示例")
                                btn_grounding_1 = gr.Button("检测目标")
                            with gr.Column():
                                output_grounding_1 = gr.Textbox(label="检测结果 (JSON)", lines=15)
                                output_grounding_image_1 = gr.Image(type="pil", label="标注图片")

                        def multi_target_detection(img, categories):
                            prompt = f'Locate every instance that belongs to the following categories: "{categories}". Report bbox coordinates in JSON format.'
                            result = inference(img, prompt, max_new_tokens=8192)
                            try:
                                annotated_img = plot_2d_bounding_boxes(img, result)
                            except:
                                annotated_img = img
                            return result, annotated_img

                        btn_grounding_1.click(
                            fn=multi_target_detection,
                            inputs=[grounding_image_1, categories_input],
                            outputs=[output_grounding_1, output_grounding_image_1]
                        )

                    # Task 2: Point-based Grounding
                    with gr.Tab("点定位"):
                        with gr.Row():
                            with gr.Column():
                                grounding_image_3 = gr.Image(type="pil", label="上传图片")
                                point_description = gr.Textbox(
                                    label="点定位提示词",
                                    placeholder='e.g., "Locate every person inside the football field with points"',
                                    value='Locate every person inside the football field with points, report their point coordinates, role(player, referee or unknown) and shirt color in JSON format like this: {"point_2d": [x, y], "label": "person", "role": "player/referee/unknown", "shirt_color": "the person\'s shirt color"}',
                                    lines=4
                                )
                                gr.Markdown("**示例（点定位场景）**")
                                gr.Examples(examples_spatial_reasoning, inputs=[grounding_image_3], label="示例")
                                btn_grounding_3 = gr.Button("定位点")
                            with gr.Column():
                                output_grounding_3 = gr.Textbox(label="检测结果 (JSON)", lines=15)
                                output_grounding_image_3 = gr.Image(type="pil", label="标注图片")

                        def point_based_grounding(img, prompt):
                            result = inference(img, prompt, max_new_tokens=8192)
                            try:
                                annotated_img = plot_2d_points(img, result)
                            except:
                                annotated_img = img
                            return result, annotated_img

                        btn_grounding_3.click(
                            fn=point_based_grounding,
                            inputs=[grounding_image_3, point_description],
                            outputs=[output_grounding_3, output_grounding_image_3]
                        )

            # Top-level: 3D Spatial Grounding
            with gr.Tab("3D空间定位"):
                with gr.Row():
                    with gr.Column():
                        grounding_3d_image = gr.Image(type="pil", label="上传图片")
                        detection_prompt_3d = gr.Textbox(
                            label="检测提示词",
                            placeholder='e.g., "Find all cars in this image. For each car, provide its 3D bounding box."',
                            value='Find all cars in this image. For each car, provide its 3D bounding box. The output format required is JSON: [{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],"label":"category"}].',
                            lines=4
                        )
                        use_default_cam = gr.Checkbox(
                            label="使用默认相机参数 (FOV=60°)",
                            value=True
                        )
                        gr.Markdown("**示例（3D场景检测）**")
                        gr.Markdown("注意：3D检测需要相机参数。如果图片在cam_infos.json中有记录会自动加载，否则使用默认参数。")
                        gr.Examples(examples_3d_detection, inputs=[grounding_3d_image], label="示例")
                        btn_grounding_3d = gr.Button("检测3D目标")
                    with gr.Column():
                        output_grounding_3d = gr.Textbox(label="检测结果 (JSON)", lines=15)
                        output_grounding_image_3d = gr.Image(type="pil", label="标注图片")
                        camera_info_display = gr.Textbox(label="相机参数信息", lines=3)

                def detect_3d_objects(img, prompt, use_default):
                    # Run inference
                    result = inference(img, prompt, max_new_tokens=8192)
                    
                    # Parse 3D bboxes
                    bbox_3d_list = parse_bbox_3d_from_text(result)
                    
                    if not bbox_3d_list:
                        return result, img, "未检测到3D边界框"
                    
                    # Get camera parameters
                    # Try to extract image name if it's from examples
                    cam_params = None
                    if use_default:
                        cam_params = generate_camera_params(img, fov=60)
                        cam_info = f"使用默认相机参数: fx={cam_params['fx']}, fy={cam_params['fy']}, cx={cam_params['cx']}, cy={cam_params['cy']}"
                    else:
                        # Try to load from cam_infos.json (would need image filename)
                        cam_params = generate_camera_params(img, fov=60)
                        cam_info = f"使用生成的相机参数: fx={cam_params['fx']}, fy={cam_params['fy']}, cx={cam_params['cx']}, cy={cam_params['cy']}"
                    
                    # Draw 3D bboxes
                    try:
                        annotated_img = draw_3dbboxes_on_image(img, cam_params, bbox_3d_list)
                    except Exception as e:
                        annotated_img = img
                        cam_info += f"\n绘制错误: {str(e)}"
                    
                    return result, annotated_img, cam_info

                btn_grounding_3d.click(
                    fn=detect_3d_objects,
                    inputs=[grounding_3d_image, detection_prompt_3d, use_default_cam],
                    outputs=[output_grounding_3d, output_grounding_image_3d, camera_info_display]
                )

            # Top-level: Computer Use
            with gr.Tab("计算机控制"):
                with gr.Row():
                    with gr.Column():
                        computer_use_image = gr.Image(type="pil", label="上传截图")
                        computer_use_query = gr.Textbox(
                            label="操作指令",
                            placeholder='e.g., "Reload cache", "open the first issue"',
                            value="Reload cache",
                            lines=2
                        )
                        gr.Markdown("**示例（GUI定位任务）**")
                        gr.Markdown("输入截图和操作指令，模型会定位需要点击的位置。")
                        gr.Examples(examples_computer_use, inputs=[computer_use_image, computer_use_query], label="示例")
                        btn_computer_use = gr.Button("定位操作位置")
                    with gr.Column():
                        output_computer_use = gr.Textbox(label="模型输出", lines=15)
                        output_computer_use_image = gr.Image(type="pil", label="标注截图")

                def run_computer_use(img, query):
                    if img is None:
                        return "请上传截图", None
                    try:
                        output_text, display_image = perform_computer_use_grounding(img, query)
                        return output_text, display_image
                    except Exception as e:
                        return f"错误: {str(e)}", img

                btn_computer_use.click(
                    fn=run_computer_use,
                    inputs=[computer_use_image, computer_use_query],
                    outputs=[output_computer_use, output_computer_use_image]
                )

            # Top-level: Long Document Understanding
            with gr.Tab("长文档理解"):
                with gr.Row():
                    with gr.Column():
                        pdf_file_input = gr.File(
                            label="上传PDF文件（或使用下方的示例链接）",
                            file_types=['.pdf'],
                            type="filepath"
                        )
                        
                        # Example URL input
                        with gr.Accordion("或使用示例PDF URL", open=False):
                            pdf_url_input = gr.Textbox(
                                label="PDF URL",
                                placeholder="输入PDF的URL地址",
                                value=""
                            )
                            gr.Markdown("**示例链接**：")
                            with gr.Row():
                                btn_example1 = gr.Button("📄 Qwen2.5-VL论文", size="sm")
                                btn_example2 = gr.Button("📊 技术文档", size="sm")
                        
                        long_doc_prompt = gr.Textbox(
                            label="文档问题",
                            placeholder='e.g., "Please summarize the key contributions of this paper."',
                            value="Please summarize the key contributions of this paper based on its abstract and introduction.",
                            lines=3
                        )
                        dpi_slider = gr.Slider(
                            minimum=72,
                            maximum=300,
                            value=144,
                            step=24,
                            label="PDF转图片DPI（数值越高质量越好但速度越慢）"
                        )
                        show_grid_checkbox = gr.Checkbox(
                            label="显示文档缩略图网格",
                            value=True
                        )
                        gr.Markdown("**说明**")
                        gr.Markdown("上传PDF文档（支持多页），模型会将所有页面作为输入进行理解和分析。支持本地文件或在线URL。")
                        btn_long_doc = gr.Button("分析文档", variant="primary")
                    with gr.Column():
                        output_long_doc = gr.Textbox(label="分析结果", lines=20)
                        output_long_doc_grid = gr.Image(type="pil", label="文档预览（缩略图网格）")

                # Set example URLs
                def set_example_url1():
                    return "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-VL/demo/longdoc/documents/Qwen2.5-VL.pdf"
                
                def set_example_url2():
                    return "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-VL/demo/longdoc/documents/fox_got_merge_code.pdf"
                
                btn_example1.click(fn=set_example_url1, outputs=[pdf_url_input])
                btn_example2.click(fn=set_example_url2, outputs=[pdf_url_input])

                def run_long_document(pdf_file, pdf_url, prompt, dpi, show_grid):
                    # Priority: use URL if provided, otherwise use uploaded file
                    if pdf_url and pdf_url.strip():
                        pdf_path = pdf_url.strip()
                    elif pdf_file is not None:
                        pdf_path = pdf_file
                    else:
                        return "请上传PDF文件或输入PDF URL", None
                    
                    try:
                        result, grid_img = long_document_inference(
                            pdf_path, 
                            prompt, 
                            dpi=int(dpi), 
                            max_new_tokens=4096,
                            show_grid=show_grid
                        )
                        return result, grid_img
                    except Exception as e:
                        return f"错误: {str(e)}", None

                btn_long_doc.click(
                    fn=run_long_document,
                    inputs=[pdf_file_input, pdf_url_input, long_doc_prompt, dpi_slider, show_grid_checkbox],
                    outputs=[output_long_doc, output_long_doc_grid]
                )

            # Top-level: Multimodal Coding
            with gr.Tab("多模态编程"):
                with gr.Tabs():
                    # Task 1: Image to HTML
                    with gr.Tab("图片转HTML"):
                        with gr.Row():
                            with gr.Column():
                                img2html_image = gr.Image(type="pil", label="上传截图或草图")
                                img2html_prompt = gr.Textbox(
                                    label="自定义提示词（可选）",
                                    placeholder="留空使用默认提示词",
                                    value="",
                                    lines=2
                                )
                                gr.Markdown("**说明**")
                                gr.Markdown("上传网页截图或UI草图，模型会生成对应的HTML代码。默认提示词会要求生成简洁、现代的HTML代码。")
                                gr.Markdown("**示例**")
                                gr.Examples(
                                    examples=[
                                        ["cookbooks/assets/multimodal_coding/screenshot_demo.png"],
                                        ["cookbooks/assets/multimodal_coding/sketch2code_input.jpeg"],
                                    ],
                                    inputs=[img2html_image],
                                    label="点击加载示例"
                                )
                                btn_img2html = gr.Button("生成HTML代码", variant="primary")
                            with gr.Column():
                                output_img2html_code = gr.Code(
                                    label="生成的HTML代码",
                                    language="html",
                                    lines=20
                                )
                                output_img2html_msg = gr.Textbox(label="状态信息", lines=2)

                        def run_img2html(img, prompt):
                            if img is None:
                                return "", "请上传图片"
                            code, msg = image_to_html(img, prompt)
                            return code, msg

                        btn_img2html.click(
                            fn=run_img2html,
                            inputs=[img2html_image, img2html_prompt],
                            outputs=[output_img2html_code, output_img2html_msg]
                        )

                    # Task 2: Chart to Code
                    with gr.Tab("图表转代码"):
                        with gr.Row():
                            with gr.Column():
                                chart2code_image = gr.Image(type="pil", label="上传图表图片")
                                chart2code_prompt = gr.Textbox(
                                    label="自定义提示词（可选）",
                                    placeholder="留空使用默认提示词",
                                    value="",
                                    lines=2
                                )
                                gr.Markdown("**说明**")
                                gr.Markdown("上传图表图片（如matplotlib生成的图表），模型会生成用于重现该图表的Python matplotlib代码。")
                                gr.Markdown("**示例**")
                                gr.Examples(
                                    examples=[
                                        ["cookbooks/assets/multimodal_coding/chart2code_input.png"],
                                    ],
                                    inputs=[chart2code_image],
                                    label="点击加载示例"
                                )
                                btn_chart2code = gr.Button("生成Python代码", variant="primary")
                            with gr.Column():
                                output_chart2code_code = gr.Code(
                                    label="生成的Matplotlib代码",
                                    language="python",
                                    lines=20
                                )
                                output_chart2code_msg = gr.Textbox(label="状态信息", lines=2)
                                gr.Markdown("**⚠️ 安全提示**")
                                gr.Markdown("生成的代码仅供参考，请不要直接执行未经审查的代码。")

                        def run_chart2code(img, prompt):
                            if img is None:
                                return "", "请上传图片"
                            code, msg = chart_to_matplotlib(img, prompt)
                            return code, msg

                        btn_chart2code.click(
                            fn=run_chart2code,
                            inputs=[chart2code_image, chart2code_prompt],
                            outputs=[output_chart2code_code, output_chart2code_msg]
                        )

            # Top-level: Mobile Agent
            with gr.Tab("移动设备代理"):
                with gr.Row():
                    with gr.Column():
                        mobile_screenshot = gr.Image(type="pil", label="上传移动设备截图")
                        mobile_instruction = gr.Textbox(
                            label="操作指令",
                            placeholder='e.g., "Search for Musk in X and go to his homepage"',
                            value="Search for Musk in X and go to his homepage to open the first post.",
                            lines=2
                        )
                        mobile_history = gr.Textbox(
                            label="操作历史（可选）",
                            placeholder='e.g., "Step 1: I opened the X app from the home screen."',
                            value="",
                            lines=3
                        )
                        gr.Markdown("**说明**")
                        gr.Markdown("""
                        上传移动设备截图，输入操作指令，模型会通过函数调用生成下一步操作动作。
                        
                        支持的动作类型：
                        - `click`: 点击屏幕坐标
                        - `swipe`: 滑动操作
                        - `type`: 输入文本
                        - `system_button`: 系统按钮（Back/Home/Menu/Enter）
                        - `wait`: 等待
                        - `terminate`: 完成任务
                        
                        屏幕坐标系统：999x999（会自动缩放到实际截图尺寸）
                        """)
                        gr.Markdown("**示例**")
                        gr.Examples(
                            examples=[
                                [
                                    "cookbooks/assets/agent_function_call/mobile_en_example.png",
                                    "Search for Musk in X and go to his homepage to open the first post.",
                                    "Step 1: I opened the X app from the home screen."
                                ],
                                [
                                    "cookbooks/assets/agent_function_call/mobile_zh_example.png",
                                    "Click on the first search result video.",
                                    "Step 1: Opening the Bilibili app.; Step 2: Clicking on the search bar to start searching for Qwen-VL videos.; Step 3: Type 'Qwen-VL' into the search bar.; Step 4: Clicking the '搜索' button to initiate the search for Qwen-VL videos."
                                ],
                            ],
                            inputs=[mobile_screenshot, mobile_instruction, mobile_history],
                            label="点击加载示例"
                        )
                        btn_mobile_agent = gr.Button("生成操作动作", variant="primary")
                    with gr.Column():
                        output_mobile_action = gr.Textbox(label="模型输出动作", lines=15)
                        output_mobile_image = gr.Image(type="pil", label="可视化截图（点击动作会标注）")

                def run_mobile_agent(img, instruction, history):
                    if img is None:
                        return "请上传移动设备截图", None
                    try:
                        action_text, display_image = mobile_agent_inference(img, instruction, history)
                        return action_text, display_image
                    except Exception as e:
                        return f"错误: {str(e)}", img

                btn_mobile_agent.click(
                    fn=run_mobile_agent,
                    inputs=[mobile_screenshot, mobile_instruction, mobile_history],
                    outputs=[output_mobile_action, output_mobile_image]
                )

            # Top-level: Video Understanding
            with gr.Tab("视频理解"):
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("视频输入", open=True):
                            video_file_input = gr.Video(
                                label="上传视频文件",
                                sources=["upload"]
                            )
                            gr.Markdown("**或使用视频URL**")
                            video_url_input = gr.Textbox(
                                label="视频URL",
                                placeholder="输入视频的URL地址",
                                value=""
                            )
                            gr.Markdown("**示例视频链接**：")
                            with gr.Row():
                                btn_video_example1 = gr.Button("📹 视频OCR示例", size="sm")
                                btn_video_example2 = gr.Button("🎬 活动检测示例", size="sm")
                        
                        video_prompt = gr.Textbox(
                            label="视频问题",
                            placeholder='e.g., "Describe the video content"',
                            value="Describe what happens in this video.",
                            lines=3
                        )
                        num_frames_slider = gr.Slider(
                            minimum=8,
                            maximum=128,
                            value=64,
                            step=8,
                            label="采样帧数（越多质量越好但速度越慢）"
                        )
                        gr.Markdown("**说明**")
                        gr.Markdown("""
                        上传视频文件或输入视频URL，模型会分析视频内容并回答问题。
                        
                        **支持的输入方式**：
                        - 本地视频文件上传
                        - 在线视频URL（http/https）
                        
                        **提示**：
                        - 采样帧数会影响处理速度和准确度，建议从64帧开始
                        - 可以点击示例按钮快速测试
                        """)
                        btn_video = gr.Button("分析视频", variant="primary")
                    with gr.Column():
                        output_video = gr.Textbox(label="分析结果", lines=20)
                        output_video_grid = gr.Image(type="pil", label="视频帧预览（采样帧网格）")

                # Set example video URLs
                def set_video_example1():
                    return "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4"
                
                def set_video_example2():
                    return "https://ofasys-multimodal-wlcb-3.oss-cn-wulanchabu.aliyuncs.com/sibo.ssb/datasets/cookbook/ead2e3f0e7f836c9ec51236befdaf2d843ac13a6.mp4"
                
                btn_video_example1.click(fn=set_video_example1, outputs=[video_url_input])
                btn_video_example2.click(fn=set_video_example2, outputs=[video_url_input])

                def run_video_understanding(video_file, video_url, prompt, num_frames):
                    # Priority: use URL if provided, otherwise use uploaded file
                    if video_url and video_url.strip():
                        video_input = video_url.strip()
                    elif video_file is not None:
                        video_input = video_file
                    else:
                        return "请上传视频文件或输入视频URL", None
                    
                    try:
                        result, grid_img = video_inference(
                            video_input,
                            prompt,
                            max_new_tokens=2048,
                            num_frames=int(num_frames)
                        )
                        return result, grid_img
                    except Exception as e:
                        import traceback
                        return f"错误: {str(e)}\n\n{traceback.format_exc()}", None

                btn_video.click(
                    fn=run_video_understanding,
                    inputs=[video_file_input, video_url_input, video_prompt, num_frames_slider],
                    outputs=[output_video, output_video_grid]
                )

        gr.Markdown("### 示例")
        gr.Markdown("上传包含文本的图片，测试OCR和文档解析功能。模型支持中文、英文、日文等多种语言。")
    
    return demo

if __name__ == "__main__":
    # Load model at startup
    load_model()
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
