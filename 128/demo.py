import gradio as gr
import json
import os
from typing import Any, List, Literal

from PIL import Image, ImageDraw
import requests
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
import torch
import re
from pydantic import BaseModel, Field

# --- Configuration ---
MODEL_ID = "checkpoints/Holo1-7B"

# --- Model and Processor Loading (Load once) ---
print(f"Loading model and processor for {MODEL_ID}...")
model = None
processor = None
model_loaded = False
load_error_message = ""

try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model_loaded = True
    print("Model and processor loaded successfully.")
except Exception as e:
    load_error_message = f"Error loading model/processor: {e}\n" \
                         "This might be due to network issues, an incorrect model ID, or missing dependencies (like flash_attention_2 if enabled by default in some config).\n" \
                         "Ensure you have a stable internet connection and the necessary libraries installed."
    print(load_error_message)

# --- app.py logic ---
def get_localization_prompt(pil_image: Image.Image, instruction: str) -> List[dict[str, Any]]:
    guidelines: str = "Localize an element on the GUI image according to my instructions and output a click position as Click(x, y) with x num pixels from the left edge and y num pixels from the top edge."
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"{guidelines}\n{instruction}"},
            ],
        }
    ]

def run_inference_localization_app(messages_for_template: List[dict[str, Any]], pil_image_for_processing: Image.Image) -> str:
    model.to("cuda")
    torch.cuda.set_device(0)
    text_prompt = processor.apply_chat_template(
        messages_for_template,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=[pil_image_for_processing],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    decoded_output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return decoded_output[0] if decoded_output else ""

def predict_click_location(input_pil_image: Image.Image, instruction: str):
    if not model_loaded or not processor or not model:
        return f"Model not loaded. Error: {load_error_message}", None
    if not input_pil_image:
        return "No image provided. Please upload an image.", None
    if not instruction or instruction.strip() == "":
        return "No instruction provided. Please type an instruction.", input_pil_image.copy().convert("RGB")
    image_proc_config = processor.image_processor
    try:
        resized_height, resized_width = smart_resize(
            input_pil_image.height,
            input_pil_image.width,
            factor=image_proc_config.patch_size * image_proc_config.merge_size,
            min_pixels=image_proc_config.min_pixels,
            max_pixels=image_proc_config.max_pixels,
        )
        resized_image = input_pil_image.resize(
            size=(resized_width, resized_height),
            resample=Image.Resampling.LANCZOS
        )
    except Exception as e:
        print(f"Error resizing image: {e}")
        return f"Error resizing image: {e}", input_pil_image.copy().convert("RGB")
    messages = get_localization_prompt(resized_image, instruction)
    try:
        coordinates_str = run_inference_localization_app(messages, resized_image)
    except Exception as e:
        print(f"Error during model inference: {e}")
        return f"Error during model inference: {e}", resized_image.copy().convert("RGB")
    output_image_with_click = resized_image.copy().convert("RGB")
    parsed_coords = None
    match = re.search(r"Click\((\d+),\s*(\d+)\)", coordinates_str)
    if match:
        try:
            x = int(match.group(1))
            y = int(match.group(2))
            parsed_coords = (x, y)
            draw = ImageDraw.Draw(output_image_with_click)
            radius = max(5, min(resized_width // 100, resized_height // 100, 15))
            bbox = (x - radius, y - radius, x + radius, y + radius)
            draw.ellipse(bbox, outline="red", width=max(2, radius // 4))
            print(f"Predicted and drawn click at: ({x}, {y}) on resized image ({resized_width}x{resized_height})")
        except Exception as e:
            print(f"Error drawing on image: {e}")
    else:
        print(f"Could not parse 'Click(x, y)' from model output: {coordinates_str}")
    return coordinates_str, output_image_with_click

example_image = None
example_instruction = "Select July 14th as the check-out date"
try:
    example_image_path = "examples/calendar_example.jpg"
    example_image = Image.open(example_image_path)
except Exception as e:
    print(f"Could not load example image from local path: {e}")
    try:
        example_image = Image.new("RGB", (200, 150), color="lightgray")
        draw = ImageDraw.Draw(example_image)
        draw.text((10, 10), "Example image\nfailed to load", fill="black")
    except:
        pass

# --- app1.py logic ---
SYSTEM_PROMPT = """Imagine you are a robot browsing the web, just like humans. Now you need to complete a task.\nIn each iteration, you will receive an Observation that includes the last  screenshots of a web browser and the current memory of the agent.\nYou have also information about the step that the agent is trying to achieve to solve the task.\nCarefully analyze the visual information to identify what to do, then follow the guidelines to choose the following action.\nYou should detail your thought (i.e. reasoning steps) before taking the action.\nAlso detail in the notes field of the action the extracted information relevant to solve the task.\nOnce you have enough information in the notes to answer the task, return an answer action with the detailed answer in the notes field.\nThis will be evaluated by an evaluator and should match all the criteria or requirements of the task.\nGuidelines:\n- store in the notes all the relevant information to solve the task that fulfill the task criteria. Be precise\n- Use both the task and the step information to decide what to do\n- if you want to write in a text field and the text field already has text, designate the text field by the text it contains and its type\n- If there is a cookies notice, always accept all the cookies first\n- The observation is the screenshot of the current page and the memory of the agent.\n- If you see relevant information on the screenshot to answer the task, add it to the notes field of the action.\n- If there is no relevant information on the screenshot to answer the task, add an empty string to the notes field of the action.\n- If you see buttons that allow to navigate directly to relevant information, like jump to ... or go to ... , use them to navigate faster.\n- In the answer action, give as many details a possible relevant to answering the task.\n- if you want to write, don't click before. Directly use the write action\n- to write, identify the web element which is type and the text it already contains\n- If you want to use a search bar, directly write text in the search bar\n- Don't scroll too much. Don't scroll if the number of scrolls is greater than 3\n- Don't scroll if you are at the end of the webpage\n- Only refresh if you identify a rate limit problem\n- If you are looking for a single flights, click on round-trip to select 'one way'\n- Never try to login, enter email or password. If there is a need to login, then go back.\n- If you are facing a captcha on a website, try to solve it.\n- if you have enough information in the screenshot and in the notes to answer the task, return an answer action with the detailed answer in the notes field\n- The current date is {timestamp}.\n# <output_json_format>\n# ```json\n# {output_format}\n# ```\n# </output_json_format>\n"""

class ClickElementAction(BaseModel):
    action: Literal["click_element"] = Field(description="Click at absolute coordinates of a web element")
    element: str = Field(description="text description of the element")
    x: int = Field(description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(description="The y coordinate, number of pixels from the top edge.")
    def log(self):
        return f"I have clicked on the element '{self.element}' at absolute coordinates {self.x}, {self.y}"

class WriteElementAction(BaseModel):
    action: Literal["write_element_abs"] = Field(description="Write content at absolute coordinates of a web page")
    content: str = Field(description="Content to write")
    element: str = Field(description="Text description of the element")
    x: int = Field(description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(description="The y coordinate, number of pixels from the top edge.")
    def log(self):
        return f"I have written '{self.content}' in the element '{self.element}' at absolute coordinates {self.x}, {self.y}"

class ScrollAction(BaseModel):
    action: Literal["scroll"] = Field(description="Scroll the page or a specific element")
    direction: Literal["down", "up", "left", "right"] = Field(description="The direction to scroll in")
    def log(self):
        return f"I have scrolled {self.direction}"

class GoBackAction(BaseModel):
    action: Literal["go_back"] = Field(description="Navigate to the previous page")
    def log(self):
        return "I have gone back to the previous page"

class RefreshAction(BaseModel):
    action: Literal["refresh"] = Field(description="Refresh the current page")
    def log(self):
        return "I have refreshed the page"

class GotoAction(BaseModel):
    action: Literal["goto"] = Field(description="Goto a particular URL")
    url: str = Field(description="A url starting with http:// or https://")
    def log(self):
        return f"I have navigated to the URL {self.url}"

class WaitAction(BaseModel):
    action: Literal["wait"] = Field(description="Wait for a particular amount of time")
    seconds: int = Field(default=2, ge=0, le=10, description="The number of seconds to wait")
    def log(self):
        return f"I have waited for {self.seconds} seconds"

class RestartAction(BaseModel):
    action: Literal["restart"] = "restart"
    def log(self):
        return "I have restarted the task from the beginning"

class AnswerAction(BaseModel):
    action: Literal["answer"] = "answer"
    content: str = Field(description="The answer content")
    def log(self):
        return f"I have answered the task with '{self.content}'"

ActionSpace = (
    ClickElementAction
    | WriteElementAction
    | ScrollAction
    | GoBackAction
    | RefreshAction
    | WaitAction
    | RestartAction
    | AnswerAction
    | GotoAction
)

class NavigationStep(BaseModel):
    note: str = Field(
        default="",
        description="Task-relevant information extracted from the previous observation. Keep empty if no new info.",
    )
    thought: str = Field(description="Reasoning about next steps (<4 lines)")
    action: ActionSpace = Field(description="Next action to take")

def get_navigation_prompt(task, image, step=1):
    system_prompt = SYSTEM_PROMPT.format(
        output_format=NavigationStep.model_json_schema(),
        timestamp="2025-06-04 14:16:03",
    )
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<task>\n{task}\n</task>\n"},
                {"type": "text", "text": f"<observation step={step}>\n"},
                {"type": "text", "text": "<screenshot>\n"},
                {"type": "image", "image": image},
                {"type": "text", "text": "\n</screenshot>\n"},
                {"type": "text", "text": "\n</observation>\n"},
            ],
        },
    ]

def run_inference_localization_app1(messages_for_template: List[dict[str, Any]], pil_image_for_processing: Image.Image) -> str:
    model.to("cuda")
    torch.cuda.set_device(0)
    text_prompt = processor.apply_chat_template(messages_for_template, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[pil_image_for_processing],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    decoded_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded_output[0] if decoded_output else ""

def navigate(input_pil_image: Image.Image, task: str) -> str:
    if not model_loaded or not processor or not model:
        return f"Model not loaded. Error: {load_error_message}", None
    if not input_pil_image:
        return "No image provided. Please upload an image.", None
    if not task or task.strip() == "":
        return "No task provided. Please type an task.", input_pil_image.copy().convert("RGB")
    image_proc_config = processor.image_processor
    try:
        resized_height, resized_width = smart_resize(
            input_pil_image.height,
            input_pil_image.width,
            factor=image_proc_config.patch_size * image_proc_config.merge_size,
            min_pixels=image_proc_config.min_pixels,
            max_pixels=image_proc_config.max_pixels,
        )
        resized_image = input_pil_image.resize(
            size=(resized_width, resized_height),
            resample=Image.Resampling.LANCZOS,
        )
    except Exception as e:
        print(f"Error resizing image: {e}")
        return f"Error resizing image: {e}", input_pil_image.copy().convert("RGB")
    prompt = get_navigation_prompt(task, resized_image, step=1)
    try:
        navigation_str = run_inference_localization_app1(prompt, resized_image)
    except Exception as e:
        print(f"Error during model inference: {e}")
        return f"Error during model inference: {e}", resized_image.copy().convert("RGB")
    return navigation_str

example_task = "Book a hotel in Paris on August 3rd for 3 nights"
article = f"""
<p style='text-align: center'>
Model: <a href='https://huggingface.co/{MODEL_ID}' target='_blank'>{MODEL_ID}</a> by HCompany | \
Paper: <a href='https://cdn.prod.website-files.com/67e2dbd9acff0c50d4c8a80c/683ec8095b353e8b38317f80_h_tech_report_v1.pdf' target='_blank'>HCompany Tech Report</a> |
Blog: <a href='https://www.hcompany.ai/surfer-h' target='_blank'>Surfer-H Blog Post</a>
</p>
"""

# --- Gradio Interface with Tabs ---
if not model_loaded:
    with gr.Blocks() as demo:
        gr.Markdown(f"# <center>⚠️ Error: Model Failed to Load ⚠️</center>")
        gr.Markdown(f"<center>{load_error_message}</center>")
        gr.Markdown("<center>Please check the console output for more details. Reloading the space might help if it's a temporary issue.</center>")
else:
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Tab("Localization (app.py)"):
            gr.Markdown(f"<h1 style='text-align: center;'>Holo1-7B: Action VLM Localization Demo</h1>")
            with gr.Row():
                with gr.Column(scale=1):
                    input_image_component = gr.Image(type="pil", label="Input UI Image", height=400)
                    instruction_component = gr.Textbox(
                        label="Instruction",
                        placeholder="e.g., Click the 'Login' button",
                        info="Type the action you want the model to localize on the image."
                    )
                    submit_button = gr.Button("Localize Click", variant="primary")
                with gr.Column(scale=1):
                    output_coords_component = gr.Textbox(label="Predicted Coordinates (Format: Click(x,y))", interactive=False)
                    output_image_component = gr.Image(type="pil", label="Image with Predicted Click Point", height=400, interactive=False)
            if example_image:
                gr.Examples(
                    examples=[[example_image, example_instruction]],
                    inputs=[input_image_component, instruction_component],
                    outputs=[output_coords_component, output_image_component],
                    fn=predict_click_location,
                    cache_examples="lazy",
                )
            gr.Markdown(article)
            submit_button.click(
                fn=predict_click_location,
                inputs=[input_image_component, instruction_component],
                outputs=[output_coords_component, output_image_component]
            )
        with gr.Tab("Navigation (app1.py)"):
            gr.Markdown(f"<h1 style='text-align: center;'>Holo1-7B: Action VLM Navigation Demo</h1>")
            with gr.Row():
                with gr.Column(scale=1):
                    input_image_component2 = gr.Image(type="pil", label="Input UI Image", height=400)
                    task_component = gr.Textbox(
                        label="task",
                        placeholder="e.g., Book a hotel in Paris on August 3rd for 3 nights",
                        info="Type the task you want the model to complete."
                    )
                    submit_button2 = gr.Button("Navigate", variant="primary")
                with gr.Column(scale=1):
                    output_coords_component2 = gr.Textbox(label="Navigation Step", interactive=False)
            if example_image:
                gr.Examples(
                    examples=[[example_image, example_task]],
                    inputs=[input_image_component2, task_component],
                    outputs=[output_coords_component2],
                    fn=navigate,
                    cache_examples="lazy",
                )
            gr.Markdown(article)
            submit_button2.click(
                fn=navigate,
                inputs=[input_image_component2, task_component],
                outputs=[output_coords_component2],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
