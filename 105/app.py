from transformers import AutoProcessor, ShieldGemma2ForImageClassification
from PIL import Image
import requests
import torch
import gradio as gr
model_id = "checkpoints/shieldgemma-2-4b-it"
model = ShieldGemma2ForImageClassification.from_pretrained(model_id).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

def infer(image, policies, policy_descriptions):
  policies = policies.split(";")
  policy_descriptions = policy_descriptions.split(";")
  custom_policies = dict(zip(policies, policy_descriptions))

  inputs = processor(
      images=[image],
      custom_policies=custom_policies,
      policies=policies,
      return_tensors="pt",
  ).to(model.device)
  
  with torch.inference_mode():
      output = model(**inputs)
      
  
  outs = {}

  for idx, policy in enumerate(output.probabilities.cpu()):
    yes_prob = policy[0]
    no_prob = policy[1]

    outs[f"Yes for {policies[idx]}"] = yes_prob
    outs[f"No for {policies[idx]}"] = no_prob

  return outs


IMG = """
  <img src='https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/SheildGemma2_WagtailBlog_RD1_V01a.original.png' />
  """


with gr.Blocks() as demo:
  gr.Markdown(IMG)
  gr.Markdown("## ShieldGemma2 for Multimodal Safety")
  gr.Markdown("ShieldGemma2 is a safety moderation model for vision language models. It can detect unsafe images. To use it, simply input an image, and provide policies. A policy is a description of what should be detected, and also provide names of policies.")
  gr.Markdown("You can test it with an image of your choice and example policies provided below.")
 
  with gr.Row():
    with gr.Column():
      image = gr.Image(type="pil")
      policies = gr.Textbox(label = "Name of policies separated by semicolon")
      policy_descriptions = gr.Textbox(label = "Description of policies separated by semicolon", lines=10)
      btn = gr.Button("Submit")
    with gr.Column():
      outputs = gr.Label()
    
    btn.click(fn=infer, inputs=[image, policies, policy_descriptions], outputs=outputs)
  gr.Examples(examples = [["Sexually Explicit content;Dangerous content;Violent content", "The image shall not contain content that depicts explicit or graphic sexual acts.; The image shall not contain content that facilitates or encourages activities that could cause real-world harm (e.g., building firearms and explosive devices, promotion of terrorism, instructions for suicide).;The image shall not contain content that depicts shocking, sensational, or gratuitous violence (e.g., excessive blood and gore, gratuitous violence against animals, extreme injury or moment of death)."]],
              inputs = [policies, policy_descriptions])


demo.launch(server_name="0.0.0.0")