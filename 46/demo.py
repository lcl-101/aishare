import os  # added for setting environment variable
os.environ["GOOGLE_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxx"  # added to set the credentials

import gradio as gr
import ai_gradio
import gradio_client.utils as utils

## Patch the get_type function to handle boolean schemas
_original_get_type = utils.get_type
def patched_get_type(schema):
    if isinstance(schema, bool):
        return bool
    return _original_get_type(schema)
utils.get_type = patched_get_type

# Patch _json_schema_to_python_type to handle boolean schemas
_original_json_schema = utils._json_schema_to_python_type
def patched_json_schema(schema, defs=None):
    if isinstance(schema, bool):
        return bool
    return _original_json_schema(schema, defs)
utils._json_schema_to_python_type = patched_json_schema

import gradio as gr
import ai_gradio

with gr.Blocks() as demo:
    with gr.Tab("Text"):
        gr.load('gemini:gemini-1.5-flash', src=ai_gradio.registry)
    with gr.Tab("Vision"):
        gr.load('gemini:gemini-1.5-flash', src=ai_gradio.registry)
    with gr.Tab("Code"):
        gr.load('gemini:gemini-2.0-flash-thinking-exp-1219', src=ai_gradio.registry)

demo.launch(server_name='0.0.0.0', share=True)