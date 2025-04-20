import gradio as gr
import ai_gradio
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxx"  # added to set the credentials
## Create a browser automation interface
gr.load(
    name='browser:gpt-4-turbo',
    src=ai_gradio.registry,
    title='AI Browser Assistant',
    description='Let AI help with web tasks'
).launch(server_name='0.0.0.0')