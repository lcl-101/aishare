import argparse
import asyncio
import atexit
import base64
import json
import logging
import shutil
import os
import copy
import random
import re
import sys
import time
import tempfile
from concurrent.futures.process import BrokenProcessPool
from io import BytesIO
from urllib.parse import urlparse
from typing import List, Optional, Tuple
import threading

import httpx
import gradio as gr
import requests
from huggingface_hub import snapshot_download
from PIL import Image
from pypdf import PdfReader
from tqdm import tqdm

from ocrflux.check import (
    check_poppler_version,
    check_vllm_version,
    check_torch_gpu_available,
)
from ocrflux.image_utils import get_page_image, is_image
from ocrflux.table_format import trans_markdown_text
from ocrflux.metrics import MetricsKeeper, WorkerTracker
from ocrflux.prompts import PageResponse, build_page_to_markdown_prompt, build_element_merge_detect_prompt, build_html_table_merge_prompt
from ocrflux.work_queue import LocalWorkQueue, WorkQueue

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

vllm_logger = logging.getLogger("vllm")
vllm_logger.propagate = False

file_handler = logging.FileHandler("OCRFlux-debug.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
vllm_logger.addHandler(file_handler)

# Quiet logs from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Global variables for token statistics and server management
metrics = MetricsKeeper(window=60 * 5)
tracker = WorkerTracker()
vllm_server_task = None
server_ready = False

class Args:
    """Configuration class to replace argparse.Namespace"""
    def __init__(self):
        self.model = "checkpoints/OCRFlux-3B"
        self.port = 40078
        self.model_max_context = 16384
        self.model_chat_template = "qwen2-vl"
        self.target_longest_image_dim = 1024
        self.max_page_retries = 8
        self.max_page_error_rate = 0.004
        self.workers = 8
        self.pages_per_group = 500
        self.skip_cross_page_merge = False
        self.task = "pdf2markdown"

# Global args instance
args = Args()

def build_page_to_markdown_query(args, pdf_path: str, page_number: int, target_longest_image_dim: int, image_rotation: int = 0) -> dict:
    assert image_rotation in [0, 90, 180, 270], "Invalid image rotation provided in build_page_query"

    image = get_page_image(pdf_path, page_number, target_longest_image_dim=target_longest_image_dim, image_rotation=image_rotation)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_page_to_markdown_prompt()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "temperature": 0.0,
    }

def build_element_merge_detect_query(args,text_list_1,text_list_2) -> dict:
    image = Image.new('RGB', (28, 28), color='black')

    buffered = BytesIO()
    image.save(buffered, format="PNG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_element_merge_detect_prompt(text_list_1,text_list_2)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "temperature": 0.0,
    }

def build_html_table_merge_query(args,text_1,text_2) -> dict:
    image = Image.new('RGB', (28, 28), color='black')

    buffered = BytesIO()
    image.save(buffered, format="PNG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_html_table_merge_prompt(text_1,text_2)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "temperature": 0.0,
    }

async def apost(url, json_data):
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port or 80
    path = parsed_url.path or "/"

    writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)

        json_payload = json.dumps(json_data)
        request = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(json_payload)}\r\n"
            f"Connection: close\r\n\r\n"
            f"{json_payload}"
        )
        writer.write(request.encode())
        await writer.drain()

        # Read status line
        status_line = await reader.readline()
        if not status_line:
            raise ConnectionError("No response from server")
        status_parts = status_line.decode().strip().split(" ", 2)
        if len(status_parts) < 2:
            raise ValueError(f"Malformed status line: {status_line.decode().strip()}")
        status_code = int(status_parts[1])

        # Read headers
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode().partition(":")
            headers[key.strip().lower()] = value.strip()

        # Read response body
        if "content-length" in headers:
            body_length = int(headers["content-length"])
            response_body = await reader.readexactly(body_length)
        else:
            raise ConnectionError("Anything other than fixed content length responses are not implemented yet")

        return status_code, response_body
    except Exception as e:
        raise e
    finally:
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass

async def process_task(args, worker_id, task_name, task_args):
    COMPLETION_URL = f"http://localhost:{args.port}/v1/chat/completions"
    MAX_RETRIES = args.max_page_retries
    TEMPERATURE_BY_ATTEMPT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    exponential_backoffs = 0
    local_image_rotation = 0
    attempt = 0
    
    while attempt < MAX_RETRIES:
        if task_name == 'page_to_markdown':
            pdf_path,page_number = task_args
            query = build_page_to_markdown_query(args, pdf_path, page_number, args.target_longest_image_dim, image_rotation=local_image_rotation)
        elif task_name == 'element_merge_detect':
            text_list_1,text_list_2 = task_args
            query = build_element_merge_detect_query(args, text_list_1, text_list_2)
        elif task_name == 'html_table_merge':
            table_1,table_2 = task_args
            query = build_html_table_merge_query(args, table_1, table_2)
        
        query["temperature"] = TEMPERATURE_BY_ATTEMPT[
            min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)
        ]

        try:
            status_code, response_body = await apost(COMPLETION_URL, json_data=query)

            if status_code == 400:
                raise ValueError(f"Got BadRequestError from server: {response_body}, skipping this response")
            elif status_code == 500:
                raise ValueError(f"Got InternalServerError from server: {response_body}, skipping this response")
            elif status_code != 200:
                raise ValueError(f"Error http status {status_code}")

            base_response_data = json.loads(response_body)

            metrics.add_metrics(
                vllm_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
                vllm_output_tokens=base_response_data["usage"].get("completion_tokens", 0),
            )

            response_content = base_response_data["choices"][0]["message"]["content"]
            if task_name == 'page_to_markdown':
                model_response_json = json.loads(response_content)
                page_response = PageResponse(**model_response_json)
                if not page_response.is_rotation_valid and attempt < MAX_RETRIES - 1:
                    local_image_rotation = page_response.rotation_correction
                    raise ValueError(f"invalid_page rotation")
                try:         
                    return_data = trans_markdown_text(page_response.natural_text,"matrix2html")
                except:
                    if attempt < MAX_RETRIES - 1:
                        raise
                    else:
                        return_data = page_response.natural_text.replace("<t>","").replace("<l>","").replace("<lt>","")
                    
            elif task_name == 'element_merge_detect':
                pattern = r"\((\d+), (\d+)\)"
                matches = re.findall(pattern, response_content)
                return_data = [(int(x), int(y)) for x, y in matches]
            elif task_name == 'html_table_merge':
                if not (response_content.startswith("<table>") and response_content.endswith("</table>")):
                    raise ValueError("Response is not a table")
                return_data = response_content
            else:
                raise ValueError(f"Unknown task_name {task_name}")
            
            return return_data
        
        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            logger.warning(f"Client error on attempt {attempt} for {worker_id}: {type(e)} {e}")
            sleep_delay = 10 * (2**exponential_backoffs)
            exponential_backoffs += 1
            logger.info(f"Sleeping for {sleep_delay} seconds on {worker_id} to allow server restart")
            await asyncio.sleep(sleep_delay)
        except asyncio.CancelledError:
            logger.info(f"Process {worker_id} cancelled")
            raise
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt} for {worker_id}: {e}")
            attempt += 1
        except ValueError as e:
            logger.warning(f"ValueError on attempt {attempt} for {worker_id}: {type(e)} - {e}")
            attempt += 1
        except Exception as e:
            logger.exception(f"Unexpected error on attempt {attempt} for {worker_id}: {type(e)} - {e}")
            attempt += 1

    logger.error(f"Failed to process {worker_id} after {MAX_RETRIES} attempts.")
    return None

def postprocess_markdown_text(args, response_text, pdf_path, page_number):
    text_list = response_text.split("\n\n")
    new_text_list = []
    for text in text_list:
        if text.startswith("<Image>") and text.endswith("</Image>"):
            pass
        else:
            new_text_list.append(text)
    return "\n\n".join(new_text_list)

def bulid_document_text(page_to_markdown_result, element_merge_detect_result, html_table_merge_result):
    page_to_markdown_keys = list(page_to_markdown_result.keys())
    element_merge_detect_keys = list(element_merge_detect_result.keys())
    html_table_merge_keys = list(html_table_merge_result.keys())

    for page_1,page_2,elem_idx_1,elem_idx_2 in sorted(html_table_merge_keys,key=lambda x: -x[0]):
        page_to_markdown_result[page_1][elem_idx_1] = html_table_merge_result[(page_1,page_2,elem_idx_1,elem_idx_2)]
        page_to_markdown_result[page_2][elem_idx_2] = ''

    for page_1,page_2 in sorted(element_merge_detect_keys,key=lambda x: -x[0]):
        for elem_idx_1,elem_idx_2 in element_merge_detect_result[(page_1,page_2)]:
            if len(page_to_markdown_result[page_1][elem_idx_1]) == 0 or page_to_markdown_result[page_1][elem_idx_1][-1] == '-' or ('\u4e00' <= page_to_markdown_result[page_1][elem_idx_1][-1] <= '\u9fff'):
                page_to_markdown_result[page_1][elem_idx_1] = page_to_markdown_result[page_1][elem_idx_1] + '' + page_to_markdown_result[page_2][elem_idx_2]
            else:
                page_to_markdown_result[page_1][elem_idx_1] = page_to_markdown_result[page_1][elem_idx_1] + ' ' + page_to_markdown_result[page_2][elem_idx_2]
            page_to_markdown_result[page_2][elem_idx_2] = ''
    
    document_text_list = []
    for page in page_to_markdown_keys:
        page_text_list = [s for s in page_to_markdown_result[page] if s]
        document_text_list += page_text_list
    return "\n\n".join(document_text_list)

async def process_single_pdf(pdf_path: str, progress_callback=None):
    """Process a single PDF file"""
    logger.info(f"Start processing PDF: {pdf_path}")
    
    if pdf_path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
        except:
            logger.exception(f"Could not count number of pages for {pdf_path}")
            return None
    else:
        num_pages = 1
    
    logger.info(f"Got {num_pages} pages to process for {pdf_path}")

    try:
        tasks = []
        results = []
        
        # Process pages
        for page_num in range(1, num_pages + 1):
            if progress_callback:
                progress_callback(f"Processing page {page_num}/{num_pages}")
            
            result = await process_task(args, 0, task_name='page_to_markdown', task_args=(pdf_path, page_num))
            results.append(result)

        fallback_pages = []
        page_to_markdown_result = {}
        page_pairs = []
        
        for i, result in enumerate(results):
            if result is not None:
                page_number = i + 1
                page_to_markdown_result[page_number] = postprocess_markdown_text(args, result, pdf_path, page_number).split("\n\n")
                if page_number - 1 in page_to_markdown_result.keys():
                    page_pairs.append((page_number - 1, page_number))
            else:
                fallback_pages.append(i)
        
        num_fallback_pages = len(fallback_pages)

        if num_fallback_pages / num_pages > args.max_page_error_rate:
            logger.error(f"Document {pdf_path} has too many failed pages, discarding.")
            return None
        elif num_fallback_pages > 0:
            logger.warning(f"Document {pdf_path} processed with {num_fallback_pages} fallback pages.")

        if args.skip_cross_page_merge:
            page_texts = {}
            document_text_list = []
            sorted_page_keys = sorted(list(page_to_markdown_result.keys()))
            for page_number in sorted_page_keys:
                page_texts[str(page_number-1)] = "\n\n".join(page_to_markdown_result[page_number])
                document_text_list.append(page_texts[str(page_number-1)])
            document_text = "\n\n".join(document_text_list)
            return {
                "orig_path": pdf_path,
                "num_pages": num_pages,
                "document_text": document_text,
                "page_texts": page_texts,
                "fallback_pages": fallback_pages,
            }

        # Process cross-page merging
        if progress_callback:
            progress_callback("Processing cross-page merging...")
            
        element_merge_detect_result = {}
        table_pairs = []
        
        for page_1, page_2 in page_pairs:
            result = await process_task(args, 0, task_name='element_merge_detect', task_args=(page_to_markdown_result[page_1], page_to_markdown_result[page_2]))
            if result is not None:
                element_merge_detect_result[(page_1, page_2)] = result
                for elem_idx_1, elem_idx_2 in result:
                    text_1 = page_to_markdown_result[page_1][elem_idx_1]
                    text_2 = page_to_markdown_result[page_2][elem_idx_2]
                    if text_1.startswith("<table>") and text_1.endswith("</table>") and text_2.startswith("<table>") and text_2.endswith("</table>"):
                        table_pairs.append((page_1, page_2, elem_idx_1, elem_idx_2))

        # Process table merging
        tmp_page_to_markdown_result = copy.deepcopy(page_to_markdown_result)
        table_pairs = sorted(table_pairs, key=lambda x: -x[0])
        html_table_merge_result = {}
        
        for page_1, page_2, elem_idx_1, elem_idx_2 in table_pairs:
            result = await process_task(args, 0, task_name='html_table_merge', task_args=(tmp_page_to_markdown_result[page_1][elem_idx_1], tmp_page_to_markdown_result[page_2][elem_idx_2]))
            if result is not None:
                html_table_merge_result[(page_1, page_2, elem_idx_1, elem_idx_2)] = result
                tmp_page_to_markdown_result[page_1][elem_idx_1] = result

        page_texts = {}
        for page_number in page_to_markdown_result.keys():
            page_texts[str(page_number-1)] = "\n\n".join(page_to_markdown_result[page_number])
        
        document_text = bulid_document_text(page_to_markdown_result, element_merge_detect_result, html_table_merge_result)

        return {
            "orig_path": pdf_path,
            "num_pages": num_pages,
            "document_text": document_text,
            "page_texts": page_texts,
            "fallback_pages": fallback_pages,
        }
    except Exception as e:
        logger.exception(f"Exception in processing PDF {pdf_path}: {e}")
        return None

async def vllm_server_ready(port):
    max_attempts = 300
    delay_sec = 1
    url = f"http://localhost:{port}/v1/models"

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url)
                if response.status_code == 200:
                    logger.info("vllm server is ready.")
                    return True
        except Exception:
            pass
        await asyncio.sleep(delay_sec)
    return False

async def start_vllm_server():
    """Start the VLLM server"""
    global vllm_server_task, server_ready
    
    if vllm_server_task is not None and not vllm_server_task.done():
        return True
    
    model_name_or_path = args.model
    
    # Check if model exists
    if not os.path.exists(model_name_or_path):
        logger.error(f"Model not found at {model_name_or_path}")
        return False
    
    cmd = [
        "vllm",
        "serve",
        model_name_or_path,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.model_max_context),
        "--gpu_memory_utilization",
        str(0.8),
        "--trust-remote-code"
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        
        vllm_server_task = proc

        async def monitor_server():
            global server_ready
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    line = line.decode("utf-8").rstrip()
                    logger.info(f"VLLM: {line}")
                    if "The server is fired up and ready to roll!" in line or "Uvicorn running on" in line:
                        server_ready = True
                except Exception:
                    pass

        asyncio.create_task(monitor_server())
        
        # Wait for server to be ready
        ready = await vllm_server_ready(args.port)
        if ready:
            server_ready = True
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to start VLLM server: {e}")
        return False

def sync_start_server():
    """Synchronous wrapper for starting server"""
    global server_ready
    
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(start_vllm_server())
        except Exception as e:
            logger.error(f"Error in sync_start_server: {e}")
            return False
        finally:
            loop.close()
    
    # Run in a separate thread to avoid blocking the Gradio interface
    import threading
    result = [False]
    error = [None]
    
    def target():
        try:
            result[0] = run_in_thread()
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=300)  # 5 minutes timeout
    
    if thread.is_alive():
        return False
    
    if error[0]:
        logger.error(f"Thread error: {error[0]}")
        return False
        
    return result[0]

def process_pdf_document(file_path, progress=gr.Progress()):
    """Process a PDF document and return markdown text"""
    if not server_ready:
        return "Error: VLLM server is not ready. Please start the server first."
    
    if file_path is None:
        return "Please upload a PDF file."
    
    def update_progress(msg):
        progress(0.5, desc=msg)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(process_single_pdf(file_path, update_progress))
        if result:
            return result["document_text"]
        else:
            return "Failed to process the PDF document."
    except Exception as e:
        return f"Error processing PDF: {str(e)}"
    finally:
        loop.close()

def process_image_document(file_path, progress=gr.Progress()):
    """Process an image document and return markdown text"""
    if not server_ready:
        return "Error: VLLM server is not ready. Please start the server first."
    
    if file_path is None:
        return "Please upload an image file."
    
    def update_progress(msg):
        progress(0.5, desc=msg)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(process_single_pdf(file_path, update_progress))
        if result:
            return result["document_text"]
        else:
            return "Failed to process the image document."
    except Exception as e:
        return f"Error processing image: {str(e)}"
    finally:
        loop.close()

def process_directory(file_paths, progress=gr.Progress()):
    """Process multiple files from a directory"""
    if not server_ready:
        return "Error: VLLM server is not ready. Please start the server first."
    
    if not file_paths:
        return "Please upload files to process."
    
    results = []
    total_files = len(file_paths)
    
    def update_progress(msg):
        progress(0.5, desc=msg)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        for i, file_path in enumerate(file_paths):
            progress((i + 1) / total_files, desc=f"Processing file {i + 1}/{total_files}")
            
            result = loop.run_until_complete(process_single_pdf(file_path, update_progress))
            if result:
                results.append(f"## File: {os.path.basename(file_path)}\n\n{result['document_text']}")
            else:
                results.append(f"## File: {os.path.basename(file_path)}\n\nFailed to process this file.")
        
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Error processing files: {str(e)}"
    finally:
        loop.close()

def check_model_availability():
    """Check if the model is available"""
    model_path = args.model
    
    # Check local path first
    if os.path.exists(model_path):
        return True, f"✅ Model found at {model_path}"
    
    # Check if it's a HuggingFace model ID
    try:
        from huggingface_hub import repo_exists
        if repo_exists(model_path):
            return False, f"⚠️ Model {model_path} exists on HuggingFace but not locally. Will download on first use."
    except:
        pass
    
    return False, f"❌ Model not found at {model_path}"

def download_model_if_needed():
    """Download model if it doesn't exist locally"""
    model_path = args.model
    
    if os.path.exists(model_path):
        return True, "Model already exists locally"
    
    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading model {model_path}...")
        snapshot_download(repo_id=model_path, local_dir=model_path)
        return True, f"✅ Model downloaded successfully to {model_path}"
    except Exception as e:
        return False, f"❌ Failed to download model: {str(e)}"

def check_server_status():
    """Check if the VLLM server is running"""
    global server_ready
    
    if server_ready:
        return "✅ VLLM Server is running"
    
    # Try to ping the server
    try:
        import requests
        response = requests.get(f"http://localhost:{args.port}/v1/models", timeout=5)
        if response.status_code == 200:
            server_ready = True
            return "✅ VLLM Server is running"
    except:
        pass
    
    return "❌ VLLM Server is not running"

def get_initial_status():
    """Get initial server status"""
    return check_server_status()

def start_server_ui():
    """Start the VLLM server (UI wrapper)"""
    global server_ready
    
    if server_ready:
        return "✅ VLLM Server is already running!"
    
    # Check if model exists
    if not os.path.exists(args.model):
        return f"❌ Model not found at {args.model}. Please check the model path."
    
    try:
        logger.info("Starting VLLM server...")
        success = sync_start_server()
        if success and server_ready:
            return "✅ VLLM Server started successfully!"
        else:
            return "❌ Failed to start VLLM server. Please check the logs and model path."
    except Exception as e:
        logger.error(f"Exception in start_server_ui: {e}")
        return f"❌ Error starting server: {str(e)}"

# Create example files
def create_examples():
    """Create example files for demonstration"""
    examples_dir = "examples"
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
    
    # Create a simple example PDF content (we'll use a text file as placeholder)
    example_text = """# Sample Document

This is a sample document for testing OCRFlux.

## Table Example
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

## Text Example
Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
"""
    
    example_file = os.path.join(examples_dir, "sample.txt")
    with open(example_file, "w") as f:
        f.write(example_text)
    
    return [example_file]

# Create the Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    # Check dependencies
    try:
        check_poppler_version()
        check_vllm_version() 
        check_torch_gpu_available()
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
    
    with gr.Blocks(title="OCRFlux Web UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 OCRFlux Web UI")
        gr.Markdown("Extract text and tables from PDFs and images using OCRFlux-3B model")
        
        # Model status
        with gr.Row():
            model_available, model_msg = check_model_availability()
            model_status = gr.Textbox(
                label="Model Status",
                value=model_msg,
                interactive=False
            )
        
        # Server status
        with gr.Row():
            server_status = gr.Textbox(
                label="Server Status", 
                value=get_initial_status(), 
                interactive=False
            )
            with gr.Column():
                start_btn = gr.Button("🚀 Start VLLM Server", variant="primary")
                refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
        
        start_btn.click(
            fn=start_server_ui,
            outputs=server_status
        )
        
        refresh_btn.click(
            fn=check_server_status,
            outputs=server_status
        )
        
        with gr.Tabs():
            # Tab 1: Single PDF Document
            with gr.TabItem("📄 PDF Document"):
                gr.Markdown("### Upload a PDF document for text extraction")
                
                pdf_file = gr.File(
                    label="Upload PDF File",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                pdf_process_btn = gr.Button("Process PDF", variant="primary")
                pdf_output = gr.Textbox(
                    label="Extracted Text (Markdown)",
                    lines=20,
                    max_lines=50
                )
                
                pdf_process_btn.click(
                    fn=process_pdf_document,
                    inputs=pdf_file,
                    outputs=pdf_output,
                    show_progress=True
                )
            
            # Tab 2: Single Image
            with gr.TabItem("🖼️ Image Document"):
                gr.Markdown("### Upload an image for text extraction")
                
                image_file = gr.File(
                    label="Upload Image File",
                    file_types=[".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
                    type="filepath"
                )
                
                image_process_btn = gr.Button("Process Image", variant="primary")
                image_output = gr.Textbox(
                    label="Extracted Text (Markdown)",
                    lines=20,
                    max_lines=50
                )
                
                image_process_btn.click(
                    fn=process_image_document,
                    inputs=image_file,
                    outputs=image_output,
                    show_progress=True
                )
            
            # Tab 3: Multiple Files
            with gr.TabItem("📁 Multiple Files"):
                gr.Markdown("### Upload multiple PDF or image files for batch processing")
                
                files_input = gr.File(
                    label="Upload Multiple Files",
                    file_count="multiple",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
                    type="filepath"
                )
                
                batch_process_btn = gr.Button("Process All Files", variant="primary")
                batch_output = gr.Textbox(
                    label="Extracted Text from All Files (Markdown)",
                    lines=20,
                    max_lines=50
                )
                
                batch_process_btn.click(
                    fn=process_directory,
                    inputs=files_input,
                    outputs=batch_output,
                    show_progress=True
                )
        
        # Configuration section
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                target_dim = gr.Slider(
                    label="Target Image Dimension",
                    minimum=512,
                    maximum=2048,
                    value=1024,
                    step=64
                )
                max_retries = gr.Slider(
                    label="Max Page Retries",
                    minimum=1,
                    maximum=20,
                    value=8,
                    step=1
                )
            
            skip_merge = gr.Checkbox(
                label="Skip Cross-Page Merge",
                value=False
            )
            
            def update_settings(dim, retries, skip):
                args.target_longest_image_dim = dim
                args.max_page_retries = retries
                args.skip_cross_page_merge = skip
                return "Settings updated!"
            
            settings_msg = gr.Textbox(label="Settings Status", interactive=False)
            
            for component in [target_dim, max_retries, skip_merge]:
                component.change(
                    fn=update_settings,
                    inputs=[target_dim, max_retries, skip_merge],
                    outputs=settings_msg
                )
        
        # Footer
        gr.Markdown("""
        ---
        **Note:** Make sure to start the VLLM server before processing any files. 
        The first startup may take a few minutes to load the model.
        """)
    
    return demo

if __name__ == "__main__":
    # Create examples
    create_examples()
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
