"""
Qwen3-VL Multimodal RAG Web Application

基于 Gradio 的多模态 RAG 演示程序，支持：
- PDF 文档上传和处理
- 图片文档上传
- 文本查询检索
- 使用 Qwen3-VL-Embedding 进行嵌入
- 使用 Qwen3-VL-Reranker 进行重排序
- 使用 Qwen3-VL 生成答案
"""

import os
import sys
import torch
import numpy as np
import gradio as gr
import requests
import logging
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Tuple

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.qwen3_vl_embedding import Qwen3VLEmbedder
from models.qwen3_vl_reranker import Qwen3VLReranker
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 全局配置 ====================
EXAMPLE_PDF_URL = "https://climate.ec.europa.eu/system/files/2018-06/youth_magazine_en.pdf"
EXAMPLE_PDF_PATH = "data/examples/climate_document.pdf"
EXAMPLE_IMAGES_DIR = "data/examples/document_pages"
TEMP_DIR = "temp"

# 模型路径配置 (优先使用本地路径)
EMBEDDING_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH", 
    "checkpoints/Qwen3-VL-Embedding-2B"
)
RERANKER_MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH", 
    "checkpoints/Qwen3-VL-Reranker-2B"
)
VLM_MODEL_PATH = os.environ.get(
    "VLM_MODEL_PATH", 
    "checkpoints/Qwen3-VL-2B-Instruct"
)

# 如果本地路径不存在，使用 HuggingFace 模型
def get_model_path(local_path: str, hf_path: str) -> str:
    if os.path.exists(local_path):
        logger.info(f"使用本地模型: {local_path}")
        return local_path
    logger.info(f"本地模型不存在，使用 HuggingFace: {hf_path}")
    return hf_path

# ==================== 工具函数 ====================
def ensure_dir(path: str):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def download_pdf(url: str, save_path: str) -> str:
    """下载 PDF 文件"""
    ensure_dir(os.path.dirname(save_path))
    if os.path.exists(save_path):
        logger.info(f"PDF 已存在: {save_path}")
        return save_path
    
    logger.info(f"下载 PDF: {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)
    logger.info(f"PDF 保存至: {save_path}")
    return save_path


def pdf_to_images(pdf_path: str, output_dir: str = None) -> List[str]:
    """将 PDF 转换为图片"""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError("请安装 pdf2image: pip install pdf2image")
    
    if output_dir is None:
        output_dir = EXAMPLE_IMAGES_DIR
    ensure_dir(output_dir)
    
    logger.info(f"转换 PDF 为图片: {pdf_path}")
    images = convert_from_path(pdf_path)
    
    image_paths = []
    for idx, img in enumerate(images):
        img_path = os.path.join(output_dir, f"page_{idx:03d}.png")
        img.save(img_path)
        image_paths.append(img_path)
    
    logger.info(f"共转换 {len(image_paths)} 页")
    return image_paths


def download_example_data():
    """下载示例数据"""
    logger.info("正在准备示例数据...")
    
    # 检查是否已有示例图片
    if os.path.exists(EXAMPLE_IMAGES_DIR):
        existing_images = list(Path(EXAMPLE_IMAGES_DIR).glob("*.png"))
        if len(existing_images) > 0:
            logger.info(f"示例数据已存在，共 {len(existing_images)} 页")
            return sorted([str(p) for p in existing_images])
    
    # 下载并转换 PDF
    try:
        pdf_path = download_pdf(EXAMPLE_PDF_URL, EXAMPLE_PDF_PATH)
        image_paths = pdf_to_images(pdf_path, EXAMPLE_IMAGES_DIR)
        return image_paths
    except Exception as e:
        logger.error(f"下载示例数据失败: {e}")
        return []


# ==================== RAG 系统类 ====================
class MultimodalRAG:
    def __init__(self):
        self.embedder = None
        self.reranker = None
        self.vlm_model = None
        self.vlm_processor = None
        
        self.document_images: List[str] = []
        self.document_embeddings = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
    
    def load_embedder(self):
        """加载 Embedding 模型"""
        if self.embedder is not None:
            return
        
        model_path = get_model_path(
            EMBEDDING_MODEL_PATH, 
            "Qwen/Qwen3-VL-Embedding-2B"
        )
        logger.info(f"加载 Embedding 模型: {model_path}")
        self.embedder = Qwen3VLEmbedder(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Embedding 模型加载完成")
    
    def unload_embedder(self):
        """卸载 Embedding 模型以节省显存"""
        if self.embedder is not None:
            del self.embedder
            self.embedder = None
            torch.cuda.empty_cache()
            logger.info("Embedding 模型已卸载")
    
    def load_reranker(self):
        """加载 Reranker 模型"""
        if self.reranker is not None:
            return
        
        model_path = get_model_path(
            RERANKER_MODEL_PATH,
            "Qwen/Qwen3-VL-Reranker-2B"
        )
        logger.info(f"加载 Reranker 模型: {model_path}")
        self.reranker = Qwen3VLReranker(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Reranker 模型加载完成")
    
    def unload_reranker(self):
        """卸载 Reranker 模型"""
        if self.reranker is not None:
            del self.reranker
            self.reranker = None
            torch.cuda.empty_cache()
            logger.info("Reranker 模型已卸载")
    
    def load_vlm(self):
        """加载 VLM 生成模型"""
        if self.vlm_model is not None:
            return
        
        model_path = get_model_path(
            VLM_MODEL_PATH,
            "Qwen/Qwen3-VL-2B-Instruct"
        )
        logger.info(f"加载 VLM 模型: {model_path}")
        self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.vlm_processor = AutoProcessor.from_pretrained(model_path)
        logger.info("VLM 模型加载完成")
    
    def unload_vlm(self):
        """卸载 VLM 模型"""
        if self.vlm_model is not None:
            del self.vlm_model
            del self.vlm_processor
            self.vlm_model = None
            self.vlm_processor = None
            torch.cuda.empty_cache()
            logger.info("VLM 模型已卸载")
    
    def index_documents(self, image_paths: List[str]) -> str:
        """索引文档图片"""
        if not image_paths:
            return "❌ 没有文档图片可索引"
        
        self.document_images = image_paths
        
        # 加载 embedder
        self.load_embedder()
        
        # 构建输入
        document_inputs = [{"image": img_path} for img_path in image_paths]
        
        logger.info(f"正在为 {len(image_paths)} 个文档生成嵌入...")
        self.document_embeddings = self.embedder.process(document_inputs)
        
        # 卸载以节省显存
        self.unload_embedder()
        
        return f"✅ 成功索引 {len(image_paths)} 个文档页面"
    
    def retrieve_top_k(
        self, 
        query_embedding: np.ndarray, 
        k: int = 3
    ) -> Tuple[List[int], List[float]]:
        """检索 Top-K 相关文档"""
        if self.document_embeddings is None:
            return [], []
        
        doc_emb = self.document_embeddings
        if torch.is_tensor(doc_emb):
            doc_emb = doc_emb.cpu().numpy()
        if torch.is_tensor(query_embedding):
            query_embedding = query_embedding.cpu().numpy()
        
        similarity_scores = query_embedding @ doc_emb.T
        top_k_indices = np.argsort(similarity_scores)[-k:][::-1]
        top_k_scores = similarity_scores[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()
    
    def search(
        self, 
        query: str, 
        top_k: int = 3, 
        use_reranker: bool = True
    ) -> Tuple[List[Tuple[str, float]], str]:
        """搜索相关文档"""
        if not self.document_images or self.document_embeddings is None:
            return [], "❌ 请先索引文档"
        
        # 加载 embedder 并生成查询嵌入
        self.load_embedder()
        query_inputs = [{"text": query}]
        query_embedding = self.embedder.process(query_inputs)
        self.unload_embedder()
        
        # 检索
        top_indices, top_scores = self.retrieve_top_k(
            query_embedding[0], k=top_k
        )
        
        results = []
        if use_reranker and len(top_indices) > 0:
            # 使用 Reranker 重排序
            self.load_reranker()
            
            reranker_inputs = {
                "instruction": "Retrieve pages relevant to the user's query.",
                "query": {"text": query},
                "documents": [{"image": self.document_images[idx]} for idx in top_indices],
            }
            reranker_scores = self.reranker.process(reranker_inputs)
            
            self.unload_reranker()
            
            # 按重排序分数排序
            sorted_results = sorted(
                zip(top_indices, reranker_scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            for idx, score in sorted_results:
                results.append((self.document_images[idx], score))
            
            status = f"✅ 检索完成（使用 Reranker 重排序）"
        else:
            for idx, score in zip(top_indices, top_scores):
                results.append((self.document_images[idx], score))
            status = f"✅ 检索完成"
        
        return results, status
    
    def generate_answer(
        self, 
        query: str, 
        image_path: str, 
        max_tokens: int = 512
    ) -> str:
        """使用 VLM 生成答案"""
        self.load_vlm()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {
                        "type": "text",
                        "text": f"Based on this document page, please answer the following question:\n\n{query}"
                    },
                ],
            }
        ]
        
        inputs = self.vlm_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.vlm_model.device)
        
        generated_ids = self.vlm_model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0]


# ==================== Gradio 界面 ====================
# 全局 RAG 实例
rag = MultimodalRAG()


def load_example_documents():
    """加载示例文档"""
    image_paths = download_example_data()
    if not image_paths:
        return None, "❌ 加载示例数据失败"
    
    # 只使用部分页面作为示例
    image_paths = image_paths[4:10]  # 第5-10页
    
    status = rag.index_documents(image_paths)
    
    # 返回图片用于显示
    gallery_images = [(path, f"Page {i+1}") for i, path in enumerate(image_paths)]
    return gallery_images, status


def upload_documents(files):
    """上传并索引文档"""
    if not files:
        return None, "❌ 请上传文档"
    
    image_paths = []
    ensure_dir(TEMP_DIR)
    
    for file in files:
        file_path = file.name
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            # 转换 PDF
            pdf_images = pdf_to_images(file_path, TEMP_DIR)
            image_paths.extend(pdf_images)
        elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
            image_paths.append(file_path)
        else:
            logger.warning(f"不支持的文件格式: {ext}")
    
    if not image_paths:
        return None, "❌ 没有可处理的文档"
    
    status = rag.index_documents(image_paths)
    gallery_images = [(path, f"Page {i+1}") for i, path in enumerate(image_paths)]
    return gallery_images, status


def search_documents(query: str, top_k: int, use_reranker: bool):
    """搜索文档"""
    if not query.strip():
        return None, "❌ 请输入查询"
    
    results, status = rag.search(query, top_k=top_k, use_reranker=use_reranker)
    
    if not results:
        return None, status
    
    gallery_images = [
        (path, f"Score: {score:.4f}") 
        for path, score in results
    ]
    return gallery_images, status


def generate_answer(query: str, selected_image: str):
    """生成答案"""
    if not query.strip():
        return "❌ 请输入查询"
    
    if not selected_image:
        # 使用第一个搜索结果
        if rag.document_images:
            selected_image = rag.document_images[0]
        else:
            return "❌ 请先索引文档并选择一个页面"
    
    try:
        answer = rag.generate_answer(query, selected_image)
        return answer
    except Exception as e:
        logger.error(f"生成答案失败: {e}")
        return f"❌ 生成失败: {str(e)}"


def select_image_for_generation(evt: gr.SelectData, gallery):
    """从搜索结果中选择图片用于生成"""
    if gallery and evt.index < len(gallery):
        return gallery[evt.index][0]
    return None


# 创建 Gradio 界面
def create_ui():
    with gr.Blocks(
        title="Qwen3-VL Multimodal RAG",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🔍 Qwen3-VL Multimodal RAG Demo
        
        使用 Qwen3-VL 模型系列实现多模态检索增强生成 (RAG)：
        - **Qwen3-VL-Embedding**: 文档和查询嵌入
        - **Qwen3-VL-Reranker**: 搜索结果重排序  
        - **Qwen3-VL-Instruct**: 基于文档生成答案
        """)
        
        selected_image_path = gr.State(None)
        
        with gr.Tabs():
            # ========== 文档索引 Tab ==========
            with gr.TabItem("📚 文档索引"):
                gr.Markdown("### 上传文档或加载示例数据")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        example_btn = gr.Button(
                            "🌍 加载示例数据 (气候变化文档)", 
                            variant="primary"
                        )
                        
                        gr.Markdown("---")
                        
                        upload_files = gr.File(
                            label="上传 PDF 或图片",
                            file_count="multiple",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
                        )
                        upload_btn = gr.Button("📤 上传并索引")
                    
                    with gr.Column(scale=2):
                        doc_gallery = gr.Gallery(
                            label="已索引文档",
                            columns=3,
                            height=400,
                            object_fit="contain",
                        )
                        index_status = gr.Textbox(
                            label="状态", 
                            interactive=False
                        )
            
            # ========== 搜索 Tab ==========
            with gr.TabItem("🔎 搜索"):
                gr.Markdown("### 输入查询进行多模态检索")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        query_input = gr.Textbox(
                            label="查询",
                            placeholder="例如：How much did the world temperature change?",
                            lines=2,
                        )
                        top_k_slider = gr.Slider(
                            minimum=1, 
                            maximum=10, 
                            value=3, 
                            step=1,
                            label="返回结果数量 (Top-K)"
                        )
                        use_reranker_checkbox = gr.Checkbox(
                            label="使用 Reranker 重排序",
                            value=True
                        )
                        search_btn = gr.Button("🔍 搜索", variant="primary")
                    
                    with gr.Column(scale=2):
                        search_gallery = gr.Gallery(
                            label="搜索结果 (点击选择用于生成答案)",
                            columns=3,
                            height=400,
                            object_fit="contain",
                            allow_preview=True,
                        )
                        search_status = gr.Textbox(
                            label="状态",
                            interactive=False
                        )
            
            # ========== 生成答案 Tab ==========
            with gr.TabItem("💬 生成答案"):
                gr.Markdown("### 基于检索到的文档生成答案")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gen_query_input = gr.Textbox(
                            label="问题",
                            placeholder="输入你想问的问题...",
                            lines=3,
                        )
                        selected_image_display = gr.Image(
                            label="选中的文档页面",
                            height=300,
                        )
                        generate_btn = gr.Button("✨ 生成答案", variant="primary")
                    
                    with gr.Column(scale=1):
                        answer_output = gr.Textbox(
                            label="生成的答案",
                            lines=15,
                            interactive=False,
                        )
        
        # ========== 事件绑定 ==========
        example_btn.click(
            fn=load_example_documents,
            outputs=[doc_gallery, index_status],
        )
        
        upload_btn.click(
            fn=upload_documents,
            inputs=[upload_files],
            outputs=[doc_gallery, index_status],
        )
        
        search_btn.click(
            fn=search_documents,
            inputs=[query_input, top_k_slider, use_reranker_checkbox],
            outputs=[search_gallery, search_status],
        )
        
        # 点击搜索结果选择图片
        search_gallery.select(
            fn=lambda evt, gallery: (gallery[evt.index][0] if gallery else None, gallery[evt.index][0] if gallery else None),
            inputs=[search_gallery],
            outputs=[selected_image_path, selected_image_display],
        )
        
        # 同步查询到生成页面
        query_input.change(
            fn=lambda x: x,
            inputs=[query_input],
            outputs=[gen_query_input],
        )
        
        generate_btn.click(
            fn=generate_answer,
            inputs=[gen_query_input, selected_image_path],
            outputs=[answer_output],
        )
        
        gr.Markdown("""
        ---
        ### 使用说明
        1. **文档索引**: 上传 PDF/图片文档，或加载示例数据
        2. **搜索**: 输入查询，系统会检索最相关的文档页面
        3. **生成答案**: 点击搜索结果选择页面，然后生成答案
        
        ### 模型配置
        - 默认使用 `checkpoints/` 目录下的本地模型
        - 可通过环境变量配置模型路径:
          - `EMBEDDING_MODEL_PATH`
          - `RERANKER_MODEL_PATH`  
          - `VLM_MODEL_PATH`
        """)
    
    return demo


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 启动时下载示例数据
    logger.info("正在初始化...")
    download_example_data()
    
    # 创建并启动 Gradio
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
