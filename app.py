# app/main.py
import gradio as gr
import os
import zipfile

from pathlib import Path
from utils.logger import logger
from config.settings import settings
from qdrant_client import QdrantClient
from core.retrieval.retriever import Retriever
from ingestions.ingestion import IngestionService

# --- Initialize global services ---
logger.info("--- Initializing Global Services (Upload-Only Mode) ---")
try:
    # Create ONE QdrantClient only for sharing
    qdrant_db_path = os.path.join(settings.DATA_DIR, "qdrant_data")
    shared_qdrant_client = QdrantClient(path=qdrant_db_path)
    logger.info("Shared Qdrant client initialized.")

    ingestion_service = IngestionService(client=shared_qdrant_client)
    retriever_instance = Retriever(client=shared_qdrant_client)
    
    logger.info("All services initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize global services: {e}")
    raise RuntimeError(f"Could not initialize services. Please check logs. Error: {e}")

def upload_handler(zip_path: str, progress=gr.Progress()):
    progress(0, desc="🚀 Starting upload process...")
    
    if not zip_path:
        return "Error: No file uploaded"
    
    if not zip_path.endswith(".zip"):
        return "Error: Please upload a zip file"
    
    progress(0.05, desc="📦 Extracting ZIP file...")
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(settings.RAW_DATA_DIR)
    except zipfile.BadZipFile:
        return "Invalid ZIP file."
    except Exception as e:
        return f"Error during extraction: {str(e)}"
    
    progress(0.15, desc="🔍 Scanning for files...")
    
    logger.info(f"Handling upload of {len(settings.RAW_DATA_DIR)} items (files/folders)...")

    # --- Retrieve all file path from input ---
    path = Path(settings.RAW_DATA_DIR)
    all_temp_file_paths = list(path.rglob("*"))
    all_temp_file_paths = [str(p) for p in all_temp_file_paths if os.path.isfile(p)]
    
    progress(0.25, desc="📊 Analyzing files...")
    
    if not all_temp_file_paths:
        return "No valid files found in the uploaded items."

    logger.info(f"Found a total of {len(all_temp_file_paths)} files to process.")
    progress(0.35, desc=f"✅ Found {len(all_temp_file_paths)} files to process...")

    files_to_ingest = all_temp_file_paths.copy()

    if not files_to_ingest:
        return "No valid files were moved for ingestion."

    # Start ingesting data
    try:
        progress(0.4, desc="🔄 Starting file ingestion...")
        # Gọi hàm ingestion với progress callback
        ingestion_service.ingest_files_with_progress(files_to_ingest, progress)
        
        success_message = f"Successfully uploaded and ingested {len(files_to_ingest)} file(s)."
        logger.success(success_message)
        return success_message
    except Exception as e:
        error_message = f"An error occurred during the ingestion process: {e}"
        logger.error(error_message)
        return error_message

# ---- HÀM XỬ LÝ CHO TAB SEARCH ----
def search_handler(text_query: str, image_query_path: str, audio_query_path: str, top_k: int):
    def create_empty_updates(max_results=10):
        updates = []
        for _ in range(max_results):
            # Chỉ 5 components cho mỗi result: group, markdown, textbox, image, audio
            updates.extend([
                gr.Group(visible=False), 
                gr.Markdown(visible=False), 
                gr.Textbox(visible=False), 
                gr.Image(visible=False), 
                gr.Audio(visible=False)
            ])
        return updates

    # Kiểm tra database trước khi xử lý query
    try:
        if retriever_instance.is_database_empty():
            empty_db_message = gr.Textbox(
                value="Database is empty. Please go to the 'Upload Data' tab to add files first.", 
                visible=True
            )
            return [empty_db_message] + create_empty_updates()
    except Exception as e:
        error_message = gr.Textbox(
            value=f"Error checking database: {str(e)}", 
            visible=True
        )
        return [error_message] + create_empty_updates()
    
    query_type, query_content = None, None
    if text_query and text_query.strip():
        query_type, query_content = "text", text_query
    elif image_query_path:
        query_type, query_content = "image", image_query_path
    elif audio_query_path:
        query_type, query_content = "audio", audio_query_path
    
    max_results = 10 # Phải khớp với số component đã tạo

    if not query_type:
        return [gr.Textbox(value="Error: Please provide a query.", visible=True)] + create_empty_updates()

    try:
        logger.info(f"Handling '{query_type}' query: {query_content}")
        results = retriever_instance.retrieve(query_content, query_type, int(top_k))
        
        if not results:
            return [gr.Textbox(value="No results found.", visible=True)] + create_empty_updates()

        output_updates = [gr.Textbox(value="", visible=False)] # Ẩn ô info_box
        for i in range(max_results):
            if i < len(results):
                res = results[i]
                score, metadata, content = res['score'], res['metadata'], res.get('content')
                chunk_type, source_id = metadata.get('type', 'N/A'), metadata.get('source_id', 'N/A')
                info_text = f"### Result {i + 1} (Score: {score:.4f})\n**Type:** `{chunk_type}` | **Source:** `{source_id}`"
                
                text_val, text_visible = "", False
                img_val, img_visible = None, False
                audio_val, audio_visible = None, False
                
                if chunk_type == 'text':
                    text_val, text_visible = content, True
                elif chunk_type == "image":
                    if content and os.path.exists(content): 
                        img_val, img_visible = content, True
                    else: 
                        text_val, text_visible = "`Image content not found at path.`", True
                elif chunk_type == 'audio':
                    if content and os.path.exists(content): 
                        audio_val, audio_visible = content, True
                    else: 
                        text_val, text_visible = "`Audio content not found at path.`", True

                # Chỉ 5 components cho mỗi result
                output_updates.extend([
                    gr.Group(visible=True), 
                    gr.Markdown(value=info_text, visible=True), 
                    gr.Textbox(value=text_val, visible=text_visible), 
                    gr.Image(value=img_val, visible=img_visible), 
                    gr.Audio(value=audio_val, visible=audio_visible)
                ])
            else:
                # Chỉ 5 components cho mỗi result
                output_updates.extend([
                    gr.Group(visible=False), 
                    gr.Markdown(visible=False), 
                    gr.Textbox(visible=False), 
                    gr.Image(visible=False), 
                    gr.Audio(visible=False)
                ])
        
        return output_updates
        
    except Exception as e:
        error_message = f"Error during search: {str(e)}"
        logger.error(error_message)
        return [gr.Textbox(value=error_message, visible=True)] + create_empty_updates()

# --- 3. Xây dựng giao diện với Gradio Blocks ---
def create_and_run_app():
    with gr.Blocks(theme=gr.themes.Soft(), title="Multimedia RAG Assistant") as demo:
        gr.Markdown("# Multimedia RAG Assistant")

        with gr.Tabs() as tabs:
            # --- TAB 1: SEARCH ---
            with gr.TabItem("Search Database", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Query")
                        text_query_input = gr.Textbox(label="Text Query", placeholder="e.g., a dog playing in a park")
                        image_query_input = gr.Image(label="Image Query", type="filepath")
                        audio_query_input = gr.Audio(label="Audio Query", type="filepath")
                        top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Top K Results")
                        search_button = gr.Button("Search", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Retrieval Results")
                        info_box = gr.Textbox(label="Info", interactive=False, visible=False)
                        max_results = 10
                        result_components = []
                        
                        for i in range(max_results):
                            with gr.Group(visible=False) as result_group:
                                result_info = gr.Markdown()
                                result_text = gr.Textbox(label="Text Content", interactive=False, visible=False)
                                result_image = gr.Image(label="Image Content", interactive=False, visible=False)
                                result_audio = gr.Audio(label="Audio Content", visible=False, type="filepath")
                                # Chỉ thêm 5 components cho mỗi result
                                result_components.extend([result_group, result_info, result_text, result_image, result_audio])
                        
                        all_outputs = [info_box] + result_components
                        search_button.click(
                            fn=search_handler,
                            inputs=[text_query_input, image_query_input, audio_query_input, top_k_slider],
                            outputs=all_outputs
                        )

            # --- TAB 2: UPLOAD ---
            with gr.TabItem("Upload Data", id=1):
                gr.Markdown("### Upload New Data to the Database")
                gr.Markdown("You can upload multiple files of different types at once (text, images, audio), or drop a folder.")
                with gr.Column():
                    upload_file_input = gr.File(
                        label="Upload ZIP file containing your data",
                        file_types=[".zip"],
                        file_count="single",
                        type="filepath"
                    )
                    upload_button = gr.Button("Upload and Ingest", variant="primary")
                    upload_status = gr.Textbox(label="Status", interactive=False, placeholder="Upload status will be shown here...")

                    upload_button.click(
                        fn=upload_handler,
                        inputs=[upload_file_input],
                        outputs=[upload_status],
                        show_progress="full"  # Hiển thị progress bar
                    )

        # Xử lý sự kiện để xóa các input khác trong tab Search
        def clear_search_inputs(input_type):
            if input_type == 'text': return gr.Image(value=None), gr.Audio(value=None)
            elif input_type == 'image': return gr.Textbox(value=""), gr.Audio(value=None)
            elif input_type == 'audio': return gr.Textbox(value=""), gr.Image(value=None)

        text_query_input.change(lambda: clear_search_inputs('text'), outputs=[image_query_input, audio_query_input], queue=False)
        image_query_input.change(lambda: clear_search_inputs('image'), outputs=[text_query_input, audio_query_input], queue=False)
        audio_query_input.change(lambda: clear_search_inputs('audio'), outputs=[text_query_input, image_query_input], queue=False)

    return demo

# --- 4. Chạy ứng dụng ---
# logger.info("Launching Gradio interface...")
# demo = create_and_run_app()
# demo.launch()