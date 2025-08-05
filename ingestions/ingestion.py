# core/ingestion/ingestion_service.py
import os
from typing import List, Optional, Callable

from utils.logger import logger
from qdrant_client import QdrantClient

from core.data_processing.text_processor import TextProcessor
from core.data_processing.audio_processor import AudioProcessor
from core.data_processing.image_processor import ImageProcessor

from core.embeddings.text_embedding_model import TextEmbeddingModel
from core.embeddings.image_embedding_model import ImageEmbeddingModel
from core.embeddings.audio_embedding_model import AudioEmbeddingModel

from core.retrieval.vector_db_manager import VectorDBManager

class IngestionService:
    def __init__(self, client: QdrantClient):
        logger.info("Initializing IngestionService (Stateless)...")
        
        self.client = client
        
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        
        self.text_embedder = TextEmbeddingModel()
        self.image_embedder = ImageEmbeddingModel()
        self.audio_embedder = AudioEmbeddingModel()
        
        text_embedding_dim = self.text_embedder.model.get_sentence_embedding_dimension()
        self.text_db_manager = VectorDBManager(
            client=self.client,
            collection_name="text_collection",
            embedding_dim=text_embedding_dim
        )
        
        image_embedding_dim = self.image_embedder.model.config.hidden_size 
        self.image_vector_db_manager = VectorDBManager(
            client=self.client,
            collection_name="image_collection", 
            embedding_dim=image_embedding_dim
        )
        
        audio_embedding_dim = self.audio_embedder.model.config.projection_dim
        self.audio_vector_db_manager = VectorDBManager(
            client=self.client,
            collection_name="audio_collection", 
            embedding_dim=audio_embedding_dim
        )
        
        logger.info("IngestionService initialized successfully.")

    def ingest_files(self, file_paths: List[str]):
        '''Ingest files without displaying progress bar'''
        return self.ingest_files_with_progress(file_paths, None)
    
    def ingest_files_with_progress(self, file_paths: List[str], progress_callback: Optional[Callable] = None):
        """
        Turn on progress bar for tracking
        """
        logger.info(f"Starting ingestion for {len(file_paths)} files...")
        
        # Kiểm tra và xử lý progress_callback an toàn
        def safe_progress(value, desc=""):
            try:
                if progress_callback is not None:
                    progress_callback(value, desc=desc)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
        
        safe_progress(0.4, desc="Starting file processing...")
        
        all_chunks_to_process = []
        
        # 1. Walk through files to split chunks
        for i, file_path in enumerate(file_paths):
            try:
                base_progress = 0.4 + (i / len(file_paths)) * 0.3  # 40% -> 70%
                file_name = os.path.basename(file_path)
                
                safe_progress(base_progress, desc=f"Processing file {i+1}/{len(file_paths)}: {file_name}")
                
                file_ext = os.path.splitext(file_path)[1].lower()
                chunks = []
                
                safe_progress(base_progress + 0.01, desc=f"Reading {file_name}...")
                
                if file_ext in ['.txt']:
                    chunks = self.text_processor.process(file_path)
                elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                    chunks = self.image_processor.process(file_path)
                elif file_ext in ['.wav', '.mp3']:
                    chunks = self.audio_processor.process(file_path)
                else:
                    logger.warning(f"Unsupported file type '{file_ext}' for file: {file_path}. Skipping.")
                    continue
                
                # Kiểm tra chunks có hợp lệ không
                if not chunks or len(chunks) == 0:
                    logger.warning(f"No chunks generated from file: {file_path}")
                    continue
                
                safe_progress(base_progress + 0.02, desc=f"Generated {len(chunks)} chunks from {file_name}")
                all_chunks_to_process.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        if not all_chunks_to_process:
            logger.warning("No processable chunks were generated from the provided files.")
            safe_progress(1.0, desc="No chunks to process")
            return

        logger.info(f"Generated {len(all_chunks_to_process)} total chunks. Now generating embeddings...")
        
        safe_progress(0.7, desc=f"Generated {len(all_chunks_to_process)} chunks. Starting embeddings...")

        # 2. Create embeddings and add to batch
        text_embeddings_batch, text_metadatas_batch = [], []
        audio_embeddings_batch, audio_metadatas_batch = [], []
        image_embeddings_batch, image_metadatas_batch = [], []
        BATCH_SIZE = 32

        for i, chunk_data in enumerate(all_chunks_to_process):
            try:
                base_progress = 0.7 + (i / len(all_chunks_to_process)) * 0.25  # 70% -> 95%
                
                # Kiểm tra chunk_data có hợp lệ không
                if not chunk_data or 'metadata' not in chunk_data or 'content' not in chunk_data:
                    logger.warning(f"Invalid chunk data at index {i}, skipping...")
                    continue
                
                chunk_type = chunk_data['metadata'].get('type', 'unknown')
                content = chunk_data['content']
                chunk_id = chunk_data['metadata'].get('chunk_id', f'chunk_{i}')
                
                # Kiểm tra content có hợp lệ không
                if not content:
                    logger.warning(f"Empty content for chunk {chunk_id}, skipping...")
                    continue
                
                safe_progress(base_progress, desc=f"Processing chunk {i+1}/{len(all_chunks_to_process)} ({chunk_type})")
                
                if chunk_type == "text":
                    safe_progress(base_progress + 0.001, desc=f"Creating text embedding for chunk {i+1}")
                    embeddings = self.text_embedder.get_embeddings([content])
                    if embeddings and len(embeddings) > 0:
                        text_embeddings_batch.append(embeddings[0])
                        text_metadatas_batch.append(chunk_data)
                    else:
                        logger.warning(f"Failed to generate text embedding for chunk {chunk_id}")
                        
                elif chunk_type == "audio":
                    safe_progress(base_progress + 0.001, desc=f"Creating audio embedding for chunk {i+1}")
                    embeddings = self.audio_embedder.get_embeddings([content])
                    if embeddings and len(embeddings) > 0:
                        audio_embeddings_batch.append(embeddings[0])
                        audio_metadatas_batch.append(chunk_data)
                    else:
                        logger.warning(f"Failed to generate audio embedding for chunk {chunk_id}")
                        
                elif chunk_type == "image":
                    safe_progress(base_progress + 0.001, desc=f"Creating image embedding for chunk {i+1}")
                    embeddings = self.image_embedder.get_embeddings([content])
                    if embeddings and len(embeddings) > 0:
                        image_embeddings_batch.append(embeddings[0])
                        image_metadatas_batch.append(chunk_data)
                    else:
                        logger.warning(f"Failed to generate image embedding for chunk {chunk_id}")

                # add batch when reaching BATCH_SIZE
                if len(text_embeddings_batch) >= BATCH_SIZE:
                    safe_progress(base_progress + 0.002, desc=f"Saving batch of {len(text_embeddings_batch)} text embeddings...")
                    self.text_db_manager.add_vectors(text_embeddings_batch, text_metadatas_batch)
                    text_embeddings_batch, text_metadatas_batch = [], []
                    
                if len(audio_embeddings_batch) >= BATCH_SIZE:
                    safe_progress(base_progress + 0.002, desc=f"Saving batch of {len(audio_embeddings_batch)} audio embeddings...")
                    self.audio_vector_db_manager.add_vectors(audio_embeddings_batch, audio_metadatas_batch)
                    audio_embeddings_batch, audio_metadatas_batch = [], []
                    
                if len(image_embeddings_batch) >= BATCH_SIZE:
                    safe_progress(base_progress + 0.002, desc=f"Saving batch of {len(image_embeddings_batch)} image embeddings...")
                    self.image_vector_db_manager.add_vectors(image_embeddings_batch, image_metadatas_batch)
                    image_embeddings_batch, image_metadatas_batch = [], []
                    
            except Exception as e:
                logger.error(f"Error ingesting chunk {i}: {e}")
                continue

        safe_progress(0.95, desc="Saving final batches...")

        # adding maintaining embeddings
        final_operations = []
        if text_embeddings_batch:
            final_operations.append(("text", len(text_embeddings_batch)))
        if audio_embeddings_batch:
            final_operations.append(("audio", len(audio_embeddings_batch)))
        if image_embeddings_batch:
            final_operations.append(("image", len(image_embeddings_batch)))
        
        # Tránh chia cho 0
        total_operations = len(final_operations)
        if total_operations == 0:
            safe_progress(1.0, desc="No final batches to save")
        else:
            for i, (batch_type, count) in enumerate(final_operations):
                try:
                    current_progress = 0.95 + (i / total_operations) * 0.04  # 95% -> 99%
                    
                    if batch_type == "text" and text_embeddings_batch:
                        safe_progress(current_progress, desc=f"Saving final {count} text embeddings...")
                        self.text_db_manager.add_vectors(text_embeddings_batch, text_metadatas_batch)
                    elif batch_type == "audio" and audio_embeddings_batch:
                        safe_progress(current_progress, desc=f"Saving final {count} audio embeddings...")
                        self.audio_vector_db_manager.add_vectors(audio_embeddings_batch, audio_metadatas_batch)
                    elif batch_type == "image" and image_embeddings_batch:
                        safe_progress(current_progress, desc=f"Saving final {count} image embeddings...")
                        self.image_vector_db_manager.add_vectors(image_embeddings_batch, image_metadatas_batch)
                except Exception as e:
                    logger.error(f"Error saving final batch {batch_type}: {e}")
        
        safe_progress(1.0, desc=f"✅ Successfully ingested {len(file_paths)} files with {len(all_chunks_to_process)} chunks!")
        
        logger.success(f"Successfully completed ingestion for {len(file_paths)} files.")