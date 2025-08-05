# core/data_processing/image_processor.py
import os

from typing import List, Dict, Any
from utils.logger import logger

class ImageProcessor:
    def __init__(self):
        logger.info("ImageProcessor initialized.")

    def process(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            logger.debug(f"Processing image file: {file_path}")

            if not os.path.exists(file_path):
                logger.error(f"Image file not found: {file_path}")
                return []

            # create id by filename
            base_name = os.path.basename(file_path)
            chunk_id = f"{os.path.splitext(base_name)[0]}_image_chunk"

            metadata = {
                "source_id": base_name,
                "type": "image",
                "chunk_id": chunk_id,
                "chunk_data_path": file_path
            }

            chunk = {
                "content": file_path,
                "metadata": metadata
            }
            
            return [chunk]

        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {e}")
            return []