import os

from typing import List, Dict, Any
from utils.logger import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len, # count character, can be replaced 
            add_start_index=True # 
        )
        
        logger.info(f"TextProcessor initialized with LangChain's RecursiveCharacterTextSplitter (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
        
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info(f"Processing text document: {file_path}")
            split_texts = self.text_splitter.split_text(text)
            
            chunks = []
            for i, chunk_content in enumerate(split_texts):
                chunk_id = f"{os.path.basename(file_path).split('.')[0]}_chunk_text_{i}"
                metadata = {
                    "source_id": os.path.basename(file_path),
                    "type": "text",
                    "chunk_id": chunk_id,
                    "content_length": len(chunk_content)
                }
                chunks.append({
                    "content": chunk_content,
                    "metadata": metadata
                })
            logger.info(f"Generated {len(chunks)} text chunks from {file_path}")
            return chunks
        except Exception as e:
            logger.error(f"Error processing text document {file_path}: {e}")
            return []