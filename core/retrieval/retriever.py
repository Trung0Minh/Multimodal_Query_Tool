# core/retrieval/retriever.py
import os

from utils.logger import logger
from config.settings import settings
from typing import List, Dict, Any, Union
from qdrant_client import QdrantClient

from core.embeddings.text_embedding_model import TextEmbeddingModel
from core.embeddings.image_embedding_model import ImageEmbeddingModel
from core.embeddings.audio_embedding_model import AudioEmbeddingModel

from core.retrieval.vector_db_manager import VectorDBManager

class Retriever:
    def __init__(self, client: QdrantClient):
        logger.info("Initializing the Retriever...")
        
        # Initialize embedding models
        self.text_embedder = TextEmbeddingModel()
        self.image_embedder = ImageEmbeddingModel()
        self.audio_embedder = AudioEmbeddingModel()
        logger.info("Embedding models initialized.")
        
        qdrant_db_path = os.path.join(settings.DATA_DIR, "qdrant_data")
        self.client = client
        logger.info(f"Single Qdrant client initialized, connected to: {qdrant_db_path}")
        
        # Initialize vector database
        text_dim = self.text_embedder.model.get_sentence_embedding_dimension()
        self.text_db_manager = VectorDBManager(collection_name="text_collection", embedding_dim=text_dim, client=self.client)
        
        image_dim = 512
        self.image_db_manager = VectorDBManager(collection_name="image_collection", embedding_dim=image_dim, client=self.client)
        
        audio_dim = 512
        self.audio_db_manager = VectorDBManager(collection_name="audio_collection", embedding_dim=audio_dim, client=self.client)
        
        logger.info("VectorDB Managers connected to Qdrant collections.")
        logger.info(f"Text collection ('{self.text_db_manager.collection_name}') contains {self.text_db_manager.get_total_vectors()} vectors.")
        logger.info(f"Image collection ('{self.image_db_manager.collection_name}') contains {self.image_db_manager.get_total_vectors()} vectors.")
        logger.info(f"Audio collection ('{self.audio_db_manager.collection_name}') contains {self.audio_db_manager.get_total_vectors()} vectors.")
        
    def retrieve(self, query: Union[str, bytes], query_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Received retrieval request. Query type: '{query_type}', Top K: {top_k}")
        
        embedding = None
        db_manager_to_use = None
        
        # create embeddings
        try:
            if query_type == "text":
                if not isinstance(query, str):
                    raise TypeError("Text query must be a string.")
                embedding = self.text_embedder.get_embeddings(query)
                db_manager_to_use = self.text_db_manager
            elif query_type == "image":
                if not isinstance(query, str) or not os.path.exists(query):
                    raise TypeError("Image query must be a valid file path.")
                embedding = self.image_embedder.get_embeddings([query])[0]
                db_manager_to_use = self.image_db_manager
            elif query_type == "audio":
                if not isinstance(query, str) or not os.path.exists(query):
                    raise TypeError("Audio query must be a valid file path.")
                embedding = self.audio_embedder.get_embeddings([query])[0]
                db_manager_to_use = self.audio_db_manager
            else:
                logger.error(f"Unsupported query type: {query_type}")
                return []
        except Exception as e:
            logger.error(f"Error generating embedding for query: {e}")
            return []
        
        if embedding is None:
            logger.warning("Could not generate embedding for the query.")
            return []
        
        # searching vectors
        try:
            search_results = db_manager_to_use.search_vectors(embedding, k=top_k)
        except Exception as e:
            logger.error(f"Error searching in vector database: {e}")
            return []
        
        formatted_results = []
        for score, payload in search_results:
            formatted_results.append({
                "score": score,
                "metadata": payload['metadata'],
                "content": payload['content']
            })
            
        logger.info(f"Retrieval complete. Found {len(formatted_results)} results.")
        return formatted_results
    
    def is_database_empty(self) -> bool:
        total_vectors = self.text_db_manager.get_total_vectors() \
            + self.image_db_manager.get_total_vectors() \
            + self.audio_db_manager.get_total_vectors()
            
        return total_vectors == 0