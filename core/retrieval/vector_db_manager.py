import os

from utils.logger import logger
from config.settings import settings
from uuid import uuid4

from typing import List, Tuple, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus

class VectorDBManager:
    def __init__(self, collection_name: str, embedding_dim: int, client: QdrantClient = None):
        logger.info(f"Initializing Qdrant VectorDBManager for collection: '{collection_name}'")
        
        if client:
            self.client = client
            logger.info("Using shared Qdrant client instance.")
        else:
            logger.warning("No shared Qdrant client provided. Creating a new local instance.")
            qdrant_db_path = os.path.join(settings.DATA_DIR, "qdrant_data")
            self.client = QdrantClient(path=qdrant_db_path)
        
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        self.create_collection_if_not_exists()
        
    def create_collection_if_not_exists(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Collection '{self.collection_name}' not found. Creating a new one...")
                
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.success(f"Collection '{self.collection_name}' created successfully.")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
                
        except Exception as e:
            logger.error(f"Error checking or creating collection '{self.collection_name}': {e}")
            raise
        
    def add_vectors(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        if not embeddings:
            logger.warning("No embeddings to add. Skipping.")
            return
        
        if len(embeddings) != len(metadatas):
            logger.error("Number of embeddings and metadatas must match.")
            raise ValueError("Embeddings and metadatas count mismatch.")
        
        points_to_add = []
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
            point_id = str(uuid4())
                
            points_to_add.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                )
            )
            
        try:
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points_to_add
            )
            if operation_info.status == UpdateStatus.COMPLETED:
                logger.debug(f"Successfully upserted {len(points_to_add)} points to collection '{self.collection_name}'.")
            else:
                logger.warning(f"Upsert operation finished with status: {operation_info.status}")
        except Exception as e:
            logger.error(f"Error upserting points to collection '{self.collection_name}': {e}")
            
    def search_vectors(self, query_embedding: List[float], k: int = 5, filter_payload: Dict = None) -> List[Tuple[float, Dict[str, Any]]]:
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_payload,
                limit=k,
                with_payload=True, # include payload in return
                with_vectors=False # exclude vectors in return
            )
            
            formatted_results = []
            for scored_point in search_results:
                score = scored_point.score
                payload = scored_point.payload
                formatted_results.append((score, payload))
            
            logger.debug(f"Searched for top {k} neighbors. Found {len(formatted_results)} results.")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching in collection '{self.collection_name}': {e}")
            return []
        
    def get_total_vectors(self) -> int:
        try:
            count_result = self.client.count(
                collection_name=self.collection_name,
                exact=True
            )
            return count_result.count
        except Exception as e:
            logger.error(f"Error counting vectors in collection '{self.collection_name}': {e}")
            return 0