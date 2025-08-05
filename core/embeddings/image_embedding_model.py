import torch
from typing import List
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from utils.logger import logger
from config.model_configs import IMAGE_EMBEDDING_MODEL

class ImageEmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Image Embedding Model '{IMAGE_EMBEDDING_MODEL}' to device: {self.device}")
        
        self.model = ViTModel.from_pretrained(IMAGE_EMBEDDING_MODEL).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info("Image Embedding Model loaded successfully.")
        
    def get_embeddings(self, image_paths: List[str]) -> List[List[float]]:
        if not image_paths:
            logger.warning("No image paths provided")
            return []
        
        images = []
        valid_paths = []
        
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                images.append(image)
                valid_paths.append(img_path)
            except Exception as e:
                logger.warning(f"Could not load image {img_path}: {e}. Skipping.")
                continue
            
        if not images:
            logger.warning("No valid images to process")
            return []
        
        try:
            # Process images
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(**inputs)
                
                # Extract embeddings from the [CLS] token (first token)
                # Shape: (batch_size, sequence_length, hidden_size)
                last_hidden_states = outputs.last_hidden_state
                
                # Take the [CLS] token embedding (index 0)
                # Shape: (batch_size, hidden_size)
                cls_embeddings = last_hidden_states[:, 0, :]
                
                # Alternatively, you can use pooler_output if available
                # cls_embeddings = outputs.pooler_output
            
            # Normalize embeddings (L2 normalization)
            embeddings = cls_embeddings / cls_embeddings.norm(p=2, dim=-1, keepdim=True)
            
            # Convert to list
            embeddings_list = embeddings.cpu().tolist()
            
            logger.debug(f"Generated {len(embeddings_list)} embeddings for {len(images)} images.")
            
            # Ensure we return the right number of embeddings
            if len(embeddings_list) != len(image_paths):
                logger.warning(f"Mismatch: {len(embeddings_list)} embeddings for {len(image_paths)} input paths")
                # Pad with empty lists if needed
                while len(embeddings_list) < len(image_paths):
                    embeddings_list.append([])
            
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return empty embeddings for all input paths
            return [[] for _ in image_paths]