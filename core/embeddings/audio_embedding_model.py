# models/embeddings/audio_embedding_model.py
import torch
import librosa

from typing import List
from transformers import AutoProcessor, AutoModel
from utils.logger import logger
from config.model_configs import AUDIO_EMBEDDING_MODEL

class AudioEmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Audio Embedding Model '{AUDIO_EMBEDDING_MODEL}' to device: {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(AUDIO_EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(AUDIO_EMBEDDING_MODEL).to(self.device)
        logger.info("Audio Embedding Model loaded successfully.")
        
    def get_embeddings(self, audio_paths: List[str]) -> List[List[float]]:
        if not audio_paths:
            return []
        
        audio_inputs = []
        sample_rate = self.processor.feature_extractor.sampling_rate
        
        for audio_path in audio_paths:
            try:
                audio_data, sr = librosa.load(audio_path, sr=sample_rate)
                audio_inputs.append(audio_data)
            except Exception as e:
                logger.warning(f"Could not load audio {audio_path}: {e}. Skipping.")
                continue
            
        if not audio_inputs:
            return []
        
        inputs = self.processor(audios=audio_inputs, sampling_rate=sample_rate, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            audio_features = self.model.get_audio_features(**inputs)
            
        embeddings = audio_features / audio_features.norm(p=2, dim=-1, keepdim=True)
        
        embeddings_list = embeddings.cpu().tolist()
        logger.debug(f"Generated {len(embeddings_list)} embeddings for {len(audio_inputs)} audio clips.")
        return embeddings_list