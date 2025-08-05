# config/model_configs.py

# Embedding Models
TEXT_EMBEDDING_MODEL: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
IMAGE_EMBEDDING_MODEL: str = "google/vit-base-patch16-224-in21k"
AUDIO_EMBEDDING_MODEL: str = "laion/clap-htsat-unfused"

# Generator Model (LLM/LMM)
GENERATOR_MODEL_NAME: str = "gpt-4o"
GENERATOR_MODEL_MAX_TOKENS: int = 4096
GENERATOR_MODEL_TEMPERATURE: float = 0.7

# Reranker Model
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"