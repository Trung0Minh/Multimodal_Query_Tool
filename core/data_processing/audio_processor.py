# core/data_processing/audio_processor.py
import os

from typing import List, Dict, Any
from utils.logger import logger
from pydub import AudioSegment
from pydub.silence import split_on_silence
from config.settings import settings

class AudioProcessor:
    def __init__(self, min_silence_len: int = 1000, silence_thresh_db: int = -40, target_sr: int = 16000):
        self.min_silence_len = min_silence_len
        self.silence_thresh_db = silence_thresh_db
        self.target_sr = target_sr
        logger.info(f"AudioProcessor initialized (min_silence_len={min_silence_len}ms, silence_thresh_db={silence_thresh_db}dB).")

    def process(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Processing audio file: {file_path}")
            audio = AudioSegment.from_file(file_path)

            if audio.frame_rate != self.target_sr:
                audio = audio.set_frame_rate(self.target_sr)

            audio_segments = split_on_silence(
                audio,
                min_silence_len=self.min_silence_len,
                silence_thresh=self.silence_thresh_db,
                keep_silence=500
            )

            chunks = []
            audio_chunks_dir = os.path.join(settings.CHUNKS_DIR, "audio")
            os.makedirs(audio_chunks_dir, exist_ok=True)

            for i, segment in enumerate(audio_segments):
                segment_id = f"{os.path.basename(file_path).split('.')[0]}_chunk_audio_{i}"
                chunk_file_path = os.path.join(audio_chunks_dir, f"{segment_id}.wav")
                
                # Save segments into data/processed/chunks
                segment.export(chunk_file_path, format="wav")

                metadata = {
                    "source_id": os.path.basename(file_path),
                    "type": "audio",
                    "chunk_id": segment_id,
                    "chunk_data_path": chunk_file_path,
                    "duration_ms": len(segment)
                }
                chunks.append({
                    "content": chunk_file_path,
                    "metadata": metadata
                })
            logger.info(f"Generated {len(chunks)} audio segments from {file_path}")
            return chunks
        except FileNotFoundError:
            logger.error(f"Audio file not found: {file_path}. Please ensure ffmpeg is installed and accessible.")
            return []
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {e}")
            return []