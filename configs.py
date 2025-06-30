import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    DATA_DIR: str = 'data'
    FFMPEG_PATH: str = 'ffmpeg'
    MAX_VIDEO_RESOLUTION: int = 360
    MAX_VIDEO_FPS: float = 30
    VIDEO_EXTENSION: str = 'mp4'
    VIDEO_EXTRACTION_FRAME_RATE: float = 1.0
    AUDIO_SEGMENT_LENGTH: int = 300
    CHATBOT_MODEL: str = 'gemini-2.0-flash'
    MODEL_BASE_API: str = 'https://generativelanguage.googleapis.com/v1beta/'
    TEXT_EMBEDDING_MODEL: str = 'nomic-ai/nomic-embed-text-v1.5'
    IMAGE_EMBEDDING_MODEL: str = 'nomic-ai/nomic-embed-vision-v1.5'




settings = Settings()

if not os.path.exists(settings.DATA_DIR):
    os.makedirs(settings.DATA_DIR)

