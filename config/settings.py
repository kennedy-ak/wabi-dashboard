import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Configuration
    API_TITLE = "Furniture Category Classification API"
    API_DESCRIPTION = "API for classifying furniture categories using text or image data"
    API_VERSION = "1.0.0"
    
    # Server Configuration
    HOST = "0.0.0.0"
    PORT = 8000
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Processing Configuration
    TEXT_BATCH_SIZE = 10
    IMAGE_BATCH_SIZE = 5
    
    # Concurrency Configuration
    MAX_IMAGE_WORKERS = 3  # Concurrent image classifications
    MAX_TEXT_WORKERS = 2   # Concurrent text classifications
    MAX_SCRAPER_WORKERS = 3  # Concurrent scraping requests per domain
    
    # Batch Processing
    TEXT_CHUNK_SIZE = 5    # Items per text API call
    
    # CORS Configuration
    CORS_ORIGINS = ["*"]
    CORS_CREDENTIALS = True
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]
    
    # Furniture Categories
    FURNITURE_CATEGORIES = [
        "SOFA", "CHAIR", "BED", "TABLE", "NIGHTSTAND", "STOOL", 
        "STORAGE", "DESK", "BENCH", "OTTOMAN", "LIGHTING", "DECOR", "OTHER"
    ]

settings = Settings()
