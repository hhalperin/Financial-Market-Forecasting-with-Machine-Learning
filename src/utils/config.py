import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class Config:
    """
    Stores configuration variables and settings.
    """
    ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
    
    OPENAI_API_KEY = os.getenv("open_ai")
