from openai import OpenAI
import os
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()

client.batches.retrieve("batch_67330e8849088190af08c593e35c33d0")