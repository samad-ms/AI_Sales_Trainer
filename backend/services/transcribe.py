import openai
import os

def transcribe_audio(audio_file_path, model="whisper-1", language="en"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    with open(audio_file_path, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=language
        )
    return response.text
