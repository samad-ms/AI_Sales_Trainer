import openai
import tempfile
import os

def text_to_speech_file(text, voice="onyx", model="tts-1"):
    response = openai.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    response.stream_to_file(tmp.name)
    tmp.close()
    return tmp.name
