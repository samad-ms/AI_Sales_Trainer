import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_transcript(transcript):
    """
    Summarizes a given transcript using OpenAI Chat API.
    Args:
        transcript (str): The full transcribed text.
    Returns:
        str: A summary of the transcript.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes transcripts."},
        {"role": "user", "content": f"Summarize the following transcript:\n\n{transcript}\n\nSummary:"}
    ]
    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()
