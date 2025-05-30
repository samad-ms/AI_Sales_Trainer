import openai
import os
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

def label_transcript_utterances(transcript):
    """
    Uses OpenAI to split the transcript into utterances and label each as 'Prospect' or 'Rep'.
    Returns a list of dicts: [{'speaker': 'Prospect'/'Rep'/'Unknown', 'text': ...}, ...]
    """
    prompt = (
        "You are given a call transcript between a sales representative and a prospect. "
        "Split the transcript into individual utterances or lines. For each, label it as either 'Rep' (sales representative), 'Prospect' (the person being sold to), or 'Unknown' if unclear. "
        "Return the result as a numbered list, each line in the format: [Speaker] Utterance text\n"
        f"Transcript:\n{transcript}\n"
        "Labeled Utterances:"
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=800
    )
    text = response.choices[0].message.content.strip()
    utterances = []
    for line in text.splitlines():
        match = re.match(r"^\d+\.\s*\[(\w+)\]\s*(.+)", line)
        if match:
            speaker, utterance = match.groups()
            utterances.append({'speaker': speaker, 'text': utterance.strip()})
    return utterances

def extract_prospect_questions_from_labels(labeled_utterances):
    """
    Uses OpenAI to extract all explicit and implicit questions from the prospect's utterances.
    Returns a list of question strings.
    """
    prospect_utterances = [item['text'] for item in labeled_utterances if item['speaker'].lower() == 'prospect']
    if not prospect_utterances:
        return []
    joined = "\n".join(prospect_utterances)
    prompt = (
        "Below are utterances from a prospect in a sales call. "
        "Identify all explicit and implicit questions the prospect is asking, even if they are not phrased as a question or do not end with a question mark. "
        "Return each question as a separate line, rephrasing statements as questions if needed. "
        "If there are no questions, return an empty list.\n\n"
        f"Utterances:\n{joined}\n\n"
        "Prospect's Questions:"
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    text = response.choices[0].message.content.strip()
    questions = [q.strip("-•1234567890. \t") for q in text.splitlines() if q.strip()]
    return questions
