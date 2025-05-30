from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.transcribe import transcribe_audio
from services.summarizer import summarize_transcript
from services.ailabeler import label_transcript_utterances, extract_prospect_questions_from_labels
from services.contextual_qa import answer_question_with_context
from services.prospect_simulator import generate_prospect_question, get_feedback
import tempfile
import os
import base64
import json

ws_router = APIRouter()

@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                req = json.loads(data)
            except Exception:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            action = req.get("action")
            payload = req.get("payload", {})

            # Example: audio file as base64
            if action == "transcribe":
                audio_b64 = payload.get("audio_base64")
                if not audio_b64:
                    await websocket.send_text(json.dumps({"error": "Missing audio_base64"}))
                    continue
                audio_bytes = base64.b64decode(audio_b64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                transcript = transcribe_audio(tmp_path)
                os.remove(tmp_path)
                await websocket.send_text(json.dumps({"action": "transcribe", "result": transcript}))

            elif action == "summarize":
                transcript = payload.get("transcript", "")
                summary = summarize_transcript(transcript)
                await websocket.send_text(json.dumps({"action": "summarize", "result": summary}))

            elif action == "label":
                transcript = payload.get("transcript", "")
                utterances = label_transcript_utterances(transcript)
                await websocket.send_text(json.dumps({"action": "label", "result": utterances}))

            elif action == "extract_questions":
                labeled_utterances = payload.get("labeled_utterances", [])
                questions = extract_prospect_questions_from_labels(labeled_utterances)
                await websocket.send_text(json.dumps({"action": "extract_questions", "result": questions}))

            elif action == "qa":
                question = payload.get("question", "")
                context = payload.get("context", "")
                answer = answer_question_with_context(question, context)
                await websocket.send_text(json.dumps({"action": "qa", "result": answer}))

            elif action == "generate_prospect_question":
                history = payload.get("conversation_history", "")
                context = payload.get("context", "")
                question = generate_prospect_question(history, context)
                await websocket.send_text(json.dumps({"action": "generate_prospect_question", "result": question}))

            elif action == "get_feedback":
                prospect_question = payload.get("prospect_question", "")
                user_answer = payload.get("user_answer", "")
                context = payload.get("context", "")
                feedback = get_feedback(prospect_question, user_answer, context)
                await websocket.send_text(json.dumps({"action": "get_feedback", "result": feedback}))

            else:
                await websocket.send_text(json.dumps({"error": "Unknown action"}))
    except WebSocketDisconnect:
        pass
        