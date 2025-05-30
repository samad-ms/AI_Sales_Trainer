import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_prospect_question(conversation_history, context=None):
    prompt = (
        "You are acting as a potential customer evaluating Easify, an all-in-one mass marketing and communication platform. "
        "Easify offers solutions like bulk SMS/MMS marketing, AI content management, geo-tagging, secure calls, ringless voicemail, email, bookings, and automation. "
        "It provides several pricing plans: Basic ($49/mo, 4,500 credits, 1 phone number, 5 sub-accounts), "
        "Premium ($99/mo, 11,000 credits, 2 phone numbers, 10 sub-accounts), Enterprise ($199/mo, 25,000 credits, 2 phone numbers, 15 sub-accounts), "
        "and Custom plans for unique needs. All plans start with a 7-day free trial, no credit card required, and no hidden charges. "
        "Easify supports 10DLC compliance, SHAKEN/STIR, CNAM, and offers integrations with Google Calendar and Chrome extensions. "
        "It is designed for industries such as e-commerce, restaurants, real estate, healthcare, education, insurance, professional services, and mortgage brokers. "
        "Based on this information, ask a realistic, specific question as a prospect about Easify’s product, pricing, features, compliance, integrations, or support. "
        "Do not answer your own question. If there is prior conversation, ask a follow-up or a new relevant question. "
        "Conversation so far:\n"
        f"{conversation_history}\n"
        "Prospect:"
    )
    if context:
        prompt += "\nProduct Info:\n" + context
    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def get_feedback(prospect_question, user_answer, context=None):
    prompt = (
        f"As a sales coach, analyze the following answer to the customer's question.\n"
        f"Customer's question: {prospect_question}\n"
        f"Sales rep's answer: {user_answer}\n"
        "Give constructive feedback and suggestions for improvement."
    )
    if context:
        prompt += "\nProduct Info:\n" + context
    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()
