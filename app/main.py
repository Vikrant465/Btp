from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import google.generativeai as genai

from fastapi.middleware.cors import CORSMiddleware

# Define the input data model
class UserInput(BaseModel):
    question: str

app = FastAPI()
origins = [
    "http://localhost:3000",  # Frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Origins that can access your API
    allow_credentials=True,
    allow_methods=["*"],    # HTTP methods allowed
    allow_headers=["*"],    # Headers allowed
)


# Configure generative AI
API = "AIzaSyCHRT6oCXDHvaSEfuAjZeLOqNWUu43bnuU"
genai.configure(api_key=API)
model1 = genai.GenerativeModel("gemini-1.5-flash")

# Load emotion detection model
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model2 = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.post("/process")
async def process_input(user_input: UserInput):
    # Generative AI response
    ai_request_prompt = "You are a friendly and respectful Emotion prediction bot. User will come to you with questions, and it’s your job to provide clear, accurate, and try to understand there emotion. Respond with empathy. there is no need to say  I'm ready to listen and do my best to understand your emotions. Please feel free to ask me anything – I'm here to help in any way I can. Remember, I'm not a therapist, but I can offer support and try my best to predict and understand what you might be feeling. every time"
    # ai_request_prompt = "You are a bot. Answer all questions directly and concisely, using as few words as possible."
    response = model1.generate_content(f"{ai_request_prompt} {user_input.question}")
    text = response.text

    # Emotion detection
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model2(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    predicted_emotion_id = torch.argmax(probabilities, dim=1).item()
    predicted_emotion = emotion_labels[predicted_emotion_id]

    return {
        "ai_response": text,
        "predicted_emotion": predicted_emotion
    }
