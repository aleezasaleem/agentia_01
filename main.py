import os
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the LangChain generative AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", max_retries=3, api_key=os.getenv("GEMINI_API_KEY")
)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic model for request validation
class MessageRequest(BaseModel):
    message: str


# Function to generate greeting or contextual responses
def get_greeting_or_contextual_response(query: str) -> str:
    prompt = f"""
    You are a highly capable and friendly conversational assistant. Your task is to engage users with warm, concise, and helpful responses.

    If the user's input is a greeting (e.g., "Hello", "Hi", "Good morning"), respond with an enthusiastic and friendly greeting that makes them feel welcomed.

    If the input is not a greeting, politely explain that your current functionality is limited to handling greetings, and offer a cheerful remark to keep the interaction pleasant.

    Ensure all responses are professional, approachable, and aligned with a high-quality conversational assistant.

    User's Message: {query}
    """
    response = llm.invoke(prompt)  # Invoke the LLM with the prompt
    return response.content.strip()


# FastAPI endpoint to process user messages
@app.post("/greeting")
async def respond_to_message(request: MessageRequest):
    # Process user message and get response
    response = get_greeting_or_contextual_response(request.message)
    return {"response": response}


# Run the FastAPI server locally
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
