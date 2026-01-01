import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from langchain_openai import OpenAIEmbeddings
from pytz import timezone
from db.mariadb import MariaAnalysisRepo
from db.mongodb import Mongodb
from db.supabase_db import SupabaseClient
from dto.requests import ChatRequest, User, get_user, AnalyzeRequest, SummaryRequest
from model.llm import LlmModel
from model.transformer import TransformerModel
from common.jwtmiddleware import JWTMiddleware
from service.chat_service import ChatService
import torch    # pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
from contextlib import asynccontextmanager

load_dotenv(".env")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seoul_tz = timezone("Asia/Seoul")

JWT_SECRET = os.getenv("JWT_SECRET")
CORS = os.getenv("CORS")

mariadb = MariaAnalysisRepo()
mongodb = Mongodb()
supabase_db = SupabaseClient()
chat_model = LlmModel(mongodb, mariadb)
transformer = TransformerModel(mongodb, mariadb, device)
embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPEN_AI_KEY"))

chat_service = ChatService(chat_model, supabase_db, embedding_model, transformer)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await mariadb.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(JWTMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=[CORS],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

@app.post("/chat")
async def chat(req: ChatRequest, user:User=Depends(get_user)):
    return await chat_service.chat(req, user.user_code)

@app.post("/summary")
async def summary(req: SummaryRequest, user:User=Depends(get_user)):
    return await chat_service.summary(req, user.user_code)

@app.post("/analyze")
async def analyze(req: AnalyzeRequest, user:User=Depends(get_user)):
    return await chat_service.analyze(req, user.user_code)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
