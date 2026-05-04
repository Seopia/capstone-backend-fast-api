import os
import uvicorn
from dotenv import load_dotenv
from requests import Response
from starlette.responses import RedirectResponse

from dto.kakao_response import KaKaoTokenResponse, KaKaoUserResponse
from service.login_service import LoginService

load_dotenv(".env")

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
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seoul_tz = timezone("Asia/Seoul")

JWT_SECRET = os.getenv("JWT_SECRET")
CORS = os.getenv("CORS")

mariadb = MariaAnalysisRepo()
mongodb = Mongodb()
supabase_db = SupabaseClient()
chat_model = LlmModel(mongodb, mariadb)
transformer = TransformerModel(mongodb, mariadb, device)
embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPEN_AI_API_KEY"))

chat_service = ChatService(chat_model, supabase_db, embedding_model, transformer)
login_service = LoginService()

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

@app.get("/login")
async def login(code:str|None=None, error:str|None=None, error_description:str|None=None, state: str|None=None):
    if error is not None:
        print(error)
        print(error_description)
    else:
        kakao_token: KaKaoTokenResponse = login_service.get_token(code) # id토큰은 아직..
        # kakao_user.properties 딕셔너리는 동의항목에 따라 다르게 나옴. 또한 kakao_account 딕셔너리도 마찬가지임
        kakao_user: KaKaoUserResponse = login_service.get_user(kakao_token.access_token)
        token: str = login_service.create_jwt(kakao_user.properties, 60)
        # 리프레시 저장하는거 만들어야함
        refresh_token: str = login_service.create_jwt(kakao_user.properties, 6000)
        response = RedirectResponse("http://localhost:3000", status_code=307)
        response.set_cookie(key="token", value=token, httponly=True)
        return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
