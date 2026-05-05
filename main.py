import os

import fastapi
import uvicorn
from dotenv import load_dotenv
from sqlalchemy import Sequence
from starlette.responses import RedirectResponse, StreamingResponse
from watchfiles import awatch

from dto.token import DecodedToken
from entity.entity import User
from db.mariadb_orm import get_db, init_db
from dto.kakao_response import KaKaoTokenResponse, KaKaoUserResponse
from service.login_service import LoginService
from service.login_service import get_user
load_dotenv(".env")

from fastapi import FastAPI, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from pytz import timezone
from db.mariadb import MariaAnalysisRepo
from db.mongodb import Mongodb
from db.supabase_db import SupabaseClient
from dto.requests import ChatRequest, AnalyzeRequest, SummaryRequest
from model.llm import LlmModel
from model.transformer import TransformerModel
from service.chat_service import ChatService
import torch    # pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
from langchain_openai import OpenAIEmbeddings
from entity.entity import Chat
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

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
#
chat_service = ChatService(chat_model, supabase_db, embedding_model, transformer)
svc = LoginService()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await init_db()
#     yield
# app = FastAPI(lifespan=lifespan)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=[CORS],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

@app.get("/test-chat")
async def get_test_chat(user:DecodedToken=Depends(get_user), db:AsyncSession=Depends(get_db)):
    # 최근 대화 n개 가져오기
    chats: list[Chat] = await chat_service.get_chats(db, user, 10)
    print(chats)
    content = "지적장애에 대해 알려줘"
    response = await chat_service.response_llm(content, chats, user)
    return response

@app.post("/chat")
async def chat(req: ChatRequest, user:DecodedToken=Depends(get_user), db:AsyncSession=Depends(get_db)):
    print(req)
    chats: list[Chat] = await chat_service.get_chats(db, user, 10)
    return StreamingResponse(
        chat_service.response_llm(req.content, chats, user),
        media_type="application/x-ndjson"
    )

@app.post("/summary")
async def summary(req: SummaryRequest, user:User=Depends(get_user)):
    return await chat_service.summary(req, user.user_code)

@app.post("/analyze")
async def analyze(req: AnalyzeRequest, user:User=Depends(get_user)):
    return await chat_service.analyze(req, user.user_code)

@app.get("/login")
async def login(code:str|None=None, error:str|None=None, error_description:str|None=None, state: str|None=None, db:AsyncSession=Depends(get_db)):
    if error is not None:
        print(error)
        print(error_description)
    else:
        kakao_token: KaKaoTokenResponse = svc.get_kakao_token(code)
        kakao_user: KaKaoUserResponse = svc.get_kakao_user(kakao_token.access_token)
        k_id = str(kakao_user.id)
        props = kakao_user.properties
        user = await svc.is_exist_user(kakao_user, db)
        token: str = svc.create_jwt(k_id, props, user.user_code)
        refresh_token: str = svc.create_jwt(k_id, props, user.user_code, is_refresh=True)
        await svc.insert_refresh_token(k_id, refresh_token, db)
        response = RedirectResponse("http://localhost:3000", status_code=307)
        response.set_cookie(key="refill_t", value=token, httponly=True)
        response.set_cookie(key="refill_rt", value=refresh_token, httponly=True)
        return response

@app.get("/refresh", response_model=None)
async def refresh(response: fastapi.Response,refill_rt: str | None = Cookie(default=None),db = Depends(get_db)):
    new_access_token = await svc.check_refresh_token(refill_rt, db)
    response.set_cookie(key="refill_t", value=new_access_token, httponly=True)
    return new_access_token


@app.get("/test")
def test(user=Depends(get_user)):
    print(user)
    return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
