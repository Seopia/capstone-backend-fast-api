import os

import fastapi
import uvicorn
from dotenv import load_dotenv
from starlette.responses import RedirectResponse, StreamingResponse

from dto.token import DecodedToken
from entity.entity import User, AnalysisResult
from db.mariadb_orm import get_db, init_db
from dto.kakao_response import KaKaoTokenResponse, KaKaoUserResponse
from service.emotion_service import EmotionService
from service.login_service import LoginService
from service.login_service import get_user
load_dotenv(".env")

from fastapi import FastAPI, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from pytz import timezone
from dto.requests import ChatRequest, SummaryRequest
from service.chat_service import ChatService
import torch    # pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
from entity.entity import Chat
from sqlalchemy.ext.asyncio import AsyncSession

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seoul_tz = timezone("Asia/Seoul")

JWT_SECRET = os.getenv("JWT_SECRET")
CORS = os.getenv("CORS")

chat_service = ChatService()
svc = LoginService()
emotion_svc = EmotionService()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await init_db()
#     yield
# app = FastAPI(lifespan=lifespan)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=[CORS],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

@app.get("/auth/me")
async def me(user:DecodedToken=Depends(get_user), db:AsyncSession = Depends(get_db)):
    return await svc.get_me(user,db)
@app.post("/chat")
async def chat(req: ChatRequest, user:DecodedToken=Depends(get_user), db:AsyncSession=Depends(get_db)):
    print(req)
    chats: list[Chat] = await chat_service.get_chats(db, user, 10)
    return StreamingResponse(
        chat_service.response_llm(req.content, chats, user, db),
        media_type="application/x-ndjson"
    )
@app.get("/chat")
async def get_chat(page: int = 0, user:DecodedToken=Depends(get_user), db:AsyncSession=Depends(get_db)):
    # 유저의 chat 가져오기
    r = await chat_service.get_chats_by_page(db, user, page)
    return r

@app.post("/summary")
async def summary(req: SummaryRequest, user:User=Depends(get_user)):
    return await chat_service.summary(req, user.user_code)

@app.get("/analyze")
async def analyze(year:int, month:int, user:DecodedToken=Depends(get_user), db:AsyncSession=Depends(get_db)):
    chat_history:list[Chat] = await emotion_svc.get_today_chats(user, db)
    if len(chat_history) != 0:
        today_analyze:dict = await emotion_svc.today_analyze_chat(chat_history)
        await emotion_svc.insert_today_emotion(today_analyze, user, db)
    calendar_data:list[AnalysisResult] = await emotion_svc.get_calendar_data(year, month, user, db)
    return calendar_data

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
