from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from db.mariadb_orm import get_db
from dto.requests import ChatRequest
from entity.entity import Chat
from service.chat_service import ChatService
from dto.token import DecodedToken
from service.login_service import get_user
from sqlalchemy.ext.asyncio import AsyncSession

cr = APIRouter(prefix="/chat")
chat_service = ChatService()


@cr.get("")
async def get_chat(page: int = 0, user:DecodedToken=Depends(get_user), db:AsyncSession=Depends(get_db)):
    # 유저의 chat 가져오기
    r = await chat_service.get_chats_by_page(db, user, page)
    return r

@cr.post("")
async def chat(req: ChatRequest, user:DecodedToken=Depends(get_user), db:AsyncSession=Depends(get_db)):
    print(req)
    chats: list[Chat] = await chat_service.get_chats(db, user, 10)
    return StreamingResponse(
        chat_service.response_llm(req.content, chats, user, db),
        media_type="application/x-ndjson"
    )
# @cr.post("/summary")
# async def summary(req: SummaryRequest, user:User=Depends(get_user)):
#     return await chat_service.summary(req, user.user_code)