from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.mariadb_orm import get_db
from dto.requests import DiaryCreateRequest
from dto.token import DecodedToken
from service.emotion_service import EmotionService
from service.login_service import get_user
from entity.entity import Chat, AnalysisResult

er = APIRouter(prefix="/analyze")
emotion_svc = EmotionService()


@er.get("")
async def analyze(year:int, month:int, user:DecodedToken=Depends(get_user), db:AsyncSession=Depends(get_db)):
    chat_history:list[Chat] = await emotion_svc.get_today_chats(user, db)
    if len(chat_history) != 0:
        today_analyze:dict = await emotion_svc.today_analyze_chat(chat_history)
        await emotion_svc.insert_today_emotion(today_analyze, user, db)
    calendar_data:list[AnalysisResult] = await emotion_svc.get_calendar_data(year, month, user, db)
    return calendar_data

@er.post("/diary")
async def create_diary(req: DiaryCreateRequest, user: DecodedToken = Depends(get_user), db: AsyncSession = Depends(get_db)):
    return await emotion_svc.generate_diary(user.user_code, req.date, db)