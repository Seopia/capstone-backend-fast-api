from typing import TypedDict, cast

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from transformers import pipeline

from dto.token import DecodedToken
from repo.chat_repo import ChatRepo
from repo.analysis_repo import AnalysisRepo
from entity.entity import Chat, AnalysisResult


class EmotionItem(TypedDict):
    label:str
    score:float

class EmotionService:
    def __init__(self):
        self.classifier = pipeline("text-classification", model="Seonghaa/korean-emotion-classifier-roberta",top_k=None)
        self.chat_repo = ChatRepo()
        self.analysis_repo = AnalysisRepo()

    async def today_analyze_chat(self, chat_history:list[Chat]) -> dict:
        result:dict = {"분노": 0.0,"불안": 0.0,"슬픔": 0.0,"평온": 0.0,"당황": 0.0,"기쁨": 0.0}
        for chat in chat_history:
            classifications = self.classifier(chat.content)
            classifications:list[dict] = classifications[0]

            for cls in classifications:
                label = cls.get("label")
                score = cls.get("score", 0)

                if label in result:
                    result[label] += score
        for r in result:
            result[r] = round((result[r] / len(chat_history)) * 100)
        return result

    async def insert_today_emotion(self, today_analyze: dict, user:DecodedToken, db:AsyncSession):
        emotion_name, emotion_score = max(today_analyze.items(), key=lambda x: x[1])
        user_code:int = user.user_code
        await self.analysis_repo.insert_today_emotion(today_analyze, emotion_name, user_code, db)

    async def get_today_chats(self, user, db) -> list[Chat]:
        chats:list[Chat] = await self.chat_repo.get_today_chat(user.user_code, db)
        return chats

    async def get_calendar_data(self, year:int, month:int, user:DecodedToken, db:AsyncSession):
        data:list[AnalysisResult] = await self.analysis_repo.get_calendar_data(year, month, user.user_code, db)
        return data


