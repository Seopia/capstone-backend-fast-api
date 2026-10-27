import os
import datetime
from typing import TypedDict, cast
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPEN_AI_API_KEY"))

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

    async def generate_diary(self, user_code: int, date_str: str, db: AsyncSession):
        try:
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식이어야 합니다.")

        chats = await self.chat_repo.get_chats_by_date(user_code, target_date, db)
        if not chats:
            raise HTTPException(status_code=400, detail="해당 날짜에 대화 기록이 없어 일기를 생성할 수 없습니다.")

        chat_text = "\n".join([f"{'사용자' if c.role == 'human' else 'AI'}: {c.content}" for c in chats])

        prompt = [
            SystemMessage(content=(
                "당신은 사용자의 하루 대화 기록을 바탕으로 일기를 작성해주는 감성적인 AI 비서입니다.\n"
                "다음 규칙을 반드시 지켜서 작성해주세요:\n"
                "1. 사용자의 입장에서 1인칭 시점('나')으로 작성하세요.\n"
                "2. 사용자가 느낀 감정과 대화 주제를 중심으로 하루를 되돌아보는 감성적이고 솔직한 일기를 쓰세요.\n"
                "3. 분량은 공백 포함 300~500자 내외로 작성하세요.\n"
                "4. 존댓말이 아닌 일기체(~했다, ~한다 등)로 자연스럽게 작성하세요.\n"
                "5. 마크다운 기호 없이 순수 텍스트로만 작성하세요."
            )),
            HumanMessage(content=f"대화 내용:\n{chat_text}\n\n위 대화 내용을 바탕으로 감성적인 일기를 작성해줘.")
        ]
        
        try:
            response = await self.llm.ainvoke(prompt)
            diary_content = response.content
        except Exception as e:
            print("LLM 일기 생성 오류:", e)
            raise HTTPException(status_code=500, detail="AI 일기 생성 중 오류가 발생했습니다.")

        human_chats = [c for c in chats if c.role == "human"]
        
        today_analyze = {}
        emotion_name = ""
        if human_chats:
            today_analyze = await self.today_analyze_chat(human_chats)
            emotion_name, _ = max(today_analyze.items(), key=lambda x: x[1])
        else:
            today_analyze = {"분노": 0.0,"불안": 0.0,"슬픔": 0.0,"평온": 0.0,"당황": 0.0,"기쁨": 0.0}
            emotion_name = "평온"

        result = await self.analysis_repo.insert_or_update_emotion_with_diary(
            target_date=target_date,
            today_analyze=today_analyze,
            emotion_name=emotion_name,
            summary=diary_content,
            user_code=user_code,
            db=db
        )
        return result


