from datetime import datetime

from sqlalchemy import select
from entity.entity import AnalysisResult
from sqlalchemy.ext.asyncio import AsyncSession

class AnalysisRepo:
    def __init__(self):
        pass

    async def insert_today_emotion(self, emotion_score, emotion_name, user_code, db:AsyncSession):
        now = datetime.now()
        start_of_day = datetime(now.year, now.month, now.day)
        
        r = await db.execute(
            select(AnalysisResult).where(
                AnalysisResult.user_code == user_code,
                AnalysisResult.create_at >= start_of_day
            )
        )
        existing = r.scalars().first()
        
        if existing:
            existing.emotion_score = emotion_score
            existing.emotion_name = emotion_name
            await db.commit()
        else:
            data = AnalysisResult(
                emotion_score=emotion_score,
                create_at=datetime.now(),
                emotion_name=emotion_name,
                user_code=user_code,
            )
            db.add(data)
            await db.commit()

    async def get_calendar_data(self,year: int,month: int, user_code: int,db: AsyncSession):
        print(year)
        print(month)
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
        r = await db.execute(
            select(AnalysisResult).where(
                AnalysisResult.user_code == user_code,
                AnalysisResult.create_at >= start,
                AnalysisResult.create_at < end
            )
        )
        return r.scalars().all()
