from datetime import datetime

from sqlalchemy import select, update

from entity.entity import AnalysisResult
from sqlalchemy.ext.asyncio import AsyncSession

class AnalysisRepo:
    def __init__(self):
        pass

    async def insert_today_emotion(self, today_analyze:dict, emotion_name, user_code, db:AsyncSession):
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
            sql = (update(AnalysisResult).where(AnalysisResult.analysis_code == existing.analysis_code)
                   .values(
                happy=today_analyze.get('기쁨'),
                anger=today_analyze.get('분노'),
                anxiety=today_analyze.get('불안'),
                sadness=today_analyze.get('슬픔'),
                calmness=today_analyze.get('평온'),
                confusion=today_analyze.get('당황'),
                emotion_name=emotion_name,
            )
            )
            await db.execute(sql)
            await db.commit()
        else:
            data = AnalysisResult(
                happy=today_analyze.get('기쁨'),
                anger=today_analyze.get('분노'),
                anxiety=today_analyze.get('불안'),
                sadness=today_analyze.get('슬픔'),
                calmness=today_analyze.get('평온'),
                confusion=today_analyze.get('당황'),
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
