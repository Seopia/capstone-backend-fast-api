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

    async def insert_or_update_emotion_with_diary(self, target_date, today_analyze: dict, emotion_name: str, summary: str, user_code: int, db: AsyncSession):
        from datetime import time, timedelta
        start_of_day = datetime.combine(target_date, time.min)
        end_of_day = start_of_day + timedelta(days=1)
        
        r = await db.execute(
            select(AnalysisResult).where(
                AnalysisResult.user_code == user_code,
                AnalysisResult.create_at >= start_of_day,
                AnalysisResult.create_at < end_of_day
            )
        )
        existing = r.scalars().first()
        
        if existing:
            sql = (update(AnalysisResult).where(AnalysisResult.analysis_code == existing.analysis_code)
                   .values(
                happy=today_analyze.get('기쁨', existing.happy),
                anger=today_analyze.get('분노', existing.anger),
                anxiety=today_analyze.get('불안', existing.anxiety),
                sadness=today_analyze.get('슬픔', existing.sadness),
                calmness=today_analyze.get('평온', existing.calmness),
                confusion=today_analyze.get('당황', existing.confusion),
                emotion_name=emotion_name or existing.emotion_name,
                summary=summary,
            ))
            await db.execute(sql)
            await db.commit()
            
            r = await db.execute(select(AnalysisResult).where(AnalysisResult.analysis_code == existing.analysis_code))
            return r.scalars().first()
        else:
            create_time = datetime.combine(target_date, datetime.now().time())
            data = AnalysisResult(
                happy=today_analyze.get('기쁨', 0.0),
                anger=today_analyze.get('분노', 0.0),
                anxiety=today_analyze.get('불안', 0.0),
                sadness=today_analyze.get('슬픔', 0.0),
                calmness=today_analyze.get('평온', 0.0),
                confusion=today_analyze.get('당황', 0.0),
                create_at=create_time,
                emotion_name=emotion_name,
                summary=summary,
                user_code=user_code,
            )
            db.add(data)
            await db.commit()
            await db.refresh(data)
            return data

