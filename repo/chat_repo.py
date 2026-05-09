from datetime import datetime, time, timedelta
from sqlalchemy import select

from dto.token import DecodedToken
from entity.entity import Chat
from sqlalchemy.ext.asyncio import AsyncSession

class ChatRepo:
    def __init__(self):
        pass

    async def get_chats(self, db: AsyncSession, user:DecodedToken, count: int):
        r = await db.execute(
            select(Chat)
            .where(Chat.user_code == user.user_code)
            .order_by(Chat.create_at.desc(), Chat.chat_id.desc())
            .limit(count)
        )
        chats = r.scalars().all()
        # Sort explicitly to handle corrupted data where ai and human have same create_at but wrong chat_id
        chats = sorted(chats, key=lambda c: (c.create_at, 1 if c.role == "ai" else 0))
        return chats

    async def get_chats_by_page(self, db: AsyncSession, user: DecodedToken, page: int, size: int = 20):
        offset = page * size
        r = await db.execute(
            select(Chat)
            .where(Chat.user_code == user.user_code)
            .order_by(Chat.create_at.desc(), Chat.chat_id.desc())
            .offset(offset)
            .limit(size)
        )
        chats = r.scalars().all()
        chats = sorted(chats, key=lambda c: (c.create_at, 1 if c.role == "ai" else 0), reverse=True)
        return chats

    async def insert_chat(self, content, final_answer, user, db: AsyncSession):
        human_msg = Chat(user_code=user.user_code, content=content, role="human", create_at=datetime.now())
        db.add(human_msg)
        await db.flush()
        
        ai_msg = Chat(user_code=user.user_code, content=final_answer, role="ai", create_at=datetime.now())
        db.add(ai_msg)
        await db.commit()

    async def get_today_chat(self, user_code: int, db:AsyncSession):
        today = datetime.now().date()
        start = datetime.combine(today, time.min)
        end = start + timedelta(days=1)
        r = await db.execute(
            select(Chat).where(
                Chat.user_code == user_code,
                Chat.role == "human",
                Chat.create_at >= start,
                Chat.create_at < end
            )
        )
        r = r.scalars().all()
        return r
