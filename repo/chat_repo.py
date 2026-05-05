from sqlalchemy import select

from dto.token import DecodedToken
from entity.entity import User
from entity.entity import Chat
from sqlalchemy.ext.asyncio import AsyncSession

class ChatRepo:
    def __init__(self):
        pass

    async def get_chats(self, db: AsyncSession, user:DecodedToken, count: int):
        r = await db.execute(select(Chat).where(Chat.user_code == user.user_code).limit(count).order_by(Chat.create_at))
        r = r.scalars().all()
        return r