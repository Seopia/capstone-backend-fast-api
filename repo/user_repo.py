from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from dto.token import DecodedToken
from entity.entity import User

class UserRepo:
    def __init__(self):
        pass

    async def is_exist_user(self, id, db: AsyncSession) -> bool:
        r = await db.execute(select(User.oauth_id).where(User.oauth_id == id))
        return r.scalar_one_or_none() is not None

    async def create_new_user(self, user: User, db: AsyncSession):
        try:
            db.add(user)
            await db.commit()
        except IntegrityError:
            await db.rollback()
            raise HTTPException(status_code=409, detail="이미 존재하는 사용자입니다.")
        except SQLAlchemyError:
            await db.rollback()
            raise HTTPException(status_code=500, detail="DB 처리 중 오류가 발생했습니다.")


    async def insert_refresh_token(self, k_id:str, refresh_token:str, db: AsyncSession):
        await db.execute(update(User).where(User.oauth_id == k_id).values(refresh_token=refresh_token))
        await db.commit()

    async def find_by_user_code(self, oauth_id: str, db: AsyncSession):
        r =  await db.execute(select(User.refresh_token).where(User.oauth_id == oauth_id))
        return r.scalar()

    async def find_by_user_oauth_id(self, oauth_id: str, db: AsyncSession):
        r = await db.execute(select(User).where(User.oauth_id == oauth_id))
        return r.scalar()

    async def get_me(self, user: DecodedToken, db:AsyncSession):
        r = await db.execute(select(User.user_code, User.name, User.profile_img).where(User.oauth_id == user.oauth_id))
        r = r.mappings().one_or_none()
        print(r)
        return r