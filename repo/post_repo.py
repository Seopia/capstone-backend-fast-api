from datetime import datetime
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from entity.entity import Post

class PostRepo:
    async def create_post(self, user_code: int, title: str, content: str | None, analysis_code: int, db: AsyncSession):
        post = Post(
            user_code=user_code,
            analysis_code=analysis_code,
            title=title,
            content=content,
            create_at=datetime.now()
        )
        db.add(post)
        await db.commit()
        await db.refresh(post)
        return post

    async def get_posts(self, db: AsyncSession):
        r = await db.execute(
            select(Post)
            .options(joinedload(Post.user), joinedload(Post.analysis_result))
            .order_by(Post.create_at.desc())
        )
        return r.scalars().all()

    async def get_post_by_id(self, post_id: int, db: AsyncSession):
        r = await db.execute(
            select(Post)
            .where(Post.post_id == post_id)
            .options(joinedload(Post.user), joinedload(Post.analysis_result))
        )
        return r.scalars().first()
