from datetime import datetime
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from entity.entity import Comment

class CommentRepo:
    async def create_comment(self, post_id: int, user_code: int, content: str, db: AsyncSession):
        comment = Comment(
            post_id=post_id,
            user_code=user_code,
            content=content,
            create_at=datetime.now()
        )
        db.add(comment)
        await db.commit()
        await db.refresh(comment)
        return comment

    async def get_comments_by_post_id(self, post_id: int, db: AsyncSession):
        r = await db.execute(
            select(Comment)
            .where(Comment.post_id == post_id)
            .options(joinedload(Comment.user))
            .order_by(Comment.create_at.asc())
        )
        return r.scalars().all()
