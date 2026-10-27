from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from repo.post_repo import PostRepo
from repo.comment_repo import CommentRepo

class BoardService:
    def __init__(self):
        self.post_repo = PostRepo()
        self.comment_repo = CommentRepo()

    async def create_post(self, user_code: int, title: str, content: str | None, analysis_code: int, db: AsyncSession):
        return await self.post_repo.create_post(user_code, title, content, analysis_code, db)

    async def get_posts(self, db: AsyncSession):
        posts = await self.post_repo.get_posts(db)
        return posts

    async def get_post_detail(self, post_id: int, db: AsyncSession):
        post = await self.post_repo.get_post_by_id(post_id, db)
        if not post:
            raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")
        comments = await self.comment_repo.get_comments_by_post_id(post_id, db)
        return {
            "post": post,
            "comments": comments
        }

    async def create_comment(self, post_id: int, user_code: int, content: str, db: AsyncSession):
        post = await self.post_repo.get_post_by_id(post_id, db)
        if not post:
            raise HTTPException(status_code=404, detail="존재하지 않는 게시글입니다.")
        return await self.comment_repo.create_comment(post_id, user_code, content, db)
