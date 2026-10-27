from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.mariadb_orm import get_db
from dto.requests import PostCreateRequest, CommentCreateRequest
from dto.token import DecodedToken
from service.board_service import BoardService
from service.login_service import get_user

br = APIRouter(prefix="/posts")
board_svc = BoardService()


@br.post("")
async def create_post(req: PostCreateRequest, user: DecodedToken = Depends(get_user), db: AsyncSession = Depends(get_db)):
    return await board_svc.create_post(user.user_code, req.title, req.content, req.analysis_code, db)

@br.get("")
async def get_posts(db: AsyncSession = Depends(get_db)):
    return await board_svc.get_posts(db)

@br.get("/{post_id}")
async def get_post_detail(post_id: int, db: AsyncSession = Depends(get_db)):
    return await board_svc.get_post_detail(post_id, db)

@br.post("/{post_id}/comments")
async def create_comment(post_id: int, req: CommentCreateRequest, user: DecodedToken = Depends(get_user), db: AsyncSession = Depends(get_db)):
    return await board_svc.create_comment(post_id, user.user_code, req.content, db)
