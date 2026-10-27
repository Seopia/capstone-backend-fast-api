import os

from fastapi import APIRouter, Cookie, HTTPException, Response
from fastapi.params import Depends
from starlette.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from db.mariadb_orm import get_db
from dto.kakao_response import KaKaoTokenResponse, KaKaoUserResponse
from dto.token import DecodedToken
from service.login_service import LoginService, get_user
ur = APIRouter()
svc = LoginService()

@ur.get("/login")
async def login(code:str|None=None, error:str|None=None, error_description:str|None=None, state: str|None=None, db:AsyncSession=Depends(get_db)):
    if error is not None:
        raise HTTPException(status_code=400, detail=error_description or error)
    else:
        kakao_token: KaKaoTokenResponse = svc.get_kakao_token(code)
        kakao_user: KaKaoUserResponse = svc.get_kakao_user(kakao_token.access_token)
        k_id = str(kakao_user.id)
        props = kakao_user.properties
        user = await svc.is_exist_user(kakao_user, db)
        token: str = svc.create_jwt(k_id, props, user.user_code)
        refresh_token: str = svc.create_jwt(k_id, props, user.user_code, is_refresh=True)
        await svc.insert_refresh_token(k_id, refresh_token, db)
        response = RedirectResponse(os.getenv("CORS"), status_code=307)
        response.set_cookie(key="refill_t", value=token, httponly=True)
        response.set_cookie(key="refill_rt", value=refresh_token, httponly=True)
        return response

@ur.get("/refresh", response_model=None)
async def refresh(response: Response,refill_rt: str | None = Cookie(default=None),db = Depends(get_db)):
    new_access_token = await svc.check_refresh_token(refill_rt, db)
    response.set_cookie(key="refill_t", value=new_access_token, httponly=True)
    return new_access_token

@ur.get("/auth/user")
async def get_user_me(user:DecodedToken=Depends(get_user), db:AsyncSession=Depends(get_db)):
    user = await svc.get_user(user, db)
    return user

@ur.post("/auth/user")
async def modify_user(edited_profile:dict, user:DecodedToken=Depends(get_user), db:AsyncSession=Depends(get_db)):
    if await svc.modify_user(user, db, edited_profile):
        return True
    else:
        return False

@ur.get("/auth/me")
async def me(user:DecodedToken=Depends(get_user), db:AsyncSession = Depends(get_db)):
    return await svc.get_me(user,db)
