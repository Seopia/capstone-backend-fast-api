import os
from datetime import datetime, timedelta
import jwt
import requests
from fastapi import Cookie, HTTPException
from requests import Response
from sqlalchemy.ext.asyncio import AsyncSession

from dto.kakao_response import KaKaoTokenResponse, KaKaoUserResponse
from dto.token import DecodedToken
from repo.user_repo import UserRepo
from entity.entity import User


def get_user(refill_t: str | None = Cookie(default=None)) -> DecodedToken:
    if refill_t is None:
        raise HTTPException(status_code=401, detail='토큰이 없습니다.')
    try:
        payload = jwt.decode(refill_t, os.environ['JWT_SECRET'], os.environ['JWT_ALGORITHM'])
        return DecodedToken.model_validate(payload)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="토큰 해독 에러")

class LoginService:
    def __init__(self):
        self.user_repo = UserRepo()
        self.JWT_SECRET = os.environ['JWT_SECRET']
        self.JWT_ALGORITHM = os.environ['JWT_ALGORITHM']
        self.JWT_EXPIRE_MINUTE = os.environ['JWT_EXPIRE_MINUTE']
        self.JWT_REFRESH_EXPIRE_MINUTE= os.environ['JWT_REFRESH_EXPIRE_MINUTE']
        self.KAKAO_CLIENT_ID = os.environ['KAKAO_CLIENT_ID']
        self.KAKAO_REDIRECT_URI=os.environ['KAKAO_REDIRECT_URI']
        pass
    def get_kakao_token(self, code: str) -> KaKaoTokenResponse:
        content_type:str = "application/x-www-form-urlencoded;charset=utf-8"
        data:dict = {
            "grant_type": "authorization_code",
            "client_id": self.KAKAO_CLIENT_ID,
            "redirect_uri": self.KAKAO_REDIRECT_URI,
            "code": code,
        }
        response:Response = requests.post("https://kauth.kakao.com/oauth/token",headers={"Content-Type": content_type}, data=data)
        if response.status_code == 200:
            return KaKaoTokenResponse.model_validate(response.json())
        else:
            raise HTTPException(status_code=500, detail=response.json())


    def get_kakao_user(self, token:str) -> KaKaoUserResponse:
        try:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
                "Authorization": f"Bearer {token}",
            }
            response:Response = requests.get("https://kapi.kakao.com/v2/user/me?secure_resource=1", headers=headers)
            if response.status_code == 200:
                return KaKaoUserResponse.model_validate(response.json())
            else:
                raise HTTPException(status_code=500, detail=response.json())
        except Exception as e:
            print(e)
    def create_jwt(self, oauth_id:str, properties:dict, user_code, is_refresh:bool=False) -> str:
        if is_refresh:
            expire = datetime.now() + timedelta(minutes=float(self.JWT_REFRESH_EXPIRE_MINUTE))
        else:
            expire = datetime.now() + timedelta(minutes=float(self.JWT_EXPIRE_MINUTE))
        properties.update({"oauth_id": oauth_id})
        properties.update({"exp":expire})
        properties.update({"user_code": user_code})
        return jwt.encode(properties, self.JWT_SECRET, algorithm=self.JWT_ALGORITHM)

    async def is_exist_user(self, kakao_user:KaKaoUserResponse, db:AsyncSession):
        user_exist = await self.user_repo.is_exist_user(kakao_user.id, db)
        print(user_exist)
        if not user_exist:
            try:
                await self.create_new_user(kakao_user, db)
            except Exception as e:
                print(e)
        return await self.user_repo.find_by_user_oauth_id(str(kakao_user.id), db)

    async def create_new_user(self, user: KaKaoUserResponse, db:AsyncSession):
        user: User = User(
            oauth_id=str(user.id),
            nickname=user.properties.get('nickname'),
            create_at=datetime.now(),
            enable=True,
            last_login_time=datetime.now(),
            profile_img=user.properties.get('profile_image'),
            role="ROLE_USER",
            name=user.properties.get('nickname'),
            oauth_provider="kakao"
        )
        await self.user_repo.create_new_user(user, db)

    async def insert_refresh_token(self, k_id, refresh_token, db:AsyncSession):
        await self.user_repo.insert_refresh_token(k_id, refresh_token, db)

    async def check_refresh_token(self, refresh_token, db):
        if refresh_token is None:
            raise HTTPException(status_code=401, detail="리프레시 토큰이 없습니다.")
        try:
            payload = jwt.decode(refresh_token, self.JWT_SECRET, self.JWT_ALGORITHM)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="리프레시 토큰이 만료되었습니다.")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="유효하지 않은 리프레시 토큰입니다.")

        oauth_id = payload.get("oauth_id")
        saved_token = await self.user_repo.find_by_user_code(oauth_id, db)

        if saved_token is None or saved_token == "":
            raise HTTPException(status_code=401, detail="저장된 리프레시 토큰이 없습니다.")
        if saved_token != refresh_token:
            raise HTTPException(status_code=401, detail="리프레시 토큰이 일치하지 않습니다.")
        payload.update({"exp":datetime.now() + timedelta(minutes=float(self.JWT_EXPIRE_MINUTE))})
        new_access_token = jwt.encode(payload, self.JWT_SECRET, algorithm=self.JWT_ALGORITHM)
        return new_access_token

    async def get_me(self, user, db):
        return await self.user_repo.get_me(user, db)