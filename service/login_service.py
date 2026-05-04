from datetime import datetime, timedelta

import jwt
import requests
from requests import Response

from dto.kakao_response import KaKaoTokenResponse, KaKaoUserResponse


class LoginService:
    def __init__(self):
        pass
    def get_token(self, code: str) -> KaKaoTokenResponse:
        content_type:str = "application/x-www-form-urlencoded;charset=utf-8"
        data:dict = {
            "grant_type": "authorization_code",
            "client_id": "aa744aa96b95e26a882f6c6d522c3d97",
            "redirect_uri": "http://localhost:8000/login",
            "code": code,
        }
        response:Response = requests.post("https://kauth.kakao.com/oauth/token",headers={"Content-Type": content_type}, data=data)
        return KaKaoTokenResponse.model_validate(response.json())

    def get_user(self, token:str) -> KaKaoUserResponse:
        try:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
                "Authorization": f"Bearer {token}",
            }
            response:Response = requests.get("https://kapi.kakao.com/v2/user/me?secure_resource=1", headers=headers)
            return KaKaoUserResponse.model_validate(response.json())
        except Exception as e:
            print(e)
    def create_jwt(self, properties:dict, expire: int) -> str:
        SECRET_KEY = "SDFIPHAEWIPDSVNaDISAHIPDHJSPgdfsIGHDFHDFSOUGVdsoghiuOFHVOSDF"
        ALGORITHMS = "HS256"
        expire = datetime.now() + timedelta(minutes=expire)
        properties.update({"exp":expire})
        return f"Bearer {jwt.encode(properties, SECRET_KEY, algorithm=ALGORITHMS)}"

