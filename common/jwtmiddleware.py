from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from fastapi import Request, Response
import os
import jwt
from starlette.responses import JSONResponse


class JWTMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "OPTIONS":
            print("씨발 진짜")
            return Response(status_code=200)
        secret_key = os.getenv('JWT_SECRET')
        auth = request.headers.get('Authorization')
        if not auth:
            return JSONResponse(status_code=403, content="로그인 후 이용가능해요")

        token = auth.replace('Bearer ', '').strip()

        try:
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            request.state.user = payload
        except jwt.ExpiredSignatureError:
            return JSONResponse(status_code=401, content="로그인이 만료되었어요.")
        except jwt.InvalidTokenError:
            return JSONResponse(status_code=401, content="로그인이 유효하지 않아요.")
        return await call_next(request)