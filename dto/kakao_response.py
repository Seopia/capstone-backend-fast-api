from datetime import datetime

from pydantic import BaseModel


class KaKaoTokenResponse(BaseModel):
    token_type: str
    access_token: str
    id_token: str | None = None
    expires_in: int
    refresh_token: str
    refresh_token_expires_in: int
    scope: str | None = None

class KaKaoUserResponse(BaseModel):
    id: int
    connected_at: datetime
    properties: dict
    kakao_account: dict