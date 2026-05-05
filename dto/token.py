from pydantic import BaseModel


class DecodedToken(BaseModel):
    user_code: int
    oauth_id: str
    nickname: str
    profile_image: str
    thumbnail_image: str
    exp: int