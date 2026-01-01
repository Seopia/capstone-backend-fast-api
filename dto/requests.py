from pydantic import BaseModel, Field
from fastapi import Request, HTTPException


def get_user(request: Request):
    return User(**request.state.user)

class User(BaseModel):
    user_code: int = Field(alias="userCode")
    user_name: str | None = Field(default=None, alias="userName")
    role: str | None = Field(default=None, alias="role")

class ChatRequest(BaseModel):
    message: str
    convId: str
    isJailbreak: bool

class SummaryRequest(BaseModel):
    convId: str
    year: int
    month: int
    day: int

class AnalyzeRequest(BaseModel):
    convId: str