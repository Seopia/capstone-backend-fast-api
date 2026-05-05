from pydantic import BaseModel, Field



# class User(BaseModel):
#     user_code: int = Field(alias="userCode")
#     user_name: str | None = Field(default=None, alias="userName")
#     role: str | None = Field(default=None, alias="role")

class ChatRequest(BaseModel):
    content: str

class SummaryRequest(BaseModel):
    convId: str
    year: int
    month: int
    day: int

class AnalyzeRequest(BaseModel):
    convId: str