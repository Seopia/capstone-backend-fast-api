import os
import jwt

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.utils import get_authorization_scheme_param
from pytz import timezone
from db.mariadb import MariaAnalysisRepo
from db.mongodb import Mongodb
from model.llm import LlmModel
from model.transformer import TransformerModel
import torch

# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seoul_tz = timezone("Asia/Seoul")

JWT_SECRET = os.getenv("JWT_SECRET")
CORS = os.getenv("CORS")

app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=[CORS],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

mariadb = MariaAnalysisRepo()
mongodb = Mongodb()
model = LlmModel(mongodb, mariadb)
transformer = TransformerModel(mongodb, mariadb, device)


def get_token(authorization: str) -> str:
    scheme, token = get_authorization_scheme_param(authorization)
    return token

def decode_jwt(authorization: str) -> int:
    token = get_token(authorization)
    payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    user_code = payload.get("userCode")
    return int(user_code)

async def get_body_and_user(request: Request, authorization: str):
    body = await request.json()
    body_data = {}
    for b in body:
        body_data[b] = body.get(b)
    return body_data, decode_jwt(authorization)

@app.post("/chat")
async def chat(request: Request, authorization: str = Header(...)):
    req, user_code = await get_body_and_user(request, authorization)
    message = req["message"]
    conv_id = req["convId"]
    return model.chat(message, user_code, conv_id, is_streaming=True)

@app.post("/summary")
async def summary(request: Request, authorization: str = Header(...)):
    req, user_code = await get_body_and_user(request, authorization)
    conv_id, year, month, day = (req["convId"], req["year"], req["month"], req["day"])
    model_result = model.summary(user_code, conv_id, year, month, day)
    return model_result

@app.post("/analyze")
async def analyze(request: Request, authorization: str = Header(...)):
    req, user_code = await get_body_and_user(request, authorization)
    conv_id = req["convId"]
    final_score, overall_emotion_label = transformer.inference(user_code, conv_id)
    if final_score and overall_emotion_label is None:
        return {"message": "분석할 대화가 없습니다."}
    transformer.update_db(user_code, final_score, overall_emotion_label)

    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
