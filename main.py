import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from pytz import timezone
from db.mariadb import MariaAnalysisRepo
from db.mongodb import Mongodb
from dto.requests import ChatRequest, User, get_user, AnalyzeRequest, SummaryRequest
from model.llm import LlmModel
from model.transformer import TransformerModel
from common.jwtmiddleware import JWTMiddleware
import torch    # pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

load_dotenv(".env.prod")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seoul_tz = timezone("Asia/Seoul")

JWT_SECRET = os.getenv("JWT_SECRET")
CORS = os.getenv("CORS")
app = FastAPI()
app.add_middleware(JWTMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=[CORS],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

mariadb = MariaAnalysisRepo()
mongodb = Mongodb()
chat_model = LlmModel(mongodb, mariadb)
transformer = TransformerModel(mongodb, mariadb, device)

@app.post("/chat")
async def chat(req: ChatRequest, user:User=Depends(get_user)):
    return chat_model.chat(req.message, user.user_code, req.convId, is_streaming=True, is_jailbreak=req.isJailbreak)

@app.post("/summary")
async def summary(req: SummaryRequest, user:User=Depends(get_user)):
    return chat_model.summary(user.user_code, req.convId, req.year, req.month, req.day)

@app.post("/analyze")
async def analyze(req: AnalyzeRequest, user:User=Depends(get_user)):
    final_score, overall_emotion_label = transformer.inference(user.user_code, req.convId)
    if final_score and overall_emotion_label is None:
        return {"message": "분석할 대화가 없습니다."}
    transformer.update_db(user.user_code, final_score, overall_emotion_label)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
