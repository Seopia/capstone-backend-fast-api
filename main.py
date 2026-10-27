import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from starlette.responses import JSONResponse

from router.board_router import br
from router.chat_router import cr
from router.emotion_router import er
from router.user_router import ur
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pytz import timezone
import torch    # pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

from utils.error_handler import ErrorReportingMiddleware

load_dotenv(".env")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seoul_tz = timezone("Asia/Seoul")

JWT_SECRET = os.getenv("JWT_SECRET")
CORS = os.getenv("CORS")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # await init_db()
    yield
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[CORS],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.add_middleware(ErrorReportingMiddleware)
app.include_router(cr)
app.include_router(ur)
app.include_router(er)
app.include_router(br)

@app.get("/health")
def health_check():
    return True

@app.get("/error")
def error():
    raise



@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
