import requests
from datetime import datetime

response = requests.post("http://localhost:5678/webhook/fastapi-error", {
    "service": "emotion-chatbot",
    "level": "ERROR",
    "status": 500,
    "error": "Internal Server Error",
    "message": "AI 모델 추론 중 예외 발생",
    "endpoint": "/api/chat",
    "timestamp": datetime.now().astimezone().isoformat()
})

print(response.status_code)
print(response.text)