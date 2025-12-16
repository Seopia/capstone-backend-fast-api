FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "fastapi==0.115.12" \
    "uvicorn[standard]==0.34.2" \
    "langchain==0.3.26" \
    "langchain-google-genai==3.0.1" \
    "dotenv==1.1.0" \
    "pyjwt==2.10.1" \
    "pymongo==4.15.3"

COPY main.py .
COPY message.py .
COPY request.py .

EXPOSE 8000
CMD ["python","-m","uvicorn","main:main","--host","0.0.0.0","--port","8000"]