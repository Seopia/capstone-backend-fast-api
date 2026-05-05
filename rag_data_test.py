import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
MONGODB_URI = os.getenv("MONGO")
DB_NAME = 'mentalcare_chat_bot'
COLLECTION_NAME = 'docs'

# 1. MongoDB Atlas 연결
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


# 2. OpenAI 임베딩 모델
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)


# 3. 텍스트 분할기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=130,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


# 4. txt 파일 읽기
with open("data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 5. 청킹
chunks = text_splitter.split_text(raw_text)

# 6. chunk 여러 개를 한 번에 임베딩
vectors = embeddings.embed_documents(chunks)


# 7. MongoDB에 넣을 문서 형태로 변환
mongo_docs = []

for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
    mongo_docs.append({
        "text": chunk,
        "embedding": vector,
        "metadata": {
            "source": "mental_health_guide",
            "chunk_id": i,
            "chunk_size": len(chunk),
            "embedding_model": "text-embedding-3-small",
            "created_at": datetime.now(timezone.utc),
        }
    })


# 8. 기존 데이터 삭제 여부
# 테스트 중이면 중복 방지를 위해 삭제하고 다시 넣는 게 편함
collection.delete_many({
    "metadata.source": "mental_health_guide"
})


# 9. MongoDB Atlas에 insert
result = collection.insert_many(mongo_docs)

print(f"생성된 chunk 수: {len(chunks)}")
print(f"생성된 embedding 수: {len(vectors)}")
print(f"저장된 MongoDB document 수: {len(result.inserted_ids)}")
print(f"임베딩 차원 수: {len(vectors[0])}")