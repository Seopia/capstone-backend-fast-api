import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient

from dto.requests import ChatRequest, AnalyzeRequest, SummaryRequest
from dto.token import DecodedToken
from model.llm import LlmModel
from model.transformer import TransformerModel
from db.supabase_db import SupabaseClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import create_agent
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from repo.chat_repo import ChatRepo
from langchain_core.tools import tool
import os
@tool
async def search_vector_db_user_chat(query: str):
    """이 도구는 사용자의 현재 질문으로 사용자와 전에 했던 말을 벡터 검색하는 도구입니다. 사용자 대화에서 꼭 필요하다고 판단될 때 사용하세요. 매개변수는 사용자의 질문입니다."""
    print("도구 호출함")
    embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPEN_AI_API_KEY"))
    embedded_query = await asyncio.to_thread(embedding_model.embed_query, query)
    return "사용자는 백앤드 서버로 파일을 업로드 할 때 발생하는 에러를 파일을 청킹하여 업로드하여 해결했다."

@tool
async def search_vector_db_mental_health(query: str):
    """이 도구는 전문적인 사용자 멘탈 케어를 위한 의학 도서관입니다. 사용자가 정신의학에 관한 질문을 하면 이 도구를 활용하세요. 매개변수는 사용자의 질문입니다."""
    print("정신의학 도구 호출함")
    MONGO = os.getenv("MONGO")
    embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPEN_AI_API_KEY"))
    embedded_query = await asyncio.to_thread(embedding_model.embed_query, query)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedded_query,
                "numCandidates": 100,
                "limit": 5
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "metadata": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]
    client = MongoClient(MONGO)
    db = client["mentalcare_chat_bot"]
    collection = db["docs"]
    print(list(collection.aggregate(pipeline)))
    return list(collection.aggregate(pipeline))

class ChatService:
    def __init__(self, chat_model: LlmModel, supabase_db: SupabaseClient, embedding_model: OpenAIEmbeddings, transformer: TransformerModel):
        self.chat_model = chat_model
        self.supabase_db = supabase_db
        self.embedding_model = embedding_model
        self.transformer = transformer
        self.repo = ChatRepo()

    async def get_chats(self, db:AsyncSession, user:DecodedToken, count:int=10):
        return await self.repo.get_chats(db, user, count)

    async def response_llm(self, content: str, chats: list, user):
        yield json.dumps({
            "type": "status",
            "message": "생각 중..."
        }, ensure_ascii=False) + "\n"
        key = os.getenv("OPEN_AI_API_KEY")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=key,
            temperature=0
        )
        tools = [search_vector_db_user_chat,search_vector_db_mental_health]
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=(
                "너는 사용자의 대화에 답변하는 AI다. "
                "대화 기록만으로 답변할 수 있으면 도구를 사용하지 마라. "
                "답변에 필요한 정보가 대화 기록에 없거나, 외부 지식/문서 검색이 필요하다고 판단될 때만 도구를 사용해라. "
                "사용자를 존중하며 높임말로 대답해라."
                "당신은 충분한 전문가이다. 전문가를 추천하지 말고 자신있게 답변해라."
                "도구 결과를 사용했다면 그 내용을 바탕으로 자연스럽게 답변해라."
            )
        )
        chat_text = "\n".join(
            f"{chat.role}: {chat.content}"
            for chat in chats
        )
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
    [이전 대화]
    {chat_text}

    [현재 사용자 질문]
    {content}
    """
                }
            ]
        }
        final_answer = ""
        async for chunk in agent.astream(payload, stream_mode="updates"):
            for node_name, node_data in chunk.items():
                messages = node_data.get("messages", [])

                for msg in messages:
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for tool_call in msg.tool_calls:
                            yield json.dumps({
                                "type": "tool_call",
                                "message": "도구를 호출했어요.",
                                "tool_name": tool_call["name"],
                                "args": tool_call["args"]
                            }, ensure_ascii=False) + "\n"

                    elif isinstance(msg, ToolMessage):
                        yield json.dumps({
                            "type": "tool_result",
                            "message": "도구 결과를 받았어요.",
                            "tool_name": msg.name
                        }, ensure_ascii=False) + "\n"

                    elif isinstance(msg, AIMessage) and msg.content:
                        final_answer = msg.content

        yield json.dumps({
            "type": "final",
            "answer": final_answer
        }, ensure_ascii=False) + "\n"


    async def chat(self, req: ChatRequest, user_code: int):
        message = req.message
        # Offload blocking embedding and supabase calls
        embedded_query = await asyncio.to_thread(self.embedding_model.embed_query, message)
        vector_search_history = await asyncio.to_thread(self.supabase_db.vector_search, embedded_query, 0.6, 10)
        await asyncio.to_thread(self.supabase_db.insert_chat, user_code, message, embedded_query)

        async def on_complete(ai_response: str):
             embedded = await asyncio.to_thread(self.embedding_model.embed_query, ai_response)
             await asyncio.to_thread(self.supabase_db.insert_chat, user_code, ai_response, embedded, "assistant")

        return await self.chat_model.chat(message, vector_search_history, user_code, req.convId, is_streaming=True, is_jailbreak=req.isJailbreak, callback=on_complete)

    async def summary(self, req: SummaryRequest, user_code: int):
        return await self.chat_model.summary(user_code, req.convId, req.year, req.month, req.day)

    async def analyze(self, req: AnalyzeRequest, user_code: int):
        final_score, overall_emotion_label = await self.transformer.inference(user_code, req.convId)
        if final_score is not None and overall_emotion_label is None:
             pass

        if final_score is None:
             return {"message": "분석할 대화가 없습니다."}

        await self.transformer.update_db(user_code, final_score, overall_emotion_label)
        return {"message": "분석 완료", "score": final_score, "emotion": overall_emotion_label}


