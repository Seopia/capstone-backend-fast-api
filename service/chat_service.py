import json

from langchain_core.messages import AIMessage, ToolMessage
from pymongo import MongoClient

from dto.token import DecodedToken
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
    def __init__(self):
        self.repo = ChatRepo()
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPEN_AI_API_KEY"))
        self.chat_model_tools = [search_vector_db_user_chat,search_vector_db_mental_health]
        self.chat_model_system_prompt = "너는 사용자의 대화에 답변하는 AI이다.\n대화 기록만으로 답변할 수 있으면 도구를 사용하지 마라.\n사용자를 존중하며 높임말로 대답해라.\n당신은 전문가다. 전문가를 추천하지 말아라.\n도구를 사용했다면 그 내용을 바탕으로 자연스럽게 대답해라.\n답변할 때는 가독성을 높이기 위해 반드시 마크다운(Markdown) 문법을 적극적으로 활용해라. (예: 굵은 글씨, 목록, 표, 인용구 등)"
    async def get_chats(self, db:AsyncSession, user:DecodedToken, count:int=10):
        return await self.repo.get_chats(db, user, count)

    async def get_chats_by_page(self, db:AsyncSession, user:DecodedToken, page:int, size:int=20):
        chats = await self.repo.get_chats_by_page(db, user, page, size)
        is_last = len(chats) < size
        content = [
            {
                "content": chat.content,
                "role": "user" if chat.role == "human" else "ai",
                "createAt": [
                    chat.create_at.year,
                    chat.create_at.month,
                    chat.create_at.day,
                    chat.create_at.hour,
                    chat.create_at.minute,
                    chat.create_at.second,
                    chat.create_at.microsecond * 1000
                ]
            }
            for chat in chats
        ]
        return {
            "last": is_last,
            "content": content
        }

    async def response_llm(self, content: str, chats: list, user, db:AsyncSession):
        yield json.dumps({
            "type": "status",
            "message": "생각 중..."
        }, ensure_ascii=False) + "\n"
        agent = create_agent(
            model=self.llm,
            tools=self.chat_model_tools,
            system_prompt=self.chat_model_system_prompt
        )
        chat_text = "\n".join(f"{chat.role}: {chat.content}"for chat in chats)
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
        # 디비 저장
        await self.repo.insert_chat(content, final_answer, user, db)
        yield json.dumps({
            "type": "final",
            "answer": final_answer
        }, ensure_ascii=False) + "\n"

    # async def summary(self, req: SummaryRequest, user_code: int):
    #     return await self.chat_model.summary(user_code, req.convId, req.year, req.month, req.day)
    #
    # async def analyze(self, req: AnalyzeRequest, user_code: int):
    #
    #
    #
    #     final_score, overall_emotion_label = await self.transformer.inference(user_code, req.convId)
    #     if final_score is not None and overall_emotion_label is None:
    #          pass
    #
    #     if final_score is None:
    #          return {"message": "분석할 대화가 없습니다."}
    #
    #     await self.transformer.update_db(user_code, final_score, overall_emotion_label)
    #     return {"message": "분석 완료", "score": final_score, "emotion": overall_emotion_label}


