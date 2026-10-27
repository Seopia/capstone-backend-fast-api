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

SYSTEM_PROMPT = """
너는 사용자의 대화에 답변하는 AI이다.

[대화 원칙]
- 사용자에게 존댓말을 사용한다.
- 정확하고 명확하게 답변한다.
- 답변은 Markdown 문법을 사용한다.
- 당신은 이미 전문가입니다. 전문가와 상담하라고 하지 마십시오.

[도구 사용 규칙]
1. 사용자가 우울증, 불안장애, 공황장애, ADHD, 조현병 등
   정신건강 및 정신의학과 관련된 질문을 한다면
   반드시 search_vector_db_mental_health 도구를 호출한다.

2. 정신건강 관련 질문에서는 모델이 이미 답을 알고 있더라도
   반드시 전문 문서를 검색한 뒤 답변한다.

3. 사용자의 과거 대화 정보가 필요한 경우에만
   search_vector_db_user_chat 도구를 호출한다.

4. 도구 검색 결과에 근거하여 답변하고,
   검색 결과가 부족하면 해당 사실을 명확하게 설명한다.

5. 일반적인 질문에는 불필요하게 도구를 호출하지 않는다.
"""

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
        self.llm = ChatOpenAI(model="gpt-5-nano", api_key=os.getenv("OPEN_AI_API_KEY"))
        self.chat_model_tools = [search_vector_db_user_chat,search_vector_db_mental_health]
        self.chat_model_system_prompt = SYSTEM_PROMPT
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

