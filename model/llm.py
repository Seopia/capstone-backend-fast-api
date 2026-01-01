import inspect
import os
from datetime import datetime
from pytz import timezone

from starlette.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

class LlmModel:
    def __init__(self, mongodb, mariadb):
        key = os.getenv("OPEN_AI_KEY")
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "너는 이 history 대화를 모두 요약하여 철저히 사용자의 시선에서 한 편의 일기를 써야하는 일기 마스터이다. "
                    "어투는 오늘은 ~ 했다 또는 ~가 있었다 등 과거형으로 집필해야하며, "
                    "안좋은 내용이 있더라도 객관적으로 작성해야한다. 또한 대화를 한 것을 요약하여 일기를 작성하는 것이 아니라, user가 무슨 일을 겪고, 어떤 일이 있었는지 등으로 작성해야 한다.",
                ),
                MessagesPlaceholder("history"),
                ("human", "이 history를 기반으로 모두 대화를 요약하고, 내 관점에서 제목, 날짜를 작성하고,일기를 마크다운 문법으로 작성해줘."),
            ]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "너는 사용자의 감정을 깊게 공감하는 정신 건강 상담 챗봇이고, 답변은 마크다운 문법을 사용하여 보기 좋게 작성해야한다. 제목은 # 기호를 사용하고, 목록은 *  또는 - 기호를 사용해 작성해야한다."
                    "너는 무조건 사용자에게 오늘 하루에 있었던 일에 대하여 지속적으로 질문하고 공감해줘야한다 또한 가능하면 해결책을 제시해야한다. 친한 친구 처럼 편하게 이야기해야한다. 유저의 채팅이 짧으면 짧게, 길면 길게 답변해야한다.",
                ),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )
        self.llm = ChatOpenAI(model="gpt-4.1-nano", openai_api_key=key)
        self._chain = prompt | self.llm
        self._summary_chain = summary_prompt | self.llm
        self._mongo = mongodb
        self._maria = mariadb
        self._seoul_tz = timezone("Asia/Seoul")

    async def chat(self, message:str, rag_history:list, user_code:int, conv_id, is_streaming=False, is_jailbreak=False, callback=None):
        recent_messages = await self._mongo.get_chat_history(user_code, conv_id, limit=10)
        context_message = None
        if rag_history:
            rag_context_str = "다음은 사용자의 과거 대화 내용 중, 현재 질문과 유사한 내용입니다. 답변 시 참고하세요:\n"
            for msg in rag_history:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                rag_context_str += f"- {role}: {msg.content}\n"
            context_message = SystemMessage(content=rag_context_str)
        
        final_history = recent_messages
        if context_message:
            final_history = [context_message] + recent_messages

        async def get_streaming_response():
            chunks = []
            if is_jailbreak is False:
                async for chunk in self._chain.astream({"history": final_history, "input": message}):
                    text = getattr(chunk, "content", str(chunk))
                    chunks.append(text)
                    yield text
                full = "".join(chunks)
                await self._mongo.add_message(HumanMessage(content=message), user_code, conv_id)
                if full.strip():
                    await self._mongo.add_message(AIMessage(content=full), user_code, conv_id)
                    if callback:
                         if inspect.iscoroutinefunction(callback):
                             await callback(full)
                         else:
                             callback(full)
            else:
                msgs = final_history + [HumanMessage(content=message)]
                async for chunk in self.llm.astream(msgs):
                    text = getattr(chunk, "content", str(chunk))
                    chunks.append(text)
                    yield text
                full = "".join(chunks)
                await self._mongo.add_message(HumanMessage(content=message), user_code, conv_id)
                if full.strip():
                    await self._mongo.add_message(AIMessage(content=full), user_code, conv_id)
                    if callback:
                         if inspect.iscoroutinefunction(callback):
                             await callback(full)
                         else:
                             callback(full)
        if is_streaming:
            return StreamingResponse(get_streaming_response(), media_type="text/plain; charset=utf-8")
        else:
            full = await self._chain.ainvoke({"history": final_history, "input": message})
            text = getattr(full, "content", str(full))
            await self._mongo.add_message(HumanMessage(content=message), user_code, conv_id)
            if text.strip():
                await self._mongo.add_message(AIMessage(content=text), user_code, conv_id)
                if callback:
                        if inspect.iscoroutinefunction(callback):
                            await callback(text)
                        else:
                            callback(text)
            return text

    async def summary(self, user_code, conv_id, year, month, day):
        past = await self._mongo.get_chat_history(user_code, conv_id, year, month, day)
        result = await self._summary_chain.ainvoke({"history": past})
        summary_text = (result.content or "").strip()

        latest = await self._maria.get_latest_by_user_and_date(user_code=user_code, target_date=datetime(year, month, day, tzinfo=self._seoul_tz).date())
        create_at = datetime(year, month, day, tzinfo=self._seoul_tz).replace(tzinfo=None)
        if latest and latest.get("analysis_code"):
            await self._maria.update(
                analysis_code=int(latest["analysis_code"]),
                emotion_score=float(latest.get("emotion_score") or 0.0),
                emotion_name=latest.get("emotion_name") or "",
                summary=summary_text,
                create_at=create_at,
            )
        else:
            await self._maria.insert(
                user_code=user_code,
                emotion_score=0.0,
                emotion_name="",
                summary=summary_text,
                create_at=create_at,
            )
        return summary_text
