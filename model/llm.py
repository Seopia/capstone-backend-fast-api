from datetime import datetime
from pytz import timezone

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os
from starlette.responses import StreamingResponse

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
                ("human", "이 history를 기반으로 모두 대화를 요약하고, 내 관점에서 일기를 작성해줘."),
            ]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "너는 사용자의 감정을 깊게 공감하는 정신 건강 상담 챗봇이고, "
                    "너는 무조건 사용자에게 오늘 하루에 있었던 일에 대하여 지속적으로 질문하고 공감해줘야한다. 친한 친구 처럼 편하게 이야기해야한다.",
                ),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )
        llm = ChatOpenAI(model="gpt-4.1-nano", openai_api_key=key)
        self._chain = prompt | llm
        self._summary_chain = summary_prompt | llm
        self._mongo = mongodb
        self._maria = mariadb
        self._seoul_tz = timezone("Asia/Seoul")

    def chat(self, message:str, user_code:int, conv_id, is_streaming=False):
        past = self._mongo.get_chat_history(user_code, conv_id)
        def get_streaming_response():
            chunks = []
            for chunk in self._chain.stream({"history": past, "input": message}):
                text = getattr(chunk, "content", str(chunk))
                chunks.append(text)
                yield text
            full = "".join(chunks)
            self._mongo.add_message(HumanMessage(content=message), user_code, conv_id)
            if full.strip():
                self._mongo.add_message(AIMessage(content=full), user_code, conv_id)
        if is_streaming:
            return StreamingResponse(get_streaming_response(), media_type="text/plain; charset=utf-8")
        else:
            full = self._chain.invoke({"history": past, "input": message})
            text = getattr(full, "content", str(full))
            self._mongo.add_message(HumanMessage(content=message), user_code, conv_id)
            if text.strip():
                self._mongo.history.add_message(AIMessage(content=text), user_code, conv_id)
            return text

    def summary(self, user_code, conv_id, year, month, day):
        past = self._mongo.get_chat_history(user_code, conv_id, year, month, day)
        result = self._summary_chain.invoke({"history": past})
        summary_text = (result.content or "").strip()

        latest = self._maria.get_latest_by_user_and_date(user_code=user_code, target_date=datetime(year, month, day, tzinfo=self._seoul_tz).date())
        create_at = datetime(year, month, day, tzinfo=self._seoul_tz).replace(tzinfo=None)
        if latest and latest.get("analysis_code"):
            self._maria.update(
                analysis_code=int(latest["analysis_code"]),
                emotion_score=float(latest.get("emotion_score") or 0.0),
                emotion_name=latest.get("emotion_name") or "",
                summary=summary_text,
                create_at=create_at,
            )
        else:
            self._maria.insert(
                user_code=user_code,
                emotion_score=0.0,
                emotion_name="",
                summary=summary_text,
                create_at=create_at,
            )
        return summary_text
