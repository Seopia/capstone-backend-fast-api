from dto.requests import ChatRequest, AnalyzeRequest, SummaryRequest
from model.llm import LlmModel
from model.transformer import TransformerModel
from db.supabase_db import SupabaseClient
from langchain_openai import OpenAIEmbeddings

import asyncio

class ChatService:
    def __init__(self, chat_model: LlmModel, supabase_db: SupabaseClient, embedding_model: OpenAIEmbeddings, transformer: TransformerModel):
        self.chat_model = chat_model
        self.supabase_db = supabase_db
        self.embedding_model = embedding_model
        self.transformer = transformer

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
