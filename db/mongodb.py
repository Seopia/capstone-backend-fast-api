from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, AIMessage
from motor.motor_asyncio import AsyncIOMotorClient
import os
from bson import ObjectId

class Mongodb:
    def __init__(self):
        url = os.getenv("MONGODB_URI")
        client = AsyncIOMotorClient(url)
        db = client["chatbot"]
        self.chat_collection = db["messages"]

    def _filter(self, user_code, conv_id):
        f = {"userCode": user_code}
        if conv_id:
            f["convId"] = conv_id
        return f

    async def get_chat_history(self, user_code, conv_id, year=None, month=None, day=None, limit=None):
        now = datetime.now()
        year = year or now.year
        month = month or now.month
        day = day or now.day

        start = datetime(year, month, day)
        end = start + timedelta(days=1)

        query = {
            **self._filter(user_code, ObjectId(conv_id)),
            "createAt": {
                "$gte": start,
                "$lt": end,
            },
        }

        if limit:
            cursor = self.chat_collection.find(query).sort("createAt", -1).limit(limit)
            docs = await cursor.to_list(length=limit)
            docs.reverse()
        else:
            cursor = self.chat_collection.find(query).sort("createAt", 1)
            docs = await cursor.to_list(length=None)

        out = []
        for d in docs:
            role = d.get("role")
            content = d.get("content", "")
            if role == "user":
                out.append(HumanMessage(content=content))
            else:
                out.append(AIMessage(content=content))
        return out

    async def add_message(self, message, user_code, conv_id):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        await self.chat_collection.insert_one({"convId": ObjectId(conv_id),"content":message.content,"createAt":datetime.now(), "role":role,"userCode":user_code})
