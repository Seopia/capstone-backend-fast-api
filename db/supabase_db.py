import supabase
import os
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage


class SupabaseClient:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self.client = supabase.Client(url, key)

    def insert_chat(self, user_code: int, query: str, embedded_query, role: str = "user"):
        data = {
            "user_code": user_code,
            "content": query,
            "encode_content": embedded_query,
            "create_at": datetime.now().isoformat(),
            "role": role
        }
        self.client.table("chat").insert(data).execute()

    def vector_search(self, embedded_query, match_threshold:float, match_count:int) -> list[HumanMessage]:
        '''
        벡터 유사도 검색을 수행합니다.
        :param embedded_query: 사용자의 질문을 임베딩한 벡터 값으로 1536차원입니다.
        :param match_threshold: 임계값입니다. 유사도가 0.n 이상인 값만 결과에 포함시킵니다.
        :param match_count: 총 몇 개를 반환할지
        :return: HumanMessage 객체 배열입니다. 랭체인에서 곧바로 사용할 수 있습니다.
        '''
        result = self.client.rpc("vector_search", {"q": embedded_query, "match_threshold": match_threshold, "match_count": match_count}).execute()
        out = []
        for r in result.data:
            role = r['role']
            content = r['content']
            if role == "user":
                out.append(HumanMessage(content=content))
            else:
                out.append(AIMessage(content=content))
        print(out)
        return out
