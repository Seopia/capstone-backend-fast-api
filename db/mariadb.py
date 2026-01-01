import os
import aiomysql
from datetime import datetime, date
from typing import Optional, Dict, Any

class MariaAnalysisRepo:
    def __init__(self):
        self.host = os.getenv("MARIADB_HOST")
        self.port = int(os.getenv("MARIADB_PORT", "3306"))
        self.db = os.getenv("MARIADB_DB")
        self.user = os.getenv("MARIADB_USER")
        self.password = os.getenv("MARIADB_PASSWORD")
        self.pool = None

    async def init_pool(self):
        if not self.pool:
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db,
                charset="utf8mb4",
                cursorclass=aiomysql.DictCursor,
                autocommit=True,
            )

    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()

    async def get_latest_by_user_and_date(self, user_code: int, target_date: date) -> Optional[Dict[str, Any]]:
        if not self.pool:
             await self.init_pool()
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                sql = """
                SELECT analysis_code, user_code, emotion_score, emotion_name, summary, create_at
                FROM analysis_result
                WHERE user_code = %s AND DATE(create_at) = %s
                ORDER BY create_at DESC
                LIMIT 1
                """
                await cur.execute(sql, (user_code, target_date.strftime("%Y-%m-%d")))
                row = await cur.fetchone()
                return row

    async def insert(self, user_code: int, emotion_score: float, emotion_name: str, summary: str, create_at: datetime) -> int:
        if not self.pool:
             await self.init_pool()
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                sql = """
                INSERT INTO analysis_result (user_code, emotion_score, emotion_name, summary, create_at)
                VALUES (%s, %s, %s, %s, %s)
                """
                await cur.execute(
                    sql,
                    (
                        user_code,
                        float(emotion_score),
                        (emotion_name[:25] if summary is not None else None),
                        (summary[:3000] if summary is not None else None),
                        create_at,
                    ),
                )
                return cur.lastrowid

    async def update(self, analysis_code: int, emotion_score: float, emotion_name: str, summary: str, create_at: datetime) -> None:
        if not self.pool:
             await self.init_pool()
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                sql = """
                UPDATE analysis_result
                SET emotion_score=%s, emotion_name=%s, summary=%s, create_at=%s
                WHERE analysis_code=%s
                """
                await cur.execute(
                    sql,
                    (
                        float(emotion_score),
                        (emotion_name or "")[:25],
                        (summary[:3000] if summary is not None else None),
                        create_at,
                        analysis_code,
                    ),
                )
