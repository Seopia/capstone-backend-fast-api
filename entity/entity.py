from datetime import datetime

from sqlalchemy import BigInteger, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, relationship, mapped_column

from db.mariadb_orm import Base

class User(Base):
    __tablename__ = 'user'
    user_code: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    oauth_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    nickname: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    create_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    enable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    last_login_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=True)
    profile_img: Mapped[str] = mapped_column(String(300), nullable=True)
    bio: Mapped[str] = mapped_column(String(100), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    oauth_provider: Mapped[str] = mapped_column(String(100), nullable=True)
    refresh_token: Mapped[str] = mapped_column(String(3000), nullable=True)
    chats: Mapped[list["Chat"]] = relationship(
        "Chat",
        back_populates="user",
        cascade="all, delete-orphan"
    )

class Chat(Base):
    __tablename__ = 'chat'
    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_code: Mapped[int] = mapped_column(BigInteger, ForeignKey('user.user_code'), nullable=False)
    content: Mapped[str] = mapped_column(String(6000), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    create_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    user: Mapped[User] = relationship("User", back_populates="chats")

    def __repr__(self):
        return f"Chat(chat_id: {self.chat_id}, user_code: {self.user_code}, content: {self.content}, role: {self.role}, create_at: {self.create_at})"



