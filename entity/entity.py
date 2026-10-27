from datetime import datetime

from sqlalchemy import BigInteger, String, DateTime, Boolean, ForeignKey, Float
from sqlalchemy.orm import Mapped, relationship, mapped_column

from db.mariadb_orm import Base

class User(Base):
    __tablename__ = 'users'
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
    analysis_result: Mapped[list["AnalysisResult"]] = relationship(
        "AnalysisResult",
        back_populates="user",
        cascade="all, delete-orphan"
    )

class Chat(Base):
    __tablename__ = 'chat'
    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_code: Mapped[int] = mapped_column(BigInteger, ForeignKey('users.user_code'), nullable=False)
    content: Mapped[str] = mapped_column(String(6000), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    create_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    user: Mapped[User] = relationship("User", back_populates="chats")

    def __repr__(self):
        return f"Chat(chat_id: {self.chat_id}, user_code: {self.user_code}, content: {self.content}, role: {self.role}, create_at: {self.create_at})"


class AnalysisResult(Base):
    __tablename__ = 'analysis_result'
    analysis_code: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    happy: Mapped[float] = mapped_column(Float, nullable=True)
    anger:Mapped[float] = mapped_column(Float, nullable=True)
    anxiety: Mapped[float] = mapped_column(Float, nullable=True)
    sadness: Mapped[float] = mapped_column(Float, nullable=True)
    calmness: Mapped[float] = mapped_column(Float, nullable=True)
    confusion: Mapped[float] = mapped_column(Float, nullable=True)
    create_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    emotion_name: Mapped[str] = mapped_column(String(100), nullable=False)
    summary: Mapped[str] = mapped_column(String(6000), nullable=True)
    user_code: Mapped[int] = mapped_column(BigInteger, ForeignKey('users.user_code'), nullable=False)
    user: Mapped[User] = relationship("User", back_populates="analysis_result")
    posts: Mapped[list["Post"]] = relationship(
        "Post",
        back_populates="analysis_result",
        cascade="all, delete-orphan"
    )
    def __repr__(self):
        return f"Chat(analysis_code: {self.analysis_code}, create_at: {self.create_at}, emotion_name: {self.emotion_name}, summary: {self.summary}, user_code: {self.user_code})"


class Post(Base):
    __tablename__ = 'posts'
    post_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_code: Mapped[int] = mapped_column(BigInteger, ForeignKey('users.user_code'), nullable=False)
    analysis_code: Mapped[int] = mapped_column(BigInteger, ForeignKey('analysis_result.analysis_code'), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(String(6000), nullable=True)  # 첨언
    create_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    user: Mapped[User] = relationship("User")
    analysis_result: Mapped[AnalysisResult] = relationship("AnalysisResult", back_populates="posts")
    comments: Mapped[list["Comment"]] = relationship(
        "Comment",
        back_populates="post",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"Post(post_id: {self.post_id}, user_code: {self.user_code}, title: {self.title}, create_at: {self.create_at})"


class Comment(Base):
    __tablename__ = 'comments'
    comment_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(BigInteger, ForeignKey('posts.post_id'), nullable=False)
    user_code: Mapped[int] = mapped_column(BigInteger, ForeignKey('users.user_code'), nullable=False)
    content: Mapped[str] = mapped_column(String(2000), nullable=False)
    create_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    post: Mapped[Post] = relationship("Post", back_populates="comments")
    user: Mapped[User] = relationship("User")

    def __repr__(self):
        return f"Comment(comment_id: {self.comment_id}, post_id: {self.post_id}, user_code: {self.user_code}, create_at: {self.create_at})"

