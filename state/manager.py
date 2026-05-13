import logging
import threading
import time
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from state import persistence as db
from core.summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)

# 初始化数据库（模块加载时执行一次）
db.init_db()


class SessionManager:
    """多会话消息管理器（线程安全 + SQLite 持久化）"""

    def __init__(self, enable_summary: bool = True, summary_threshold: int = 20, keep_recent: int = 5, user_id: str = ""):
        self._lock = threading.RLock()
        self.sessions: dict[str, dict] = {}
        self.current_session_id: str = "default"
        self.user_id = user_id or ""
        self.enable_summary = enable_summary
        self.summarizer = ConversationSummarizer(
            threshold=summary_threshold,
            keep_recent=keep_recent,
        ) if enable_summary else None

        # 尝试从数据库加载历史会话
        self._load_from_db()

        # 如果没有加载到任何会话，创建默认会话
        if not self.sessions:
            self._create_session("default", "新对话")

    def _load_from_db(self):
        """从数据库加载所有会话和消息"""
        loaded_sessions = db.load_sessions(user_id=self.user_id)
        for s in loaded_sessions:
            session_id = s["id"]
            self.sessions[session_id] = {
                "id": session_id,
                "title": s["title"],
                "messages": [],
                "created_at": s["created_at"],
                "updated_at": s["updated_at"],
            }
            # 加载消息
            db_messages = db.load_messages(session_id, user_id=self.user_id)
            for m in db_messages:
                if m["role"] == "human":
                    msg = HumanMessage(content=m["content"], name="Human")
                elif m["role"] == "assistant":
                    msg = AIMessage(content=m["content"], name=m.get("agent_name") or "assistant")
                else:
                    continue
                self.sessions[session_id]["messages"].append(msg)

        if loaded_sessions:
            # 切换到最新的会话
            self.current_session_id = loaded_sessions[0]["id"]
            logger.info(f"Loaded {len(loaded_sessions)} sessions from database")

    def _create_session(self, session_id: str, title: str = "新对话"):
        with self._lock:
            now = time.time()
            self.sessions[session_id] = {
                "id": session_id,
                "title": title,
                "messages": [],
                "created_at": now,
                "updated_at": now,
            }
            db.save_session(session_id, title, now, now, user_id=self.user_id)

    def new_session(self, title: str = "新对话") -> str:
        """创建新会话，返回会话ID"""
        with self._lock:
            import secrets
            session_id = f"session_{secrets.token_hex(8)}"
            self._create_session(session_id, title)
            self.current_session_id = session_id
            logger.info(f"Created new session: {session_id}")
            return session_id

    def switch_session(self, session_id: str) -> bool:
        """切换到指定会话"""
        with self._lock:
            if session_id in self.sessions:
                self.current_session_id = session_id
                logger.info(f"Switched to session: {session_id}")
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        """删除会话，如果删除的是当前会话，自动切换到其他会话"""
        with self._lock:
            if session_id not in self.sessions:
                return False
            del self.sessions[session_id]
            db.delete_session(session_id, user_id=self.user_id)
            # 如果删除的是当前会话，切换到最新的
            if self.current_session_id == session_id:
                if self.sessions:
                    self.current_session_id = max(
                        self.sessions.keys(),
                        key=lambda k: self.sessions[k]["updated_at"]
                    )
                    logger.info(f"Deleted current session, switched to: {self.current_session_id}")
                else:
                    self._create_session("default", "新对话")
                    self.current_session_id = "default"
                    logger.info("Deleted all sessions, created default")
            return True

    def list_sessions(self) -> list[dict]:
        """获取会话列表，按更新时间倒序"""
        with self._lock:
            return sorted(
                [
                    {
                        "id": s["id"],
                        "title": s["title"],
                        "created_at": s["created_at"],
                        "updated_at": s["updated_at"],
                        "message_count": len(s["messages"]),
                        "is_current": s["id"] == self.current_session_id,
                    }
                    for s in self.sessions.values()
                ],
                key=lambda x: x["updated_at"],
                reverse=True,
            )

    def _current(self) -> dict:
        return self.sessions[self.current_session_id]

    def _update_session_meta(self, now: float) -> None:
        """更新会话的 updated_at 并持久化会话元数据"""
        current = self._current()
        current["updated_at"] = now
        db.save_session(self.current_session_id, current["title"],
                        current["created_at"], now, user_id=self.user_id)

    def add_human_message(self, content: str) -> HumanMessage:
        with self._lock:
            msg = HumanMessage(content=content, name="Human")
            self._current()["messages"].append(msg)
            now = time.time()
            # 更新标题（取第一条用户消息的前20字）
            if self._current()["title"] == "新对话" and len(self._current()["messages"]) == 1:
                self._current()["title"] = content[:20] + ("..." if len(content) > 20 else "")
                db.update_session_title(self.current_session_id, self._current()["title"], now, user_id=self.user_id)
            self._update_session_meta(now)
            db.save_message(self.current_session_id, "human", content, user_id=self.user_id)
            return msg

    def add_agent_message(self, content: str, agent_name: str) -> AIMessage:
        with self._lock:
            msg = AIMessage(content=content, name=agent_name)
            self._current()["messages"].append(msg)
            now = time.time()
            self._update_session_meta(now)
            db.save_message(self.current_session_id, "assistant", content, agent_name, user_id=self.user_id)
            return msg

    def get_messages(self) -> list[BaseMessage]:
        with self._lock:
            return self._current()["messages"].copy()

    def get_last_n(self, n: int) -> list[BaseMessage]:
        with self._lock:
            return self._current()["messages"][-n:]

    def get_messages_for_model(self, max_turns: int = 10, llm=None) -> list[BaseMessage]:
        """获取最近 N 轮对话（用户 + AI = 一轮），用于传给 LLM。

        避免长对话时 token 爆炸，默认保留最近 10 轮（约 20 条消息）。
        如果启用摘要且消息数超过阈值，会自动生成摘要替换旧消息。
        """
        with self._lock:
            msgs = self._current()["messages"]

            # 如果启用摘要，使用 summarizer 处理
            if self.enable_summary and self.summarizer is not None:
                return self.summarizer.prepare_messages_for_model(
                    msgs, max_turns=max_turns, llm=llm
                )

            # 否则简单截断
            max_msgs = max_turns * 2
            if len(msgs) <= max_msgs:
                return list(msgs)
            return list(msgs[-max_msgs:])

    def clear(self) -> None:
        with self._lock:
            sid = self.current_session_id
            self._current()["messages"].clear()
            self._current()["title"] = "新对话"
            now = time.time()
            self._current()["updated_at"] = now
            # 清空数据库中的消息
            db.delete_session(sid, user_id=self.user_id)
            db.save_session(sid, "新对话", now, now, user_id=self.user_id)

    def get_current_session_id(self) -> str:
        return self.current_session_id
