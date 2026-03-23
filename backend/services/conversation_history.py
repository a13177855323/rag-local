import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from backend.config import settings


@dataclass
class ConversationTurn:
    turn_id: str
    question: str
    answer: str
    sources: List[Dict]
    timestamp: str
    response_time_ms: Optional[float] = None
    question_category: Optional[str] = None
    quality_score: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationTurn":
        return cls(**data)


@dataclass
class ConversationSession:
    session_id: str
    created_at: str
    updated_at: str
    turns: List[ConversationTurn]
    metadata: Dict

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [t.to_dict() for t in self.turns],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationSession":
        turns = [ConversationTurn.from_dict(t) for t in data.get("turns", [])]
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            turns=turns,
            metadata=data.get("metadata", {})
        )


class ConversationHistoryManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.history_dir = os.path.join(settings.VECTOR_DB_PATH, "conversations")
        os.makedirs(self.history_dir, exist_ok=True)
        self.current_session: Optional[ConversationSession] = None
        self._load_or_create_session()

    def _get_session_file(self, session_id: str) -> str:
        return os.path.join(self.history_dir, f"{session_id}.json")

    def _load_or_create_session(self):
        today = datetime.now().strftime("%Y-%m-%d")
        session_id = f"session_{today}"
        session_file = self._get_session_file(session_id)

        if os.path.exists(session_file):
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.current_session = ConversationSession.from_dict(data)
        else:
            self.current_session = ConversationSession(
                session_id=session_id,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                turns=[],
                metadata={"total_turns": 0}
            )
            self._save_session()

    def _save_session(self):
        if self.current_session:
            session_file = self._get_session_file(self.current_session.session_id)
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_session.to_dict(), f, ensure_ascii=False, indent=2)

    def add_turn(
        self,
        question: str,
        answer: str,
        sources: List[Dict],
        response_time_ms: float = None
    ) -> ConversationTurn:
        if not self.current_session:
            self._load_or_create_session()

        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            question=question,
            answer=answer,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            response_time_ms=response_time_ms
        )

        self.current_session.turns.append(turn)
        self.current_session.updated_at = datetime.now().isoformat()
        self.current_session.metadata["total_turns"] = len(self.current_session.turns)
        self._save_session()

        return turn

    def update_turn_analysis(
        self,
        turn_id: str,
        category: str = None,
        quality_score: float = None
    ):
        if not self.current_session:
            return

        for turn in self.current_session.turns:
            if turn.turn_id == turn_id:
                if category:
                    turn.question_category = category
                if quality_score is not None:
                    turn.quality_score = quality_score
                break

        self._save_session()

    def get_current_session(self) -> Optional[ConversationSession]:
        return self.current_session

    def get_all_sessions(self) -> List[Dict]:
        sessions = []
        for filename in os.listdir(self.history_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.history_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "turn_count": len(data.get("turns", []))
                })
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        session_file = self._get_session_file(session_id)
        if os.path.exists(session_file):
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ConversationSession.from_dict(data)
        return None

    def get_all_turns(self) -> List[ConversationTurn]:
        all_turns = []
        for filename in os.listdir(self.history_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.history_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for turn_data in data.get("turns", []):
                    turn = ConversationTurn.from_dict(turn_data)
                    all_turns.append(turn)
        return sorted(all_turns, key=lambda x: x.timestamp, reverse=True)

    def delete_session(self, session_id: str) -> bool:
        session_file = self._get_session_file(session_id)
        if os.path.exists(session_file):
            os.remove(session_file)
            if self.current_session and self.current_session.session_id == session_id:
                self._load_or_create_session()
            return True
        return False

    def clear_all_history(self) -> int:
        count = 0
        for filename in os.listdir(self.history_dir):
            if filename.endswith('.json'):
                os.remove(os.path.join(self.history_dir, filename))
                count += 1
        self._load_or_create_session()
        return count

    def start_new_session(self) -> ConversationSession:
        self._save_session()
        self._load_or_create_session()
        return self.current_session


conversation_history = ConversationHistoryManager()


def get_conversation_history() -> ConversationHistoryManager:
    return conversation_history
