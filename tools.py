"""
Функції-інструменти екзаменатора: start_exam, get_next_topic, end_exam.

"База даних" спрощена до JSON-файлів:
  data/students.json    – список студентів
  data/topics.json      – список тем
  results/exams.jsonl   – журнал початку іспитів
  results/results.jsonl – журнал результатів
  results/<email>_<час>.json – повний протокол кожного іспиту
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypedDict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TOPICS_PER_EXAM = (2, 3)  # мінімум і максимум тем на іспит

# Рівні складності. "style" - як ставити запитання, "grading" - чого очікувати на 10/10.
LEVELS = {
    "basic": {
        "title": "Початковий",
        "emoji": "🟢",
        "description": "Базові поняття та інтуїція, без формул.",
        "style": "Питай про базові означення та інтуїцію, простими словами, з прикладами. "
                 "Не вимагай формул і технічних деталей. Одне просте запитання за раз.",
        "grading": "10/10 - студент правильно пояснює суть поняття своїми словами і наводить приклад.",
    },
    "intermediate": {
        "title": "Середній",
        "emoji": "🟡",
        "description": "Як працюють методи: механізми, основні формули, порівняння підходів.",
        "style": "Питай, як працюють методи: механізми, основні формули, порівняння підходів, "
                 "типові проблеми та їх розв'язання.",
        "grading": "10/10 - студент пояснює механізм, порівнює підходи і знає основні формули.",
    },
    "advanced": {
        "title": "Складний",
        "emoji": "🔴",
        "description": "Глибоке розуміння: чому це працює, компроміси, сучасні методи, поглиблені запитання.",
        "style": "Питай про глибоке розуміння: чому метод працює, математичні обґрунтування, "
                 "компроміси, граничні випадки, сучасні варіанти й практичні проблеми. "
                 "Став уточнення «а чому?», «а що буде, якщо...?».",
        "grading": "10/10 - студент обґрунтовує, аналізує компроміси і впевнено відповідає на поглиблені запитання.",
    },
}
DEFAULT_LEVEL = "intermediate"


class Message(TypedDict):
    role: str       # "system" | "user" | "tool_call"
    content: str
    datetime: str   # "2025-12-08T13:08:50"


class StudentNotFoundError(Exception):
    pass


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


# ---------------------------------------------------------------- дані

def load_students() -> list[dict]:
    with open(DATA_DIR / "students.json", encoding="utf-8") as f:
        return json.load(f)


def load_topics() -> list[dict]:
    with open(DATA_DIR / "topics.json", encoding="utf-8") as f:
        return json.load(f)


def topic_key_points(topic_name: str, level: str = DEFAULT_LEVEL) -> str:
    """Орієнтири для екзаменатора: що саме питати з цієї теми на заданому рівні."""
    for t in load_topics():
        if t["name"] == topic_name:
            return t.get("levels", {}).get(level, "")
    return ""


def _normalize_name(name: str) -> set[str]:
    # "Петренко Іван" і "іван петренко" вважаємо однаковими
    return {part for part in name.lower().replace("'", "’").split() if part}


def find_student(email: str, name: str) -> dict | None:
    email = email.strip().lower()
    for s in load_students():
        if s["email"].lower() == email and _normalize_name(s["name"]) == _normalize_name(name):
            return s
    return None


def _append_jsonl(path: Path, record: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- стан сесії

@dataclass
class ExamSession:
    """Стан одного іспиту (одного користувача у вебінтерфейсі)."""
    email: str | None = None
    name: str | None = None
    level: str = DEFAULT_LEVEL   # обирається до старту, далі не змінюється
    topics: list[str] = field(default_factory=list)       # усі теми іспиту
    remaining: list[str] = field(default_factory=list)    # ще не видані
    current_topic: str | None = None
    topic_grades: dict[str, dict] = field(default_factory=dict)
    history: list[Message] = field(default_factory=list)  # історія у форматі завдання
    llm_messages: list[dict] = field(default_factory=list)  # історія у форматі LLM API
    pending_end: dict | None = None   # end_exam відкладено до кінця ходу
    finished: bool = False
    final_score: float | None = None

    @property
    def started(self) -> bool:
        return bool(self.topics)

    def log(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content, "datetime": now_iso()})


# ---------------------------------------------------------------- функції із завдання

def start_exam(email: str, name: str, level: str = DEFAULT_LEVEL) -> list[str]:
    """Записує початок іспиту і повертає 2-3 випадкові теми.
    Якщо студента немає в базі, кидає StudentNotFoundError."""
    if level not in LEVELS:
        raise ValueError(f"Невідомий рівень: {level}")
    student = find_student(email, name)
    if student is None:
        raise StudentNotFoundError(
            f"Студента з іменем '{name}' та email '{email}' не знайдено в базі."
        )
    all_topics = [t["name"] for t in load_topics()]
    k = random.randint(*TOPICS_PER_EXAM)
    topics = random.sample(all_topics, k)
    _append_jsonl(RESULTS_DIR / "exams.jsonl", {
        "event": "start", "email": student["email"], "name": student["name"],
        "level": level, "topics": topics, "datetime": now_iso(),
    })
    return topics


def get_next_topic(session: ExamSession) -> str:
    """Повертає назву наступної теми або порожній рядок, якщо теми закінчилися.
    (У завданні функція без аргументів; тут сесія передається явно,
    щоб кілька студентів могли складати іспит одночасно.)"""
    if not session.remaining:
        session.current_topic = None
        return ""
    session.current_topic = session.remaining.pop(0)
    return session.current_topic


def end_exam(email: str, score: float, history: list[Message], level: str | None = None) -> None:
    """Записує результат іспиту. score: 0..10, history: повна історія чату,
    level: рівень складності, на якому складали іспит."""
    score = max(0.0, min(10.0, float(score)))
    finished_at = now_iso()
    RESULTS_DIR.mkdir(exist_ok=True)
    safe_email = email.replace("@", "_at_").replace("/", "_")
    protocol_path = RESULTS_DIR / f"{safe_email}_{finished_at.replace(':', '-')}.json"
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump({"email": email, "score": score, "level": level, "finished_at": finished_at,
                   "history": history}, f, ensure_ascii=False, indent=2)
    _append_jsonl(RESULTS_DIR / "results.jsonl", {
        "email": email, "score": score, "level": level, "datetime": finished_at,
        "protocol": protocol_path.name,
    })
    print(f"[end_exam] {email}: {score}/10 -> {protocol_path.name}")


def read_results() -> list[dict]:
    path = RESULTS_DIR / "results.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
