"""
LLM-агент-екзаменатор: опис інструментів для моделі та цикл tool calling.
Працює з будь-яким OpenAI-сумісним API (Google Gemini, OpenAI, Groq, ...).
"""

from __future__ import annotations

import json

import openai
from openai import OpenAI

import tools
from prompts import GREETING, build_system_prompt
from tools import ExamSession

PROVIDERS = {
    "Google Gemini (безплатно)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.5-flash"],
        "key_url": "https://aistudio.google.com/apikey",
        "secret": "GEMINI_API_KEY",
    },
    "Groq (безплатно)": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b"],
        "key_url": "https://console.groq.com/keys",
        "secret": "GROQ_API_KEY",
    },
    "xAI Grok": {
        "base_url": "https://api.x.ai/v1",
        "models": ["grok-4.7", "grok-4-1-fast-non-reasoning", "grok-3-mini"],
        "key_url": "https://console.x.ai",
        "secret": "XAI_API_KEY",
    },
}

MAX_STEPS_PER_TURN = 12  # захист від зациклення

# Коди, за яких є сенс спробувати іншу модель того ж провайдера:
# 5xx - сервер перевантажений, 429 - вичерпано ліміт саме цієї моделі,
# 404 - модель перейменували або вимкнули.
FALLBACK_STATUSES = {404, 429, 500, 502, 503, 504, 529}

# ---------------------------------------------------------------- опис інструментів для LLM

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "start_exam",
            "description": "Починає іспит: перевіряє студента в базі й повертає 2-3 теми. "
                           "Викликай, коли знаєш ім'я та email студента.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Email студента"},
                    "name": {"type": "string", "description": "Ім'я та прізвище студента"},
                },
                "required": ["email", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_next_topic",
            "description": "Повертає наступну тему, рівень складності іспиту та приховані орієнтири "
                           "для екзаменатора саме для цього рівня. Якщо тем більше немає, повертає topic = null.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grade_topic",
            "description": "Записує оцінку за поточну тему. Викликай перед переходом до наступної теми.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Назва теми"},
                    "score": {"type": "number", "description": "Оцінка за тему від 0 до 10"},
                    "comment": {"type": "string", "description": "Коротке обґрунтування оцінки"},
                },
                "required": ["topic", "score", "comment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_exam",
            "description": "Завершує іспит і записує підсумкову оцінку. Історія чату додається автоматично.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Email студента"},
                    "score": {"type": "number", "description": "Підсумкова оцінка від 0 до 10"},
                },
                "required": ["email", "score"],
            },
        },
    },
]

# ---------------------------------------------------------------- виконання інструментів


def execute_tool(session: ExamSession, name: str, args: dict) -> dict:
    """Викликає python-функцію інструмента і повертає результат для LLM."""
    if name == "start_exam":
        if session.started:
            return {"error": "Іспит уже розпочато.", "topics": session.topics}
        try:
            topics = tools.start_exam(args.get("email", ""), args.get("name", ""), session.level)
        except tools.StudentNotFoundError as e:
            return {"error": str(e) + " Попроси студента перевірити ім'я та email."}
        session.email = args["email"].strip().lower()
        session.name = args["name"].strip()
        session.topics = topics
        session.remaining = list(topics)
        level = tools.LEVELS[session.level]
        return {"topics": topics, "count": len(topics),
                "level": level["title"],
                "level_instructions": level["style"] + " " + level["grading"],
                "next_step": "Повідом студенту кількість тем і рівень складності, "
                             "потім виклич get_next_topic()."}

    if name == "get_next_topic":
        if not session.started:
            return {"error": "Спочатку виклич start_exam."}
        if session.current_topic and session.current_topic not in session.topic_grades:
            return {"error": f"Спочатку оціни поточну тему '{session.current_topic}' через grade_topic."}
        topic = tools.get_next_topic(session)
        if not topic:
            return {"topic": None,
                    "message": "Тем більше немає. Порахуй середню оцінку, виклич end_exam, "
                               "а потім напиши студенту фінальний відгук."}
        number = session.topics.index(topic) + 1
        level = tools.LEVELS[session.level]
        return {"topic": topic, "number": number, "total": len(session.topics),
                "level": level["title"],
                "level_instructions": level["style"],
                "examiner_hints": tools.topic_key_points(topic, session.level)}

    if name == "grade_topic":
        topic = args.get("topic", "")
        if topic not in session.topics:
            # модель могла трохи змінити назву: беремо поточну тему
            topic = session.current_topic
        if topic is None:
            return {"error": "Немає активної теми для оцінювання."}
        score = max(0.0, min(10.0, float(args.get("score", 0))))
        session.topic_grades[topic] = {"score": score, "comment": args.get("comment", "")}
        return {"status": "ok", "topic": topic, "score": score}

    if name == "end_exam":
        if not session.started:
            return {"error": "Іспит ще не розпочато."}
        if session.finished or session.pending_end:
            return {"error": "Іспит уже завершено."}
        ungraded = [t for t in session.topics if t not in session.topic_grades]
        if ungraded:
            return {"error": "Ще не всі теми оцінено: " + "; ".join(ungraded)}
        score = max(0.0, min(10.0, float(args.get("score", 0))))
        # Сам запис робимо в кінці ходу, щоб у history потрапив і фінальний відгук.
        session.pending_end = {"email": session.email, "score": score}
        return {"status": "ok", "score": score,
                "next_step": "Тепер напиши студенту оцінку та відгук."}

    return {"error": f"Невідомий інструмент {name}"}


def public_view(result: dict) -> dict:
    """Результат інструмента без прихованих підказок (для показу в UI)."""
    return {k: v for k, v in result.items() if k not in ("examiner_hints", "level_instructions")}


# ---------------------------------------------------------------- агент


def new_session(level: str = tools.DEFAULT_LEVEL) -> ExamSession:
    session = ExamSession(level=level)
    session.llm_messages = [
        {"role": "system", "content": build_system_prompt(level)},
        {"role": "assistant", "content": GREETING},
    ]
    session.log("system", GREETING)
    return session


def make_client(api_key: str, provider: str) -> OpenAI:
    base_url = PROVIDERS[provider]["base_url"]
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=2, timeout=90)


def assistant_message_for_history(msg) -> dict:
    """Відповідь моделі у форматі, який приймуть усі провайдери.
    Прибираємо нестандартні поля (напр. `reasoning` у Groq), але лишаємо
    tool_calls повністю: Gemini кладе туди thought signatures, які треба повертати."""
    data = msg.model_dump(exclude_none=True)
    clean = {"role": "assistant"}
    if data.get("content"):
        clean["content"] = data["content"]
    if data.get("tool_calls"):
        clean["tool_calls"] = data["tool_calls"]
    if "content" not in clean and "tool_calls" not in clean:
        clean["content"] = ""
    return clean


def create_completion(client, models: list[str], messages: list[dict], events: list[dict]):
    """Запит до LLM з автоматичним переходом на запасну модель, якщо основна
    перевантажена (503 тощо). Повертає (відповідь, модель, що спрацювала)."""
    last_error = None
    for i, m in enumerate(models):
        try:
            response = client.chat.completions.create(model=m, messages=messages, tools=TOOL_SCHEMAS)
        except openai.APIStatusError as e:
            if e.status_code not in FALLBACK_STATUSES:
                raise
            last_error = e
            continue
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            last_error = e
            continue
        if i > 0:
            events.append({"type": "notice",
                           "content": f"Модель `{models[0]}` зараз недоступна, тому відповідає `{m}`."})
        return response, m
    raise last_error


def run_turn(session: ExamSession, user_text: str, client, model: str | list[str]) -> list[dict]:
    """Один хід діалогу. `model` - назва моделі або список: основна + запасні.
    Повертає список подій для показу в UI:
    {"type": "tool", "name", "args", "result"}, {"type": "text", "content"}
    або {"type": "notice", "content"}."""
    models = [model] if isinstance(model, str) else list(model)
    if session.finished:
        return [{"type": "text", "content": "Іспит уже завершено. Щоб почати новий, натисніть «Новий іспит»."}]

    # До старту іспиту рівень ще можна змінити кнопками в інтерфейсі:
    # перед кожним ходом підставляємо в системний промпт актуальний рівень.
    if not session.started:
        session.llm_messages[0] = {"role": "system", "content": build_system_prompt(session.level)}

    # знімок стану: якщо API впаде посеред ходу, відкотимося
    snapshot = (len(session.llm_messages), len(session.history))

    session.llm_messages.append({"role": "user", "content": user_text})
    session.log("user", user_text)
    events: list[dict] = []

    try:
        for _ in range(MAX_STEPS_PER_TURN):
            response, used = create_completion(client, models, session.llm_messages, events)
            if used != models[0]:
                # до кінця ходу лишаємось на моделі, яка відповіла
                models = [used] + [m for m in models if m != used]
            msg = response.choices[0].message
            session.llm_messages.append(assistant_message_for_history(msg))

            if msg.content and msg.content.strip():
                session.log("system", msg.content)
                events.append({"type": "text", "content": msg.content})

            if not msg.tool_calls:
                break

            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = execute_tool(session, call.function.name, args)
                session.log("tool_call", json.dumps(
                    {"name": call.function.name, "arguments": args, "result": result},
                    ensure_ascii=False))
                events.append({"type": "tool", "name": call.function.name,
                               "args": args, "result": public_view(result)})
                session.llm_messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            events.append({"type": "text", "content": "⚠️ Агент зробив забагато кроків. Спробуйте переформулювати відповідь."})
    except Exception:
        del session.llm_messages[snapshot[0]:]
        del session.history[snapshot[1]:]
        session.pending_end = None
        raise

    # Іспит завершено: показали відгук, тепер записуємо результат з повною історією.
    if session.pending_end:
        if not events or events[-1]["type"] != "text":
            # модель не написала відгук: показуємо підсумок самі
            lines = [f"**Іспит завершено. Рівень: {tools.LEVELS[session.level]['title']}. "
                     f"Підсумкова оцінка: {session.pending_end['score']}/10**", ""]
            for t, g in session.topic_grades.items():
                lines.append(f"- {t}: {g['score']}/10 — {g['comment']}")
            summary = "\n".join(lines)
            session.log("system", summary)
            events.append({"type": "text", "content": summary})
        tools.end_exam(session.pending_end["email"], session.pending_end["score"], session.history,
                       level=session.level)
        session.final_score = session.pending_end["score"]
        session.pending_end = None
        session.finished = True

    return events
