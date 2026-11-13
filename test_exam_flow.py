"""
Тест повного сценарію іспиту без справжньої LLM: фейкова модель відповідає
заздалегідь написаним сценарієм, а ми перевіряємо, що інструменти
викликаються правильно і результат записується.

Запуск:  python -m pytest tests -q     (або просто  python tests/test_exam_flow.py)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai.types.chat import ChatCompletion  # noqa: E402

import agent  # noqa: E402
import tools  # noqa: E402


def completion(content=None, calls=None):
    tool_calls = [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
        for i, (name, args) in enumerate(calls or [])
    ] or None
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}],
    })


class FakeClient:
    """Імітує client.chat.completions.create, віддаючи відповіді по черзі."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        kwargs = {**kwargs, "messages": list(kwargs["messages"])}  # знімок на момент запиту
        self.requests.append(kwargs)
        step = self.script.pop(0)
        return step(kwargs) if callable(step) else step


def last_tool_result(req):
    return json.loads(req["messages"][-1]["content"])


def test_full_exam(tmp_path):
    tools.RESULTS_DIR = tmp_path
    session = agent.new_session()

    # хід 1: неправильний email -> помилка -> бот перепитує
    client = FakeClient([
        completion(calls=[("start_exam", {"email": "nobody@x.com", "name": "Ніхто"})]),
        completion("Не знайшов вас у базі, перевірте email."),
    ])
    agent.run_turn(session, "Ніхто, nobody@x.com", client, "fake")
    assert "error" in last_tool_result(client.requests[1])
    assert not session.started

    # хід 2: правильні дані -> старт + перша тема
    client = FakeClient([
        completion(calls=[("start_exam", {"email": "Test@Example.com", "name": "Student Test"})]),
        completion(calls=[("get_next_topic", {})]),
        lambda req: completion(f"Перша тема: {last_tool_result(req)['topic']}. Що це?"),
    ])
    agent.run_turn(session, "Test Student, test@example.com", client, "fake")
    assert session.started and session.email == "test@example.com"
    assert 2 <= len(session.topics) <= 3
    assert "examiner_hints" in last_tool_result(client.requests[2])

    # не можна взяти нову тему, не оцінивши поточну
    assert "error" in agent.execute_tool(session, "get_next_topic", {})
    # не можна завершити, поки не все оцінено
    assert "error" in agent.execute_tool(session, "end_exam", {"email": session.email, "score": 9})

    # відповідаємо на кожну тему: оцінка + наступна тема
    scores = [8, 6, 7][: len(session.topics)]
    for s in scores:
        client = FakeClient([
            completion(calls=[("grade_topic", {"topic": session.current_topic, "score": s, "comment": "ok"}),
                              ("get_next_topic", {})]),
            completion("Добре, далі."),
        ])
        agent.run_turn(session, "моя відповідь", client, "fake")

    assert session.remaining == [] and len(session.topic_grades) == len(session.topics)

    # фінал: end_exam + відгук
    avg = round(sum(scores) / len(scores), 1)
    client = FakeClient([
        completion(calls=[("end_exam", {"email": session.email, "score": avg})]),
        completion(f"Ваша оцінка {avg}/10. Добре: ... Попрацюйте над: ..."),
    ])
    agent.run_turn(session, "дякую", client, "fake")
    assert session.finished and session.final_score == avg

    # результат записано, history у форматі завдання і містить фінальний відгук
    results = tools.read_results()
    assert results[-1]["email"] == "test@example.com" and results[-1]["score"] == avg
    protocol = json.loads((tmp_path / results[-1]["protocol"]).read_text(encoding="utf-8"))
    roles = {m["role"] for m in protocol["history"]}
    assert roles <= {"system", "user", "tool_call"} and "tool_call" in roles
    assert all(set(m) == {"role", "content", "datetime"} for m in protocol["history"])
    assert f"{avg}/10" in protocol["history"][-1]["content"]

    # після завершення LLM більше не викликається
    events = agent.run_turn(session, "ще питання?", FakeClient([]), "fake")
    assert "завершено" in events[0]["content"]


def test_rollback_on_api_error(tmp_path):
    tools.RESULTS_DIR = tmp_path
    session = agent.new_session()

    def boom(_):
        raise RuntimeError("API down")

    client = FakeClient([completion(calls=[("get_next_topic", {})]), boom])
    n_msgs, n_hist = len(session.llm_messages), len(session.history)
    try:
        agent.run_turn(session, "привіт", client, "fake")
    except RuntimeError:
        pass
    assert len(session.llm_messages) == n_msgs and len(session.history) == n_hist


def test_providers_and_message_cleanup():
    # усі три провайдери налаштовані
    assert {"Google Gemini (безплатно)", "Groq (безплатно)", "xAI Grok"} <= set(agent.PROVIDERS)
    for name in agent.PROVIDERS:
        client = agent.make_client("test-key", name)
        assert str(client.base_url).startswith(agent.PROVIDERS[name]["base_url"].rstrip("/"))

    # поле reasoning (Groq gpt-oss) прибирається, а службові поля tool_calls (Gemini) лишаються
    msg = completion(calls=[("get_next_topic", {})]).choices[0].message
    msg.reasoning = "думки моделі"
    msg.tool_calls[0].extra_content = {"google": {"thought_signature": "abc"}}
    clean = agent.assistant_message_for_history(msg)
    assert "reasoning" not in clean
    assert clean["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "abc"


def api_error(status):
    import httpx
    import openai
    response = httpx.Response(status, request=httpx.Request("POST", "http://test"))
    cls = openai.InternalServerError if status >= 500 else openai.BadRequestError
    return cls(f"error {status}", response=response, body=None)


class RoutingClient(FakeClient):
    """Модель 'busy' завжди відповідає 503, інші - за сценарієм."""

    def create(self, **kwargs):
        if kwargs["model"] == "busy":
            self.requests.append({"model": "busy"})
            raise api_error(503)
        return super().create(**kwargs)


def test_fallback_on_503(tmp_path):
    tools.RESULTS_DIR = tmp_path
    session = agent.new_session()
    client = RoutingClient([
        completion(calls=[("start_exam", {"email": "test@example.com", "name": "Test Student"})]),
        completion("Почнемо!"),
    ])
    events = agent.run_turn(session, "Test Student, test@example.com", client, ["busy", "backup"])
    assert session.started
    assert events[0]["type"] == "notice" and "backup" in events[0]["content"]
    # після переходу ходу більше не стукаємо в перевантажену модель
    assert [r["model"] for r in client.requests] == ["busy", "backup", "backup"]

    # якщо всі моделі перевантажені - помилка, а стан відкочено
    n = len(session.llm_messages)
    try:
        agent.run_turn(session, "відповідь", RoutingClient([]), ["busy"])
        assert False, "мала бути помилка"
    except Exception as e:
        assert getattr(e, "status_code", None) == 503
    assert len(session.llm_messages) == n

    # помилки на кшталт 400 не маскуються переходом на іншу модель
    class BadClient(FakeClient):
        def create(self, **kwargs):
            raise api_error(400)
    try:
        agent.run_turn(session, "відповідь", BadClient([]), ["a", "b"])
        assert False
    except Exception as e:
        assert e.status_code == 400


def run_exam(level, tmp_path):
    """Проходить іспит на заданому рівні; повертає (сесію, підказки, які бачила модель)."""
    tools.RESULTS_DIR = tmp_path
    session = agent.new_session(level=level)
    client = FakeClient([
        completion(calls=[("start_exam", {"email": "test@example.com", "name": "Test Student"})]),
        completion(calls=[("get_next_topic", {})]),
        completion("Перше питання"),
    ])
    agent.run_turn(session, "Test Student, test@example.com", client, "fake")
    start_result = last_tool_result(client.requests[1])
    topic_result = last_tool_result(client.requests[2])
    for _ in session.topics:
        agent.run_turn(session, "відповідь", FakeClient([
            completion(calls=[("grade_topic", {"topic": session.current_topic, "score": 8, "comment": "ok"}),
                              ("get_next_topic", {})]),
            completion("далі"),
        ]), "fake")
    agent.run_turn(session, "все", FakeClient([
        completion(calls=[("end_exam", {"email": "test@example.com", "score": 8})]),
        completion("Оцінка 8/10"),
    ]), "fake")
    return session, start_result, topic_result


def test_levels(tmp_path):
    seen_hints = {}
    for level, info in tools.LEVELS.items():
        session, start_result, topic_result = run_exam(level, tmp_path)
        # рівень передається моделі одразу при старті і нагадується з кожною темою
        assert start_result["level"] == info["title"]
        assert info["style"] in start_result["level_instructions"]
        assert topic_result["level"] == info["title"]
        # підказки саме для цього рівня
        assert topic_result["examiner_hints"] == tools.topic_key_points(topic_result["topic"], level)
        seen_hints[level] = topic_result["examiner_hints"]
        # рівень записано в результати
        assert session.finished and tools.read_results()[-1]["level"] == level

    # у кожної теми є непорожні орієнтири для всіх трьох рівнів, і вони різні
    for t in tools.load_topics():
        hints = [t["levels"][lvl] for lvl in tools.LEVELS]
        assert all(hints) and len(set(hints)) == 3, t["name"]

    # приховані поля не показуються в інтерфейсі
    view = agent.public_view({"topic": "x", "examiner_hints": "секрет", "level_instructions": "секрет"})
    assert "секрет" not in json.dumps(view, ensure_ascii=False)


def test_level_in_prompt_before_start(tmp_path):
    """Рівень з інтерфейсу видно моделі ще до start_exam, щоб вона не питала про нього студента."""
    tools.RESULTS_DIR = tmp_path
    session = agent.new_session()
    session.level = "advanced"            # студент перемкнув рівень до старту
    client = FakeClient([completion("Напишіть, будь ласка, email.")])
    agent.run_turn(session, "Привіт", client, "fake")
    prompt = client.requests[0]["messages"][0]["content"]
    assert "Обраний рівень" in prompt and tools.LEVELS["advanced"]["title"] in prompt
    assert "Ніколи не питай студента про рівень" in prompt

    session.level = "basic"               # перемкнув ще раз, теж до старту
    client = FakeClient([completion(calls=[("start_exam", {"email": "test@example.com", "name": "Test Student"})]),
                         completion("Почнімо.")])
    agent.run_turn(session, "Test Student, test@example.com", client, "fake")
    assert tools.LEVELS["basic"]["title"] in client.requests[0]["messages"][0]["content"]
    fixed_prompt = session.llm_messages[0]["content"]

    session.level = "advanced"            # після старту зміна не впливає
    agent.run_turn(session, "відповідь", FakeClient([completion("Ок.")]), "fake")
    assert session.llm_messages[0]["content"] == fixed_prompt


if __name__ == "__main__":
    import tempfile as _t
    test_levels(Path(_t.mkdtemp()))
    test_level_in_prompt_before_start(Path(_t.mkdtemp()))
    test_fallback_on_503(Path(_t.mkdtemp()))
    test_providers_and_message_cleanup()
    import tempfile
    test_full_exam(Path(tempfile.mkdtemp()))
    test_rollback_on_api_error(Path(tempfile.mkdtemp()))
    print("OK: усі тести пройшли")
