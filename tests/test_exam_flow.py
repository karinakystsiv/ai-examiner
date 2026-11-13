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


if __name__ == "__main__":
    import tempfile
    test_full_exam(Path(tempfile.mkdtemp()))
    test_rollback_on_api_error(Path(tempfile.mkdtemp()))
    print("OK: усі тести пройшли")
