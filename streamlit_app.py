"""
AI Examiner — вебінтерфейс на Streamlit.
Локальний запуск:  streamlit run streamlit_app.py
"""

import json
import os

import openai
import streamlit as st

import tools
from agent import PROVIDERS, make_client, new_session, run_turn

st.set_page_config(page_title="AI Examiner · NLP", page_icon="🎓", layout="centered")


def secret_key(name: str) -> str:
    """Ключ із Secrets (Streamlit Cloud) або змінної середовища, якщо є."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, "")


if "session" not in st.session_state:
    st.session_state.session = new_session()
    st.session_state.display = [{"role": "assistant", "events": [
        {"type": "text", "content": st.session_state.session.history[0]["content"]}]}]

session = st.session_state.session

# ---------------------------------------------------------------- бічна панель

with st.sidebar:
    st.header("⚙️ Налаштування")
    provider = st.radio("LLM-провайдер", list(PROVIDERS),
                        help="Провайдера можна змінити навіть посеред іспиту")
    model = st.selectbox("Модель", PROVIDERS[provider]["models"], accept_new_options=True,
                         help="Можна вписати назву іншої моделі")
    api_key = st.text_input(f"API-ключ ({provider.split(' (')[0]})", type="password",
                            key=f"key_{provider}",
                            help="Ключ не зберігається, він живе лише у вашій сесії браузера")
    st.caption(f"Отримати ключ: {PROVIDERS[provider]['key_url']}")
    if not api_key:
        api_key = secret_key(PROVIDERS[provider]["secret"])
    show_tools = st.toggle("Показувати виклики інструментів", value=True,
                           help="Режим демонстрації: видно, які функції викликає агент")

    if st.button("🔄 Новий іспит", width="stretch"):
        del st.session_state["session"]
        st.rerun()

    with st.expander("👩‍🎓 Тестові студенти"):
        st.caption("Іспит можуть складати лише студенти з бази:")
        for s in tools.load_students():
            st.markdown(f"- {s['name']} — `{s['email']}`")

    with st.expander("📊 Журнал результатів"):
        results = tools.read_results()
        if results:
            st.dataframe([{"email": r["email"], "score": r["score"],
                           "рівень": tools.LEVELS.get(r.get("level"), {}).get("title", "—"),
                           "datetime": r["datetime"]} for r in results],
                         hide_index=True)
        else:
            st.caption("Поки що немає завершених іспитів.")

# ---------------------------------------------------------------- головна частина

st.title("🎓 AI Examiner: іспит з NLP")
st.markdown(
    "LLM-агент, який приймає усний іспит з курсу обробки природної мови. "
    "Він перевіряє студента в базі, видає 2–3 випадкові теми, ставить запитання "
    "в діалозі на обраному рівні складності, оцінює кожну тему й наприкінці показує "
    "оцінку (0–10) та відгук. "
    "Під капотом агент викликає інструменти `start_exam`, `get_next_topic`, "
    "`grade_topic` та `end_exam`."
)
# ---- вибір рівня: доступний лише до старту іспиту, далі заблокований
level = st.segmented_control(
    "Рівень складності",
    options=list(tools.LEVELS),
    format_func=lambda k: f"{tools.LEVELS[k]['emoji']} {tools.LEVELS[k]['title']}",
    default=session.level,
    disabled=session.started,
    key="level_choice",
)
if not session.started:
    session.level = level or session.level   # None, якщо користувач зняв вибір
    st.caption(tools.LEVELS[session.level]["description"])
else:
    st.caption(f"🔒 Рівень зафіксовано на весь іспит: "
               f"**{tools.LEVELS[session.level]['emoji']} {tools.LEVELS[session.level]['title']}**")

if not api_key:
    st.info("👈 Оберіть провайдера і введіть API-ключ у бічній панелі. "
            "Безплатні ключі дають Google Gemini (AI Studio) і Groq.")


def render_events(events: list[dict]) -> None:
    for ev in events:
        if ev["type"] == "text":
            st.markdown(ev["content"])
        elif ev["type"] == "notice":
            st.caption("⚠️ " + ev["content"])
        elif show_tools:
            with st.expander(f"🔧 {ev['name']}({', '.join(f'{k}={v!r}' for k, v in ev['args'].items())})"):
                st.code(json.dumps(ev["result"], ensure_ascii=False, indent=2), language="json")


for item in st.session_state.display:
    with st.chat_message(item["role"]):
        if item["role"] == "user":
            st.markdown(item["content"])
        else:
            render_events(item["events"])

if session.finished:
    st.success(f"Іспит завершено. Оцінка: **{session.final_score}/10**. Результат збережено.")

was_started = session.started
prompt = st.chat_input("Ваша відповідь…", disabled=session.finished or not api_key)
if prompt:
    st.session_state.display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Екзаменатор думає…"):
                # обрана модель + інші моделі цього провайдера як запасні
                models = [model] + [m for m in PROVIDERS[provider]["models"] if m != model]
                events = run_turn(session, prompt, make_client(api_key, provider), models)
        except openai.AuthenticationError:
            st.session_state.display.pop()
            st.error("Невірний API-ключ. Перевірте його в бічній панелі.")
            st.stop()
        except openai.RateLimitError:
            st.session_state.display.pop()
            st.error("Перевищено ліміт запитів безплатного тарифу. Зачекайте хвилину й надішліть відповідь ще раз.")
            st.stop()
        except openai.NotFoundError:
            st.session_state.display.pop()
            st.error(f"Модель `{model}` не знайдено. Оберіть іншу модель у бічній панелі.")
            st.stop()
        except openai.APIStatusError as e:
            st.session_state.display.pop()
            if e.status_code >= 500:
                st.error(f"Сервери {provider.split(' (')[0]} зараз перевантажені (код {e.status_code}), "
                         "і запасні моделі теж не відповіли. Зачекайте 1–2 хвилини й надішліть відповідь ще раз "
                         "або оберіть іншого провайдера в бічній панелі. Діалог не загубиться.")
            else:
                st.error(f"Помилка LLM API (код {e.status_code}): {e.message}")
            st.stop()
        except openai.APIError as e:
            st.session_state.display.pop()
            st.error(f"Помилка LLM API: {e}. Спробуйте надіслати відповідь ще раз.")
            st.stop()
        render_events(events)
    st.session_state.display.append({"role": "assistant", "events": events})
    if session.finished or session.started != was_started:
        st.rerun()   # оновити плашку рівня (заблокувати вибір) і статус іспиту
