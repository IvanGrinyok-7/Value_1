import io
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# CONFIG
# =========================
APP_TITLE = "PPPoker: риск-оценка по DB.xlsx (все недели)"
CACHE_DIR = Path(".pppoker_db_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Индексы колонок (0-based) по структуре из “шапки”, которую ты дал:
# A=0 Номер недели, B=1 ID игрока, J=9 Общий выигрыш игроков + События
IDX_WEEK = 0
IDX_PLAYER_ID = 1
IDX_TOTAL_WIN_J = 9

# Доп. важные колонки из этого же файла (по порядку из “шапки”)
IDX_PLAYER_WIN_TOTAL = 14   # "Выигрыш игрока Общий"
IDX_PLAYER_WIN_RING = 15    # "Выигрыш игрока Ring Game"
IDX_PLAYER_WIN_MTT = 16     # "Выигрыш игрока MTT, SNG"
IDX_CLUB_INCOME_TOTAL = 27  # "Доход клуба Общий"
IDX_CLUB_COMMISSION = 28    # "Доход клуба Комиссия"

# Пороговые решения
T_APPROVE = 25
T_FAST_CHECK = 55


# =========================
# PERSISTENT FILE CACHE
# =========================
class BytesFile:
    def __init__(self, content: bytes, name: str):
        self._content = content
        self.name = name

    def getvalue(self) -> bytes:
        return self._content


def _bin_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.bin"


def _meta_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def cache_save_uploaded(key: str, uploaded_file) -> None:
    if uploaded_file is None:
        return
    content = uploaded_file.getvalue()
    name = getattr(uploaded_file, "name", key)

    _bin_path(key).write_bytes(content)
    _meta_path(key).write_text(
        json.dumps(
            {
                "name": name,
                "bytes": len(content),
                "saved_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def cache_load_file(key: str):
    bp = _bin_path(key)
    mp = _meta_path(key)
    if not bp.exists():
        return None

    content = bp.read_bytes()
    name = key
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
            name = meta.get("name", key)
        except Exception:
            name = key
    return BytesFile(content, name)


def cache_meta(key: str) -> dict | None:
    mp = _meta_path(key)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return None


def cache_clear(key: str) -> None:
    bp = _bin_path(key)
    mp = _meta_path(key)
    if bp.exists():
        bp.unlink()
    if mp.exists():
        mp.unlink()


def resolve_file(key: str, uploaded_file):
    if uploaded_file is not None:
        cache_save_uploaded(key, uploaded_file)
        return uploaded_file
    return cache_load_file(key)


# =========================
# HELPERS
# =========================
def to_float_series(s: pd.Series) -> pd.Series:
    x = s.copy()
    x = x.fillna("").astype(str).str.replace("\u00a0", "", regex=False).str.strip()
    x = x.str.replace(",", ".", regex=False)
    x = x.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    return pd.to_numeric(x, errors="coerce")


def fmt_money(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NaN"
    return f"{float(v):.2f}"


def risk_decision(score: int) -> str:
    if score < T_APPROVE:
        return "APPROVE"
    if score < T_FAST_CHECK:
        return "FAST_CHECK"
    return "MANUAL_REVIEW"


def safe_div(a, b):
    if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return a / b


def detect_header_row(df0: pd.DataFrame) -> bool:
    """
    Пользователь может загружать файл без шапки.
    Если в первой строке видим 'Номер недели' или 'ID игрока' — считаем, что это header.
    """
    row0 = df0.iloc[0].astype(str).tolist()
    row0 = [x.strip().lower() for x in row0]
    joined = " ".join(row0)
    return ("номер недели" in joined) or ("id игрока" in joined)


# =========================
# LOAD + NORMALIZE DB
# =========================
def load_db_xlsx(file_obj) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(file_obj.getvalue()), sheet_name="Общий", header=None)
    # fallback: если лист не найден, читаем первый
    if raw is None or raw.empty:
        raw = pd.read_excel(io.BytesIO(file_obj.getvalue()), sheet_name=0, header=None)

    has_header = detect_header_row(raw)
    if has_header:
        data = raw.iloc[1:].copy()
    else:
        data = raw.copy()

    # оставляем разумное число колонок (по шапке их 48)
    data = data.iloc[:, :48].copy()

    # базовые ключи
    data["_week"] = pd.to_numeric(data.iloc[:, IDX_WEEK], errors="coerce")
    data["_player_id"] = pd.to_numeric(data.iloc[:, IDX_PLAYER_ID], errors="coerce")

    # выкидываем строки без player_id (и строку "Итог" тоже отвалится)
    data = data.dropna(subset=["_player_id"]).copy()
    data["_player_id"] = data["_player_id"].astype(int)

    # числовые метрики
    data["_total_win_j"] = to_float_series(data.iloc[:, IDX_TOTAL_WIN_J])  # J: общий выигрыш + события
    data["_player_win_total"] = to_float_series(data.iloc[:, IDX_PLAYER_WIN_TOTAL])
    data["_player_win_ring"] = to_float_series(data.iloc[:, IDX_PLAYER_WIN_RING])
    data["_player_win_mtt"] = to_float_series(data.iloc[:, IDX_PLAYER_WIN_MTT])
    data["_club_income_total"] = to_float_series(data.iloc[:, IDX_CLUB_INCOME_TOTAL])
    data["_club_commission"] = to_float_series(data.iloc[:, IDX_CLUB_COMMISSION])

    # если week пустой — ставим -1, чтобы не ломать группировки
    data["_week"] = data["_week"].fillna(-1).astype(int)

    return data[
        [
            "_week",
            "_player_id",
            "_total_win_j",
            "_player_win_total",
            "_player_win_ring",
            "_player_win_mtt",
            "_club_income_total",
            "_club_commission",
        ]
    ].copy()


# =========================
# ANALYTICS (АГРЕГАТНАЯ, БЕЗ ПАР/СТОЛОВ)
# =========================
def build_player_views(df: pd.DataFrame, player_id: int, weeks_mode: str, last_n: int, week_from: int, week_to: int):
    d = df[df["_player_id"] == int(player_id)].copy()
    if d.empty:
        return None, None

    # фильтр по неделям
    weeks = sorted([w for w in d["_week"].unique().tolist() if w >= 0])
    if weeks_mode == "Все недели":
        pass
    elif weeks_mode == "Последние N недель":
        if weeks:
            max_w = max(weeks)
            min_w = max_w - max(0, int(last_n) - 1)
            d = d[(d["_week"] >= min_w) & (d["_week"] <= max_w)].copy()
    else:  # "Диапазон недель"
        d = d[(d["_week"] >= int(week_from)) & (d["_week"] <= int(week_to))].copy()

    # группировка по неделе (на случай дублей)
    by_week = (
        d.groupby("_week", as_index=False)[
            [
                "_total_win_j",
                "_player_win_total",
                "_player_win_ring",
                "_player_win_mtt",
                "_club_income_total",
                "_club_commission",
            ]
        ]
        .sum(min_count=1)
        .sort_values("_week")
    )

    # агрегат по периоду
    agg = by_week[[
        "_total_win_j",
        "_player_win_total",
        "_player_win_ring",
        "_player_win_mtt",
        "_club_income_total",
        "_club_commission",
    ]].sum(numeric_only=True)

    total_win = float(agg.get("_total_win_j", 0.0) or 0.0)
    ring_win = float(agg.get("_player_win_ring", 0.0) or 0.0)
    mtt_win = float(agg.get("_player_win_mtt", 0.0) or 0.0)
    comm = float(agg.get("_club_commission", 0.0) or 0.0)
    pure_player_win = float(agg.get("_player_win_total", 0.0) or 0.0)

    # derived
    ring_share = safe_div(ring_win, total_win) if total_win > 0 else np.nan
    mtt_share = safe_div(mtt_win, total_win) if total_win > 0 else np.nan
    profit_to_rake = safe_div(total_win, comm) if comm > 0 else np.nan
    events_delta = total_win - pure_player_win  # насколько J "больше/меньше" чистого выигрыша

    # концентрация по неделям
    if by_week.empty:
        top_week = None
        top_week_win = 0.0
        top_week_share = np.nan
    else:
        top_row = by_week.sort_values("_total_win_j", ascending=False).iloc[0]
        top_week = int(top_row["_week"])
        top_week_win = float(top_row["_total_win_j"] or 0.0)
        top_week_share = safe_div(top_week_win, total_win) if total_win > 0 else np.nan

    summary = {
        "weeks_count": int(len(by_week)),
        "week_min": int(by_week["_week"].min()) if len(by_week) else None,
        "week_max": int(by_week["_week"].max()) if len(by_week) else None,
        "total_win": total_win,
        "pure_player_win": pure_player_win,
        "events_delta": float(events_delta),
        "ring_win": ring_win,
        "mtt_win": mtt_win,
        "commission": comm,
        "ring_share": ring_share,
        "mtt_share": mtt_share,
        "profit_to_rake": profit_to_rake,
        "top_week": top_week,
        "top_week_win": top_week_win,
        "top_week_share": top_week_share,
    }
    return summary, by_week


def score_player(summary: dict) -> tuple[int, str, str, list[str]]:
    """
    Это риск-скоринг 'под вывод' по агрегатным данным.
    Он не доказывает перелив, а выделяет кейсы для проверки.
    """
    reasons = []
    score = 0

    total_win = summary["total_win"]
    ring_win = summary["ring_win"]
    comm = summary["commission"]
    ring_share = summary["ring_share"]
    top_week_share = summary["top_week_share"]
    weeks_count = summary["weeks_count"]
    profit_to_rake = summary["profit_to_rake"]
    events_delta = summary["events_delta"]

    # 1) База: если игрок не в плюсе — как "получатель" менее вероятен
    if total_win <= 0:
        score = 5
        reasons.append("Игрок в минусе за выбранный период (по J), риск 'получателя под вывод' ниже.")
    else:
        score = 20
        reasons.append("Игрок в плюсе за выбранный период (по J) — требуется базовая проверка источника профита.")

    # 2) Ring как основной источник профита
    if total_win > 0 and pd.notna(ring_share):
        if ring_share >= 0.70 and abs(ring_win) >= 50:
            score += 20
            reasons.append("Высокая доля профита из Ring (кэш) — для переливов это более рискованный формат.")
        elif ring_share >= 0.50 and abs(ring_win) >= 50:
            score += 10
            reasons.append("Заметная доля профита из Ring (кэш).")

    # 3) Соотношение профита к комиссии (очень грубый прокси)
    if total_win > 0:
        if comm <= 0:
            score += 10
            reasons.append("Комиссия (рейк) за период не видна/нулевая — оценка менее надёжна, нужен ручной взгляд.")
        else:
            if pd.notna(profit_to_rake) and profit_to_rake >= 8 and total_win >= 100:
                score += 15
                reasons.append("Высокое отношение профита к комиссии — профит выглядит 'дешёвым' относительно активности.")
            elif pd.notna(profit_to_rake) and profit_to_rake >= 5 and total_win >= 100:
                score += 8
                reasons.append("Профит заметно выше комиссии — стоит проверить стабильность и источник результата.")

    # 4) Концентрация профита в одной неделе (типовой паттерн для вывода)
    if total_win > 0 and weeks_count >= 2 and pd.notna(top_week_share):
        if top_week_share >= 0.60:
            score += 15
            reasons.append("Сильная концентрация профита в одной неделе — похоже на 'разовый занос под вывод'.")

    # 5) “События/бонусы” (J включает события)
    # Это НЕ признак перелива, но важный контекст для трактовки.
    if total_win != 0:
        ev_share = safe_div(abs(events_delta), abs(total_win))
        if pd.notna(ev_share) and ev_share >= 0.50 and abs(total_win) >= 50:
            score -= 5
            reasons.append("Существенная доля результата может быть из событий/бонусов (J включает события) — это снижает подозрительность перелива, но требует уточнения.")

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)

    if decision == "APPROVE":
        main_risk = "Явных агрегатных признаков 'получателя под вывод' не обнаружено."
    elif decision == "FAST_CHECK":
        main_risk = "Есть агрегатные признаки риска — рекомендуется быстрая проверка перед выводом."
    else:
        main_risk = "Высокий агрегатный риск — требуется ручная проверка СБ."

    return score, decision, main_risk, reasons


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Загрузка DB.xlsx")
    st.caption("Файл сохраняется на сервере и не пропадает при обновлении страницы, пока его не удалить вручную.")

    uploaded = st.file_uploader("DB.xlsx (все недели, без шапки или с шапкой)", type=["xlsx"], key="db_upload")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Очистить файл", use_container_width=True):
            cache_clear("db")
            st.rerun()
    with col2:
        if st.button("Очистить всё", use_container_width=True):
            cache_clear("db")
            st.rerun()

    meta = cache_meta("db")
    st.divider()
    st.subheader("Сохранённый файл")
    if meta:
        st.write(f"- Имя: {meta.get('name','db')}")
        st.write(f"- Размер: {meta.get('bytes',0)} bytes")
        st.write(f"- Сохранён: {meta.get('saved_at','')}")
    else:
        st.write("- Нет сохранённого файла.")

db_file = resolve_file("db", uploaded)
if db_file is None:
    st.info("Загрузи DB.xlsx или восстанови его из сохранённого состояния (если уже загружал ранее).")
    st.stop()

# Load DB
try:
    df = load_db_xlsx(db_file)
except Exception as e:
    st.error("Не удалось прочитать DB.xlsx. Проверь формат файла/лист 'Общий'.")
    st.exception(e)
    st.stop()

# Top bar
c1, c2, c3 = st.columns(3, gap="small")
c1.metric("Строк (после очистки)", f"{len(df)}", border=True)
c2.metric("Уникальных игроков", f"{df['_player_id'].nunique()}", border=True)
valid_weeks = sorted([w for w in df["_week"].unique().tolist() if w >= 0])
c3.metric("Недель в файле", f"{len(valid_weeks)}", border=True)

st.divider()

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Проверка игрока")
    default_id = int(df["_player_id"].iloc[0]) if len(df) else 0
    player_id = st.number_input("ID игрока", min_value=0, value=default_id, step=1)

    st.subheader("Период")
    weeks_mode = st.selectbox("Фильтр недель", ["Все недели", "Последние N недель", "Диапазон недель"])
    last_n = st.number_input("N (если выбран режим 'Последние N недель')", min_value=1, value=4, step=1)

    if valid_weeks:
        w_min, w_max = min(valid_weeks), max(valid_weeks)
    else:
        w_min, w_max = 0, 0

    week_from = st.number_input("Неделя от", value=w_min, step=1)
    week_to = st.number_input("Неделя до", value=w_max, step=1)

    run = st.button("Проверить", type="primary", use_container_width=True)

with right:
    st.subheader("Что умеет эта версия")
    st.markdown(
        "- Оценивает риск 'получателя под вывод' по **агрегатной** статистике (без раздач/столов).\n"
        "- Показывает разложение выигрыша (J) по Ring/MTT и связь с комиссией.\n"
        "- Показывает концентрацию результата по неделям.\n"
        "- Не строит пары 'донор→получатель' (для этого нужны данные уровня игр/раздач)."
    )

if not run:
    st.stop()

summary, by_week = build_player_views(df, int(player_id), weeks_mode, int(last_n), int(week_from), int(week_to))
if summary is None:
    st.error("Игрок не найден в DB.xlsx по выбранному периоду.")
    st.stop()

score, decision, main_risk, reasons = score_player(summary)

st.divider()

h1, h2, h3 = st.columns([1.2, 1, 1], gap="small")
with h1:
    st.subheader("Итог")
h2.metric("Risk score", f"{score}/100", border=True)
h3.metric("Decision", decision, border=True)

if decision == "APPROVE":
    st.success("ВЫВОД: РАЗРЕШИТЬ")
elif decision == "FAST_CHECK":
    st.warning("ВЫВОД: БЫСТРАЯ ПРОВЕРКА")
else:
    st.error("ВЫВОД: РУЧНАЯ ПРОВЕРКА СБ")

tabs = st.tabs(["Кратко", "Детали", "Недели"])

with tabs[0]:
    st.subheader("Главный риск")
    st.info(main_risk)

    st.subheader("Причины (как приложение думало)")
    for r in reasons[:10]:
        st.markdown(f"- {r}")

with tabs[1]:
    st.subheader("Агрегаты по периоду")
    a, b, c, d = st.columns(4, gap="small")
    a.metric("J: итог (+события)", fmt_money(summary["total_win"]), border=True)
    b.metric("O: выигрыш игрока общий", fmt_money(summary["pure_player_win"]), border=True)
    c.metric("P: Ring win", fmt_money(summary["ring_win"]), border=True)
    d.metric("Q: MTT win", fmt_money(summary["mtt_win"]), border=True)

    e, f, g, h = st.columns(4, gap="small")
    e.metric("Комиссия клуба", fmt_money(summary["commission"]), border=True)
    f.metric("Ring доля (P/J)", "NaN" if pd.isna(summary["ring_share"]) else f"{summary['ring_share']:.0%}", border=True)
    g.metric("MTT доля (Q/J)", "NaN" if pd.isna(summary["mtt_share"]) else f"{summary['mtt_share']:.0%}", border=True)
    h.metric("Профит/комиссия", "NaN" if pd.isna(summary["profit_to_rake"]) else f"{summary['profit_to_rake']:.1f}x", border=True)

    st.subheader("Контекст по событиям")
    st.markdown(f"- Разница (J - O): {fmt_money(summary['events_delta'])} (это может быть лидерборд/бонусы, т.к. J включает события).")

    st.subheader("Концентрация")
    st.markdown(
        f"- Недель в периоде: {summary['weeks_count']}.\n"
        f"- Мин/макс неделя: {summary['week_min']}–{summary['week_max']}.\n"
        f"- Топ-неделя по J: {summary['top_week']} (J={fmt_money(summary['top_week_win'])}).\n"
        f"- Доля топ-недели в профите: "
        + ("NaN" if pd.isna(summary["top_week_share"]) else f"{summary['top_week_share']:.0%}")
        + "."
    )

with tabs[2]:
    st.subheader("Таблица по неделям")
    out = by_week.rename(
        columns={
            "_week": "Неделя",
            "_total_win_j": "J: итог (+события)",
            "_player_win_total": "O: выигрыш игрока общий",
            "_player_win_ring": "P: Ring win",
            "_player_win_mtt": "Q: MTT win",
            "_club_income_total": "Доход клуба общий",
            "_club_commission": "Комиссия клуба",
        }
    ).copy()

    out["J-O (события/дельта)"] = out["J: итог (+события)"] - out["O: выигрыш игрока общий"]
    out = out.sort_values("Неделя", ascending=False)
    st.dataframe(out, use_container_width=True)
