import json
import re
from io import StringIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------
# Helpers: config
# ---------------------------
DEFAULT_CFG = {
    "id_col": 0,
    "nick_col": 2,
    # These are unknown in your export (no headers). You'll choose in UI:
    "net_total_col": None,
    "net_ring_col": None,
    "net_mtt_col": None,
    "commission_col": None,

    # thresholds (tune later)
    "t_approve": 25,
    "t_fast_check": 55,

    "min_sessions_for_coplay": 5,
    "coplay_top1_share_susp": 0.55,
    "coplay_top2_share_susp": 0.75,
}

def load_cfg():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return {**DEFAULT_CFG, **json.load(f)}
    except Exception:
        return DEFAULT_CFG.copy()

def save_cfg(cfg):
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---------------------------
# Parsing: summary table
# ---------------------------
def read_summary(uploaded_file) -> pd.DataFrame:
    # The export looks like ';' separated, no header [file:2]
    raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(raw), sep=";", header=None, dtype=str)
    return df

def to_float_series(s: pd.Series) -> pd.Series:
    # Handles decimal comma: "-2,4" -> -2.4
    x = s.fillna("").astype(str).str.replace("\u00a0", "", regex=False).str.strip()
    x = x.str.replace(",", ".", regex=False)
    x = x.replace({"": np.nan, "None": np.nan})
    return pd.to_numeric(x, errors="coerce")


# ---------------------------
# Parsing: games log -> co-play
# ---------------------------
SESSION_ID_RE = re.compile(r"\bID\s*(\d{12,}-\d+)\b")
DIGITS_RE = re.compile(r"\b(\d{6,10})\b")

def build_coplay_from_games(uploaded_file, known_player_ids: set[int]) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - session_id
    - players (list[int])
    """
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

    # Find session boundaries by session_id pattern like "ID 251217011516-834882" [file:1]
    matches = list(SESSION_ID_RE.finditer(text))
    sessions = []
    for i, m in enumerate(matches):
        sid = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        # Extract numeric tokens and keep only those that are valid player IDs from summary
        ids = []
        for tok in DIGITS_RE.findall(block):
            try:
                v = int(tok)
                if v in known_player_ids:
                    ids.append(v)
            except Exception:
                pass

        # Unique players per session
        players = sorted(set(ids))
        if len(players) >= 2:
            sessions.append({"session_id": sid, "players": players})

    return pd.DataFrame(sessions)


def coplay_features(target_id: int, sessions_df: pd.DataFrame) -> dict:
    rows = sessions_df[sessions_df["players"].apply(lambda ps: target_id in ps)]
    sessions_count = int(len(rows))
    if sessions_count == 0:
        return {
            "sessions_count": 0,
            "unique_opponents": 0,
            "top1_coplay_share": 0.0,
            "top2_coplay_share": 0.0,
            "top_partners": [],
        }

    counter = {}
    for ps in rows["players"]:
        for pid in ps:
            if pid == target_id:
                continue
            counter[pid] = counter.get(pid, 0) + 1

    # sort partners by coplay count
    partners = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    unique_opponents = len(partners)

    top1 = partners[0][1] if len(partners) >= 1 else 0
    top2 = partners[1][1] if len(partners) >= 2 else 0
    top1_share = top1 / sessions_count if sessions_count else 0.0
    top2_share = (top1 + top2) / sessions_count if sessions_count else 0.0

    return {
        "sessions_count": sessions_count,
        "unique_opponents": unique_opponents,
        "top1_coplay_share": float(top1_share),
        "top2_coplay_share": float(top2_share),
        "top_partners": partners[:5],
    }


# ---------------------------
# Scoring
# ---------------------------
def safe_get_val(df_row: pd.Series, col_idx: int | None) -> float | None:
    if col_idx is None:
        return None
    return float(to_float_series(pd.Series([df_row.iloc[col_idx]])).iloc[0])

def risk_decision(score: int, cfg: dict) -> str:
    if score < cfg["t_approve"]:
        return "APPROVE"
    if score < cfg["t_fast_check"]:
        return "FAST_CHECK"
    return "MANUAL_REVIEW"

def score_player(row: pd.Series, cop: dict, cfg: dict) -> tuple[int, list[str], dict]:
    reasons = []
    score = 0

    net_total = safe_get_val(row, cfg["net_total_col"])
    net_ring = safe_get_val(row, cfg["net_ring_col"])
    net_mtt = safe_get_val(row, cfg["net_mtt_col"])
    comm = safe_get_val(row, cfg["commission_col"])

    # ---- A) Profit/Loss logic
    if net_total is not None:
        if net_total <= 0:
            reasons.append(f"net_total<=0 ({net_total})")
            score += 0
        else:
            reasons.append(f"net_total>0 ({net_total})")
            score += 30
    else:
        reasons.append("net_total column not set")
        score += 10

    # MTT reduces suspicion if player is in profit mainly via MTT
    if net_total is not None and net_total > 0 and net_mtt is not None:
        mtt_share = min(1.0, max(0.0, net_mtt / net_total)) if net_total else 0.0
        if mtt_share >= 0.7:
            reasons.append(f"High MTT share ({mtt_share:.0%}) -> lower risk")
            score -= 15

    # Ring increases suspicion if profit mainly via ring
    if net_total is not None and net_total > 0 and net_ring is not None:
        ring_share = min(1.0, max(0.0, net_ring / net_total)) if net_total else 0.0
        if ring_share >= 0.7:
            reasons.append(f"High Ring share ({ring_share:.0%}) -> higher risk")
            score += 15

    # Commission while losing
    if net_total is not None and net_total < 0 and comm is not None:
        # heuristic threshold: adjust later
        if comm > abs(net_total) * 0.5 and comm > 1:
            reasons.append(f"High commission with loss (comm={comm})")
            score += 30

    # ---- B) Co-play signals
    if cop["sessions_count"] >= cfg["min_sessions_for_coplay"]:
        if cop["top1_coplay_share"] >= cfg["coplay_top1_share_susp"]:
            reasons.append(f"Top1 co-play share high ({cop['top1_coplay_share']:.0%})")
            score += 25
        if cop["top2_coplay_share"] >= cfg["coplay_top2_share_susp"]:
            reasons.append(f"Top2 co-play share high ({cop['top2_coplay_share']:.0%})")
            score += 15
        if cop["unique_opponents"] <= 5:
            reasons.append(f"Few unique opponents ({cop['unique_opponents']})")
            score += 10
    else:
        reasons.append("Not enough sessions for stable co-play signal")

    score = int(max(0, min(100, score)))
    details = {
        "net_total": net_total,
        "net_ring": net_ring,
        "net_mtt": net_mtt,
        "commission": comm,
        **cop,
    }
    return score, reasons, details


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="PPPoker Risk Checker (MVP)", layout="wide")
st.title("PPPoker: авто-оценка риска вывода (MVP)")

cfg = load_cfg()

with st.sidebar:
    st.header("1) Загрузка файлов")
    summary_file = st.file_uploader("Общая таблица (CSV ';')", type=["csv"])
    games_file = st.file_uploader("Игры/сессии (CSV/текст)", type=["csv", "txt"])

    st.header("2) Настройка колонок")
    st.caption("В выгрузке нет заголовков — выбери индексы колонок (0,1,2...).")
    id_col = st.number_input("ID колонка", min_value=0, value=int(cfg["id_col"]), step=1)
    nick_col = st.number_input("Nick колонка", min_value=0, value=int(cfg["nick_col"]), step=1)

    net_total_col = st.text_input("net_total_col (например 8)", value="" if cfg["net_total_col"] is None else str(cfg["net_total_col"]))
    net_ring_col  = st.text_input("net_ring_col (опц.)", value="" if cfg["net_ring_col"] is None else str(cfg["net_ring_col"]))
    net_mtt_col   = st.text_input("net_mtt_col (опц.)", value="" if cfg["net_mtt_col"] is None else str(cfg["net_mtt_col"]))
    comm_col      = st.text_input("commission_col (опц.)", value="" if cfg["commission_col"] is None else str(cfg["commission_col"]))

    st.header("3) Пороги решений")
    t_approve = st.slider("APPROVE если score <", 0, 100, int(cfg["t_approve"]))
    t_fast = st.slider("FAST_CHECK если score <", 0, 100, int(cfg["t_fast_check"]))

    if st.button("Сохранить настройки"):
        cfg["id_col"] = int(id_col)
        cfg["nick_col"] = int(nick_col)
        cfg["net_total_col"] = int(net_total_col) if net_total_col.strip() != "" else None
        cfg["net_ring_col"] = int(net_ring_col) if net_ring_col.strip() != "" else None
        cfg["net_mtt_col"] = int(net_mtt_col) if net_mtt_col.strip() != "" else None
        cfg["commission_col"] = int(comm_col) if comm_col.strip() != "" else None
        cfg["t_approve"] = int(t_approve)
        cfg["t_fast_check"] = int(t_fast)
        save_cfg(cfg)
        st.success("Сохранено в config.json")

st.divider()

if not summary_file or not games_file:
    st.info("Загрузи оба файла в сайдбаре.")
    st.stop()

df = read_summary(summary_file)
df["_player_id"] = pd.to_numeric(df.iloc[:, cfg["id_col"]], errors="coerce")
df["_nick"] = df.iloc[:, cfg["nick_col"]].astype(str)
df = df.dropna(subset=["_player_id"]).copy()
df["_player_id"] = df["_player_id"].astype(int)

known_ids = set(df["_player_id"].tolist())
sessions_df = build_coplay_from_games(games_file, known_ids)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Проверка вывода")
    target_id = st.number_input("ID игрока", min_value=0, value=int(df["_player_id"].iloc[0]))
    run = st.button("Оценить риск")

with col2:
    st.subheader("Диагностика файлов")
    st.write(f"Игроков в summary: {len(df)}")
    st.write(f"Сессий (из games): {len(sessions_df)}")
    st.caption("Если 'Сессий=0' — значит парсер не поймал блоки или ID игроков не совпали.")

if not run:
    st.stop()

row_df = df[df["_player_id"] == int(target_id)]
if row_df.empty:
    st.error("ID игрока не найден в общей таблице.")
    st.stop()

row = row_df.iloc[0]
cop = coplay_features(int(target_id), sessions_df)
score, reasons, details = score_player(row, cop, cfg)
decision = risk_decision(score, cfg)

st.success(f"Decision: {decision} | Risk score: {score}/100")

st.subheader("Почему так (reasons)")
for r in reasons[:10]:
    st.write("- " + r)

st.subheader("Детали (для СБ)")
st.json(details)

# show top partners
if details.get("top_partners"):
    st.subheader("Топ партнёры по совместным сессиям")
    p = pd.DataFrame(details["top_partners"], columns=["partner_id", "coplay_sessions"])
    st.dataframe(p, use_container_width=True)
