import re
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------
# Column mapping (Excel letters -> 0-based index)
# ---------------------------
def col_idx(col_letter: str) -> int:
    col_letter = col_letter.strip().upper()
    n = 0
    for ch in col_letter:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


# Excel columns from your "общая таблица"
IDX_ID = col_idx("A")         # A - ID
IDX_NET_TOTAL = col_idx("I")  # I - общий выигрыш
IDX_NET_RING = col_idx("O")   # O - Ring Game
IDX_NET_MTT = col_idx("P")    # P - MTT/SNG
IDX_COMM = col_idx("AB")      # AB - комиссия


# thresholds (tune later)
T_APPROVE = 25
T_FAST_CHECK = 55

MIN_SESSIONS_FOR_COPLAY = 5
COPLAY_TOP1_SHARE_SUSP = 0.55
COPLAY_TOP2_SHARE_SUSP = 0.75


# ---------------------------
# Summary table (Excel)
# ---------------------------
def read_summary_excel(uploaded_xlsx) -> pd.DataFrame:
    # Ты удаляешь шапку -> читаем как "сырой" грид
    return pd.read_excel(uploaded_xlsx, engine="openpyxl", header=None)


def to_float_series(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.replace("\u00a0", "", regex=False).str.strip()
    x = x.str.replace(",", ".", regex=False)
    x = x.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    return pd.to_numeric(x, errors="coerce")


# ---------------------------
# Games CSV -> co-play
# ---------------------------
# Пример из твоего файла:
# "2025/12/17\nUTC +0500";ID игры: 251216053643-795459   Название стола: ...
# ;ID игрока;Ник;...
# ;11637827;Masalay;...
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\-]+)", re.IGNORECASE)
# строка игрока начинается с ';<digits>;' или '"...";;<digits>;'
PLAYER_ID_IN_ROW_RE = re.compile(r"(?:^|;)\s*(\d{6,10})\s*;", re.IGNORECASE)


def build_coplay_from_games(uploaded_file, known_player_ids: set[int]) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = str(raw)

    lines = text.splitlines()

    sessions = []
    current_game_id = None
    current_players = set()

    def flush():
        nonlocal current_game_id, current_players, sessions
        if current_game_id and len(current_players) >= 2:
            sessions.append(
                {"session_id": current_game_id, "players": sorted(current_players)}
            )
        current_game_id = None
        current_players = set()

    for line in lines:
        # 1) новый блок игры
        m_game = GAME_ID_RE.search(line)
        if m_game:
            flush()
            current_game_id = m_game.group(1).strip()
            continue

        # 2) строки игроков внутри блока
        if current_game_id:
            m_pid = PLAYER_ID_IN_ROW_RE.search(line)
            if m_pid:
                pid = int(m_pid.group(1))
                # фильтруем по тем ID, которые реально есть в Excel
                if pid in known_player_ids:
                    current_players.add(pid)

    flush()

    out = pd.DataFrame(sessions)
    if out.empty:
        out = pd.DataFrame(columns=["session_id", "players"])
    return out


def coplay_features(target_id: int, sessions_df: pd.DataFrame) -> dict:
    if sessions_df.empty or "players" not in sessions_df.columns:
        return {
            "sessions_count": 0,
            "unique_opponents": 0,
            "top1_coplay_share": 0.0,
            "top2_coplay_share": 0.0,
            "top_partners": [],
        }

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
def risk_decision(score: int) -> str:
    if score < T_APPROVE:
        return "APPROVE"
    if score < T_FAST_CHECK:
        return "FAST_CHECK"
    return "MANUAL_REVIEW"


def score_player(net_total, net_ring, net_mtt, comm, cop: dict):
    reasons = []
    score = 0

    # A) Profit/Loss logic
    if pd.notna(net_total):
        if net_total <= 0:
            reasons.append(f"net_total<=0 ({net_total})")
            score += 0
        else:
            reasons.append(f"net_total>0 ({net_total})")
            score += 30
    else:
        reasons.append("net_total is NaN")
        score += 10

    # MTT reduces suspicion if profit mainly via MTT
    if pd.notna(net_total) and net_total > 0 and pd.notna(net_mtt) and net_total != 0:
        mtt_share = min(1.0, max(0.0, net_mtt / net_total))
        if mtt_share >= 0.7:
            reasons.append(f"High MTT/SNG share ({mtt_share:.0%}) -> lower risk")
            score -= 15

    # Ring increases suspicion if profit mainly via ring
    if pd.notna(net_total) and net_total > 0 and pd.notna(net_ring) and net_total != 0:
        ring_share = min(1.0, max(0.0, net_ring / net_total))
        if ring_share >= 0.7:
            reasons.append(f"High Ring share ({ring_share:.0%}) -> higher risk")
            score += 15

    # Commission while losing
    if pd.notna(net_total) and net_total < 0 and pd.notna(comm):
        if comm > abs(net_total) * 0.5 and comm > 1:
            reasons.append(f"High commission with loss (comm={comm})")
            score += 30

    # B) Co-play signals
    if cop["sessions_count"] >= MIN_SESSIONS_FOR_COPLAY:
        if cop["top1_coplay_share"] >= COPLAY_TOP1_SHARE_SUSP:
            reasons.append(f"Top1 co-play share high ({cop['top1_coplay_share']:.0%})")
            score += 25
        if cop["top2_coplay_share"] >= COPLAY_TOP2_SHARE_SUSP:
            reasons.append(f"Top2 co-play share high ({cop['top2_coplay_share']:.0%})")
            score += 15
        if cop["unique_opponents"] <= 5:
            reasons.append(f"Few unique opponents ({cop['unique_opponents']})")
            score += 10
    else:
        reasons.append("Not enough sessions for stable co-play signal")

    score = int(max(0, min(100, score)))
    return score, reasons


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="PPPoker Risk Checker (Excel)", layout="wide")
st.title("PPPoker: авто-оценка риска вывода (Excel + Games CSV)")

with st.sidebar:
    st.header("Загрузка файлов")
    summary_file = st.file_uploader("Общая таблица (.xlsx)", type=["xlsx"])
    games_file = st.file_uploader("Игры/сессии (CSV из PPPoker)", type=["csv", "txt"])

st.divider()

if not summary_file or not games_file:
    st.info("Загрузи Excel 'общая таблица' и файл 'игры'.")
    st.stop()

# Read summary
df = read_summary_excel(summary_file)

df["_player_id"] = pd.to_numeric(df.iloc[:, IDX_ID], errors="coerce")
df = df.dropna(subset=["_player_id"]).copy()
df["_player_id"] = df["_player_id"].astype(int)

df["_net_total"] = to_float_series(df.iloc[:, IDX_NET_TOTAL])
df["_net_ring"] = to_float_series(df.iloc[:, IDX_NET_RING])
df["_net_mtt"] = to_float_series(df.iloc[:, IDX_NET_MTT])
df["_comm"] = to_float_series(df.iloc[:, IDX_COMM])

known_ids = set(df["_player_id"].tolist())

# Build sessions
sessions_df = build_coplay_from_games(games_file, known_ids)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Проверка вывода")
    default_id = int(df["_player_id"].iloc[0]) if len(df) else 0
    target_id = st.number_input("ID игрока", min_value=0, value=default_id, step=1)
    run = st.button("Оценить риск")

with col2:
    st.subheader("Диагностика")
    st.write(f"Игроков в Excel: {len(df)}")
    st.write(f"Сессий (из games): {len(sessions_df)}")
    st.write("sessions_df columns:", list(sessions_df.columns))
    st.caption("Если сессий = 0 — значит в файле 'Игры' не нашли строки 'ID игры:' или строки игроков ';ID игрока;...'.")

if not run:
    st.stop()

row_df = df[df["_player_id"] == int(target_id)]
if row_df.empty:
    st.error("ID игрока не найден в Excel.")
    st.stop()

row = row_df.iloc[0]
cop = coplay_features(int(target_id), sessions_df)

score, reasons = score_player(
    net_total=row["_net_total"],
    net_ring=row["_net_ring"],
    net_mtt=row["_net_mtt"],
    comm=row["_comm"],
    cop=cop,
)
decision = risk_decision(score)

st.success(f"Decision: {decision} | Risk score: {score}/100")

st.subheader("Почему так (reasons)")
for r in reasons[:12]:
    st.write("- " + r)

st.subheader("Co-play (контакты)")
st.write(
    {
        "sessions_count": cop["sessions_count"],
        "unique_opponents": cop["unique_opponents"],
        "top1_coplay_share": cop["top1_coplay_share"],
        "top2_coplay_share": cop["top2_coplay_share"],
    }
)

if cop["top_partners"]:
    st.dataframe(
        pd.DataFrame(cop["top_partners"], columns=["partner_id", "coplay_sessions"]),
        use_container_width=True,
    )

st.subheader("Экономика (из Excel)")
st.write(
    {
        "net_total (I)": row["_net_total"],
        "ring (O)": row["_net_ring"],
        "mtt/sng (P)": row["_net_mtt"],
        "commission (AB)": row["_comm"],
    }
)
