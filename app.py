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


# Excel columns from "общая таблица"
IDX_ID = col_idx("A")         # A - ID
IDX_NET_TOTAL = col_idx("I")  # I - общий выигрыш
IDX_NET_RING = col_idx("O")   # O - Ring Game
IDX_NET_MTT = col_idx("P")    # P - MTT/SNG
IDX_COMM = col_idx("AB")      # AB - комиссия


# ---------------------------
# Thresholds / weights
# ---------------------------
T_APPROVE = 25
T_FAST_CHECK = 55

# co-play thresholds
MIN_SESSIONS_FOR_COPLAY = 6
COPLAY_TOP1_SHARE_SUSP = 0.60
COPLAY_TOP2_SHARE_SUSP = 0.80

# transfer proxy thresholds
PAIR_NET_TRANSFER_ALERT_RING = 25.0
PAIR_NET_TRANSFER_ALERT_TOUR = 60.0  # tournaments are noisier -> higher threshold
PAIR_DOMINANCE_ALERT = 0.70

# single-game extremes
SINGLE_GAME_WIN_ALERT_RING = 60.0
SINGLE_GAME_LOSS_ALERT_RING = 60.0
SINGLE_GAME_WIN_ALERT_TOUR = 150.0
SINGLE_GAME_LOSS_ALERT_TOUR = 150.0


# ---------------------------
# Helpers
# ---------------------------
def to_float(x) -> float:
    if x is None:
        return np.nan
    s = str(x).strip().replace("\u00a0", "").replace(",", ".")
    if s == "" or s.lower() in ("none", "nan"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def to_float_series(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.replace("\u00a0", "", regex=False).str.strip()
    x = x.str.replace(",", ".", regex=False)
    x = x.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    return pd.to_numeric(x, errors="coerce")


def fmt_money(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NaN"
    return f"{v:.2f}"


def risk_decision(score: int) -> str:
    if score < T_APPROVE:
        return "APPROVE"
    if score < T_FAST_CHECK:
        return "FAST_CHECK"
    return "MANUAL_REVIEW"


# ---------------------------
# Read summary (Excel)
# ---------------------------
def read_summary_excel(uploaded_xlsx) -> pd.DataFrame:
    return pd.read_excel(uploaded_xlsx, engine="openpyxl", header=None)


# ---------------------------
# Parse PPPoker games CSV blocks
# ---------------------------
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\-]+)", re.IGNORECASE)
PLAYER_ROW_ID_RE = re.compile(r"(?:^|;)\s*(\d{6,10})\s*;", re.IGNORECASE)

# Lines that indicate tournament-like format
TOUR_HINT_RE = re.compile(r"\bPPST/|бай-ин:\s*|satellite|pko|mko\b", re.IGNORECASE)
# Lines that indicate ring/cash-like format
RING_HINT_RE = re.compile(r"\bPPSR/|NLH\s+\d|\bPLO\b|Bomb Pot|Ante\b", re.IGNORECASE)


def classify_game_type(lines_in_block: list[str]) -> str:
    """
    Returns: 'TOURNAMENT' or 'RING' or 'UNKNOWN'
    Based on text lines like:
      PPST/NLH ... Бай-ин...
      PPSR/NLH ... 0.05/0.1 ...
    """
    text = " ".join(lines_in_block)
    if TOUR_HINT_RE.search(text):
        return "TOURNAMENT"
    if RING_HINT_RE.search(text):
        return "RING"
    return "UNKNOWN"


def parse_games_csv(uploaded_file, known_player_ids: set[int]) -> pd.DataFrame:
    """
    Output rows: game_id, game_type, player_id, win, fee
    Supports both tournament blocks (Выигрыш/Комиссия) and ring blocks (Выигрыш игрока ...).
    """
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    lines = text.splitlines()

    rows = []
    current_game_id = None
    current_block_lines = []
    header_cols = None
    current_game_type = "UNKNOWN"

    def split_semicolon(line: str):
        return [c.strip().strip('"') for c in line.split(";")]

    def finalize_block():
        nonlocal current_block_lines, current_game_type
        if current_block_lines:
            current_game_type = classify_game_type(current_block_lines)

    for line in lines:
        m_game = GAME_ID_RE.search(line)
        if m_game:
            # new block -> finalize previous
            finalize_block()
            current_game_id = m_game.group(1).strip()
            current_block_lines = [line]
            header_cols = None
            current_game_type = "UNKNOWN"
            continue

        if not current_game_id:
            continue

        current_block_lines.append(line)

        # Detect header row inside the block
        if "ID игрока" in line:
            header_cols = split_semicolon(line)
            continue

        # Skip totals
        if "Итог" in line:
            continue

        # Parse player row
        m_pid = PLAYER_ROW_ID_RE.search(line)
        if not m_pid:
            continue

        pid = int(m_pid.group(1))
        if pid not in known_player_ids:
            continue

        parts = split_semicolon(line)

        win = np.nan
        fee = np.nan

        # Tournament style: explicit columns
        if header_cols is not None and len(parts) == len(header_cols):
            if "Выигрыш" in header_cols:
                win = to_float(parts[header_cols.index("Выигрыш")])
            if "Комиссия" in header_cols:
                fee = to_float(parts[header_cols.index("Комиссия")])
            # PKO: fee exists, win exists, bounty is separate (we ignore bounty here on purpose)
        else:
            # Ring fallback: ;ID;Nick;Name;Buy-in;Hands;Выигрыш игрока;...
            if len(parts) >= 7:
                win = to_float(parts[6])
            # fee fallback: take small positive from row tail (conservative)
            tail = parts[-6:] if len(parts) >= 6 else parts
            candidates = [to_float(t) for t in tail]
            candidates = [c for c in candidates if not np.isnan(c)]
            if candidates:
                pos = [c for c in candidates if c >= 0]
                fee = min(pos) if pos else np.nan

        # game_type for the current row: classify from accumulated block lines so far
        gtype = classify_game_type(current_block_lines)

        rows.append(
            {"game_id": current_game_id, "game_type": gtype, "player_id": pid, "win": win, "fee": fee}
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["game_id", "game_type", "player_id", "win", "fee"])

    # Normalize UNKNOWN: if many blocks are UNKNOWN, keep them but score lower confidence
    return df


def build_sessions_from_games(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["session_id", "game_type", "players"])

    g = (
        games_df.groupby(["game_id", "game_type"])["player_id"]
        .apply(lambda x: sorted(set(int(v) for v in x.dropna().tolist())))
        .reset_index()
        .rename(columns={"game_id": "session_id", "player_id": "players"})
    )
    g = g[g["players"].apply(lambda ps: len(ps) >= 2)].copy()
    if g.empty:
        return pd.DataFrame(columns=["session_id", "game_type", "players"])
    return g


# ---------------------------
# Co-play features per type
# ---------------------------
def coplay_features(target_id: int, sessions_df: pd.DataFrame, game_type: str | None = None) -> dict:
    if sessions_df.empty or "players" not in sessions_df.columns:
        return {
            "sessions_count": 0,
            "unique_opponents": 0,
            "top1_coplay_share": 0.0,
            "top2_coplay_share": 0.0,
            "top_partners": [],
        }

    df = sessions_df
    if game_type:
        df = df[df["game_type"] == game_type]

    rows = df[df["players"].apply(lambda ps: target_id in ps)]
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
        "top_partners": partners[:8],
    }


# ---------------------------
# Transfer proxy per type
# ---------------------------
def transfer_features(target_id: int, games_df: pd.DataFrame, game_type: str | None = None) -> dict:
    if games_df.empty:
        return {
            "target_games": 0,
            "target_total_win_from_games": 0.0,
            "top_sources": [],
            "top_source_net": 0.0,
            "top_source_share": 0.0,
            "single_game_extremes": [],
        }

    df = games_df.copy()
    if game_type:
        df = df[df["game_type"] == game_type]

    df = df[pd.notna(df["win"])].copy()
    if df.empty:
        return {
            "target_games": 0,
            "target_total_win_from_games": 0.0,
            "top_sources": [],
            "top_source_net": 0.0,
            "top_source_share": 0.0,
            "single_game_extremes": [],
        }

    t = df[df["player_id"] == target_id][["game_id", "win"]].copy()
    target_games = int(len(t))
    if target_games == 0:
        return {
            "target_games": 0,
            "target_total_win_from_games": 0.0,
            "top_sources": [],
            "top_source_net": 0.0,
            "top_source_share": 0.0,
            "single_game_extremes": [],
        }

    transfer = {}
    extremes = []

    if game_type == "TOURNAMENT":
        win_alert = SINGLE_GAME_WIN_ALERT_TOUR
        loss_alert = SINGLE_GAME_LOSS_ALERT_TOUR
    else:
        win_alert = SINGLE_GAME_WIN_ALERT_RING
        loss_alert = SINGLE_GAME_LOSS_ALERT_RING

    for _, tr in t.iterrows():
        gid = tr["game_id"]
        t_win = float(tr["win"])
        game = df[df["game_id"] == gid][["player_id", "win"]]

        if t_win >= win_alert or t_win <= -loss_alert:
            extremes.append({"game_id": gid, "target_win": t_win})

        if t_win <= 0:
            continue

        losers = game[game["win"] < 0]
        if losers.empty:
            continue

        for _, lr in losers.iterrows():
            pid = int(lr["player_id"])
            loss = float(-lr["win"])
            amt = min(t_win, loss)
            transfer[pid] = transfer.get(pid, 0.0) + amt

    total_from_sources = float(sum(transfer.values()))
    sources_sorted = sorted(transfer.items(), key=lambda x: x[1], reverse=True)

    top_source_net = float(sources_sorted[0][1]) if sources_sorted else 0.0
    top_source_share = (top_source_net / total_from_sources) if total_from_sources > 0 else 0.0

    return {
        "target_games": target_games,
        "target_total_win_from_games": float(t["win"].sum()),
        "top_sources": sources_sorted[:8],
        "top_source_net": top_source_net,
        "top_source_share": float(top_source_share),
        "single_game_extremes": extremes[:10],
    }


# ---------------------------
# Commentary + scoring (type-aware)
# ---------------------------
def build_commentary_and_score(row, cop_ring, cop_tour, trf_ring, trf_tour, coverage: dict):
    net_total = row["_net_total"]
    net_ring = row["_net_ring"]
    net_mtt = row["_net_mtt"]
    comm = row["_comm"]

    reasons = []
    comments = []
    score = 0

    # Coverage comment
    comments.append(
        f"Покрытие по файлу 'Игры': RING игр с результатом={coverage['ring_games_with_target']}, "
        f"TOURNAMENT игр с результатом={coverage['tour_games_with_target']}, UNKNOWN={coverage['unknown_games_with_target']}."
    )

    # Excel side
    if pd.notna(net_total) and net_total > 0:
        score += 30
        reasons.append(f"Плюс по общей таблице: net_total={fmt_money(net_total)}")
        comments.append(f"По общей таблице (I) игрок в плюсе: {fmt_money(net_total)}.")
    elif pd.notna(net_total) and net_total <= 0:
        reasons.append(f"Не в плюсе по общей таблице: net_total={fmt_money(net_total)}")
        comments.append(f"По общей таблице (I) игрок не в плюсе: {fmt_money(net_total)}.")
    else:
        score += 10
        reasons.append("Нет net_total (I)")
        comments.append("По общей таблице (I) общий выигрыш не распознан.")

    if pd.notna(comm):
        comments.append(f"Комиссия по общей таблице (AB): {fmt_money(comm)}.")
    else:
        score += 5
        reasons.append("Нет комиссии AB")
        comments.append("Комиссия (AB) не распознана.")

    # Modality shares
    if pd.notna(net_total) and net_total > 0 and pd.notna(net_mtt) and net_total != 0:
        mtt_share = float(min(1.0, max(0.0, net_mtt / net_total)))
        if mtt_share >= 0.7:
            score -= 15
            reasons.append(f"Профит в основном MTT/SNG: {mtt_share:.0%}")
            comments.append(f"Профит в основном MTT/SNG (P≈{mtt_share:.0%}) — снижает риск.")
        else:
            comments.append(f"Доля MTT/SNG в профите: {mtt_share:.0%}.")

    if pd.notna(net_total) and net_total > 0 and pd.notna(net_ring) and net_total != 0:
        ring_share = float(min(1.0, max(0.0, net_ring / net_total)))
        if ring_share >= 0.7:
            score += 15
            reasons.append(f"Профит в основном Ring: {ring_share:.0%}")
            comments.append(f"Профит в основном Ring (O≈{ring_share:.0%}) — повышает риск.")
        else:
            comments.append(f"Доля Ring в профите: {ring_share:.0%}.")

    # Co-play analysis: ring is stronger signal
    if cop_ring["sessions_count"] > 0:
        comments.append(
            f"RING co-play: сессий={cop_ring['sessions_count']}, уникальных оппонентов={cop_ring['unique_opponents']}."
        )
        if cop_ring["sessions_count"] >= MIN_SESSIONS_FOR_COPLAY:
            if cop_ring["top1_coplay_share"] >= COPLAY_TOP1_SHARE_SUSP:
                score += 30
                reasons.append(f"RING: один партнёр слишком часто ({cop_ring['top1_coplay_share']:.0%})")
            if cop_ring["top2_coplay_share"] >= COPLAY_TOP2_SHARE_SUSP:
                score += 15
                reasons.append(f"RING: узкий круг (топ-2={cop_ring['top2_coplay_share']:.0%})")
            if cop_ring["unique_opponents"] <= 5:
                score += 10
                reasons.append("RING: мало уникальных оппонентов")
        else:
            score += 5
            reasons.append("RING: мало сессий для устойчивого co-play")
    else:
        comments.append("RING co-play: данных по совместной игре нет или недостаточно.")

    if cop_tour["sessions_count"] > 0:
        comments.append(
            f"TOURNAMENT co-play: сессий={cop_tour['sessions_count']}, уникальных оппонентов={cop_tour['unique_opponents']}."
        )
        # Tournament co-play is weaker evidence -> smaller weights
        if cop_tour["sessions_count"] >= MIN_SESSIONS_FOR_COPLAY:
            if cop_tour["top1_coplay_share"] >= 0.70:
                score += 10
                reasons.append(f"TOURNAMENT: частый партнёр ({cop_tour['top1_coplay_share']:.0%})")
        else:
            reasons.append("TOURNAMENT: мало сессий для co-play")
    else:
        comments.append("TOURNAMENT co-play: данных нет или недостаточно.")

    # Transfer proxy: ring stronger, tournaments need higher thresholds
    if trf_ring["target_games"] > 0:
        comments.append(
            f"RING transfer-proxy: игр={trf_ring['target_games']}, суммарный результат={fmt_money(trf_ring['target_total_win_from_games'])}."
        )
        if trf_ring["single_game_extremes"]:
            score += 10
            reasons.append("RING: аномальный результат в одной игре")
        if trf_ring["top_sources"]:
            top_pid, top_amt = trf_ring["top_sources"][0]
            comments.append(f"RING: главный 'источник' {top_pid} дал ~{fmt_money(top_amt)}.")
            if top_amt >= PAIR_NET_TRANSFER_ALERT_RING:
                score += 30
                reasons.append(f"RING: крупный поток от одного ID ({top_pid} -> {fmt_money(top_amt)})")
            if trf_ring["top_source_share"] >= PAIR_DOMINANCE_ALERT and top_amt >= (PAIR_NET_TRANSFER_ALERT_RING / 2):
                score += 20
                reasons.append(f"RING: доминирование источника ({trf_ring['top_source_share']:.0%})")
    else:
        comments.append("RING transfer-proxy: нет игр с результатом для игрока.")

    if trf_tour["target_games"] > 0:
        comments.append(
            f"TOURNAMENT transfer-proxy: игр={trf_tour['target_games']}, суммарный результат={fmt_money(trf_tour['target_total_win_from_games'])}."
        )
        if trf_tour["single_game_extremes"]:
            score += 5
            reasons.append("TOURNAMENT: аномальный результат в одной игре")
        if trf_tour["top_sources"]:
            top_pid, top_amt = trf_tour["top_sources"][0]
            comments.append(f"TOURNAMENT: главный 'источник' {top_pid} дал ~{fmt_money(top_amt)}.")
            if top_amt >= PAIR_NET_TRANSFER_ALERT_TOUR:
                score += 10
                reasons.append(f"TOURNAMENT: крупный поток от одного ID ({top_pid} -> {fmt_money(top_amt)})")
            if trf_tour["top_source_share"] >= 0.80 and top_amt >= (PAIR_NET_TRANSFER_ALERT_TOUR / 2):
                score += 10
                reasons.append("TOURNAMENT: доминирование источника")
    else:
        comments.append("TOURNAMENT transfer-proxy: нет игр с результатом для игрока.")

    # If no games coverage at all -> risk up (insufficient evidence)
    if coverage["ring_games_with_target"] == 0 and coverage["tour_games_with_target"] == 0 and coverage["unknown_games_with_target"] == 0:
        score += 15
        reasons.append("Нет данных по игроку в файле 'Игры' (нельзя подтвердить чистоту)")

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)

    if decision == "APPROVE":
        verdict = "ВЫВОД РАЗРЕШИТЬ: по текущим данным явных признаков перелива не выявлено."
    elif decision == "FAST_CHECK":
        verdict = "БЫСТРАЯ ПРОВЕРКА: есть слабые/неустойчивые признаки, требуется повышенное внимание."
    else:
        verdict = "РУЧНАЯ ПРОВЕРКА СБ: повышенная вероятность перелива/сговора по текущим данным."

    if decision == "MANUAL_REVIEW":
        # include only the strongest reasons
        key = []
        for r in reasons:
            if any(k in r.lower() for k in ["ring:", "поток", "доминир", "узкий", "партн", "аномал", "нет данных"]):
                key.append(r)
        if not key:
            key = reasons[:6]
        comments.append("Причины для ручной проверки СБ: " + "; ".join(key) + ".")

    return score, decision, reasons, comments, verdict


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="PPPoker Anti-Fraud (Type-aware)", layout="wide")
st.title("PPPoker: проверка риска вывода (RING vs TOURNAMENT)")

with st.sidebar:
    st.header("Загрузка файлов")
    summary_file = st.file_uploader("Общая таблица (.xlsx) — без шапки", type=["xlsx"])
    games_file = st.file_uploader("Игры (.csv/.txt) — выгрузка PPPoker", type=["csv", "txt"])
    show_debug = st.checkbox("Показать технические детали", value=False)

st.divider()

if not summary_file or not games_file:
    st.info("Загрузи Excel 'общая таблица' и файл 'Игры'.")
    st.stop()

# Read Excel summary
df = read_summary_excel(summary_file)

df["_player_id"] = pd.to_numeric(df.iloc[:, IDX_ID], errors="coerce")
df = df.dropna(subset=["_player_id"]).copy()
df["_player_id"] = df["_player_id"].astype(int)

df["_net_total"] = to_float_series(df.iloc[:, IDX_NET_TOTAL])
df["_net_ring"] = to_float_series(df.iloc[:, IDX_NET_RING])
df["_net_mtt"] = to_float_series(df.iloc[:, IDX_NET_MTT])
df["_comm"] = to_float_series(df.iloc[:, IDX_COMM])

known_ids = set(df["_player_id"].tolist())

# Parse games
games_df = parse_games_csv(games_file, known_ids)
sessions_df = build_sessions_from_games(games_df)

# UI columns
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Проверка")
    default_id = int(df["_player_id"].iloc[0]) if len(df) else 0
    target_id = st.number_input("ID игрока на вывод", min_value=0, value=default_id, step=1)
    run = st.button("Проверить")

with col2:
    st.subheader("Статус данных")
    st.write(f"Игроков в Excel: {len(df)}")
    st.write(f"Строк результатов в games: {len(games_df)}")
    st.write(f"Сессий (ID игры с >=2 игроками): {len(sessions_df)}")
    if not games_df.empty:
        st.write("Распределение типов игр:", dict(games_df["game_type"].value_counts()))
    if show_debug:
        st.write("games_df columns:", list(games_df.columns))
        st.write("sessions_df columns:", list(sessions_df.columns))

if not run:
    st.stop()

row_df = df[df["_player_id"] == int(target_id)]
if row_df.empty:
    st.error("ID игрока не найден в Excel.")
    st.stop()

row = row_df.iloc[0]

# Coverage info
target_games = games_df[games_df["player_id"] == int(target_id)] if not games_df.empty else pd.DataFrame()
coverage = {
    "ring_games_with_target": int((target_games["game_type"] == "RING").sum()) if not target_games.empty else 0,
    "tour_games_with_target": int((target_games["game_type"] == "TOURNAMENT").sum()) if not target_games.empty else 0,
    "unknown_games_with_target": int((target_games["game_type"] == "UNKNOWN").sum()) if not target_games.empty else 0,
}

# Per-type analytics
cop_ring = coplay_features(int(target_id), sessions_df, "RING")
cop_tour = coplay_features(int(target_id), sessions_df, "TOURNAMENT")

trf_ring = transfer_features(int(target_id), games_df, "RING")
trf_tour = transfer_features(int(target_id), games_df, "TOURNAMENT")

score, decision, reasons, comments, verdict = build_commentary_and_score(
    row, cop_ring, cop_tour, trf_ring, trf_tour, coverage
)

# Main verdict
if decision == "APPROVE":
    st.success(f"{verdict}  (score {score}/100)")
elif decision == "FAST_CHECK":
    st.warning(f"{verdict}  (score {score}/100)")
else:
    st.error(f"{verdict}  (score {score}/100)")

# Comments
st.subheader("Комментарии по анализу")
for c in comments:
    st.write("- " + c)

# Numeric blocks
st.subheader("Цифры (Excel)")
st.write(
    {
        "net_total (I)": row["_net_total"],
        "ring (O)": row["_net_ring"],
        "mtt/sng (P)": row["_net_mtt"],
        "commission (AB)": row["_comm"],
    }
)

st.subheader("Цифры (RING co-play)")
st.write(
    {
        "sessions_count": cop_ring["sessions_count"],
        "unique_opponents": cop_ring["unique_opponents"],
        "top1_coplay_share": cop_ring["top1_coplay_share"],
        "top2_coplay_share": cop_ring["top2_coplay_share"],
    }
)
if cop_ring["top_partners"]:
    st.dataframe(pd.DataFrame(cop_ring["top_partners"], columns=["partner_id", "coplay_sessions"]), use_container_width=True)

st.subheader("Цифры (TOURNAMENT co-play)")
st.write(
    {
        "sessions_count": cop_tour["sessions_count"],
        "unique_opponents": cop_tour["unique_opponents"],
        "top1_coplay_share": cop_tour["top1_coplay_share"],
        "top2_coplay_share": cop_tour["top2_coplay_share"],
    }
)
if cop_tour["top_partners"]:
    st.dataframe(pd.DataFrame(cop_tour["top_partners"], columns=["partner_id", "coplay_sessions"]), use_container_width=True)

st.subheader("Цифры (RING transfer-proxy)")
st.write(
    {
        "target_games_in_file": trf_ring["target_games"],
        "target_total_win_from_games": trf_ring["target_total_win_from_games"],
        "top_source_net": trf_ring["top_source_net"],
        "top_source_share": trf_ring["top_source_share"],
    }
)
if trf_ring["top_sources"]:
    st.dataframe(
        pd.DataFrame(trf_ring["top_sources"], columns=["source_player_id", "estimated_transfer_to_target"]),
        use_container_width=True,
    )

st.subheader("Цифры (TOURNAMENT transfer-proxy)")
st.write(
    {
        "target_games_in_file": trf_tour["target_games"],
        "target_total_win_from_games": trf_tour["target_total_win_from_games"],
        "top_source_net": trf_tour["top_source_net"],
        "top_source_share": trf_tour["top_source_share"],
    }
)
if trf_tour["top_sources"]:
    st.dataframe(
        pd.DataFrame(trf_tour["top_sources"], columns=["source_player_id", "estimated_transfer_to_target"]),
        use_container_width=True,
    )

st.subheader("Триггеры (reasons)")
for r in reasons[:25]:
    st.write("- " + r)
