"""
PPPoker Chip Dumping & Collusion Detection System
================================================
Профессиональная система аналитики для выявления переливов фишек и сговоров.
Оптимизирована для скорости и точности с использованием кэширования и индексации.
"""

import io
import re
import json
import datetime as dt
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# =========================
# CONFIG & THRESHOLDS
# =========================
APP_TITLE = "PPPoker | ANTI-FRAUD — Выявление переливов фишек (chip dumping / collusion)"
CACHE_DIR = Path(".pppoker_app_cache")
CACHE_DIR.mkdir(exist_ok=True)

DB_KEY = "db_file"
GAMES_KEY = "games_file"

# ========================
# RISK SCORING THRESHOLDS (улучшенные)
# ========================
T_APPROVE = 20  # Зелёный свет — проводить выплату
T_WARN = 45     # Жёлтый — быстрая проверка
# > 45 = красный — в СБ на ручную проверку

# === Ring Game метрики (BB-aware) ===
# Шаг 1: покрытие и основной профиль
RING_MIN_GAMES_MEANINGFUL = 5  # минимум игр для анализа
RING_WIN_RATIO_ALERT = 0.65    # если выиграл в 65%+ игр — подозрение

# Шаг 2: flow анализ (HU и многочеловечные)
# HU специально (2 игрока):
HU_DOMINANCE_MIN_GAMES = 2     # минимум HU-игр с одним противником
HU_ONE_SIDED_RATIO = 0.85      # если 85%+ побед против одного — RED FLAG
HU_PROFIT_VS_ONE_ALERT_BB = 10  # >= 10 BB профит с одним в HU

# Multi-way (3+ игроков):
MULTIWAY_DOMINANCE_MIN_GAMES = 3
MULTIWAY_WIN_RATIO_ALERT = 0.70

# Sharkpool метрики (Ring совместные сессии):
SHARED_SESSION_MIN = 2         # минимум сессий с партнёром
PAIR_FLOW_ALERT_BB = 8.0       # >= 8 BB net с одним партнёром
PAIR_FLOW_CRITICAL_BB = 20.0   # >= 20 BB = высокий риск
PAIR_FLOW_AMOUNT_ALERT = 30.0  # в деньгах (параллельный критерий)

# Consistency маркеры:
FLOW_CONSISTENCY_ALERT = 0.78  # >= 78% денег идёт одному
FLOW_DIRECTIONALITY_ALERT = 0.82  # >= 82% однонаправленного потока

# === MTT/Tournament метрики ===
TOUR_WIN_RATIO_ALERT = 0.55    # турниры более волатильны
TOUR_EXTREME_SINGLE_GAME = 80.0 # разовый выигрыш > 80 USD

# === Database (Общий) метрики ===
DB_WEEK_CONCENTRATION = 0.45   # если 45%+ выигрыша в одну неделю
DB_RG_VS_MTT_RATIO = 0.65      # если 65%+ с Ring Game
DB_JACKPOT_SPIKE = 1.5         # если джекпот > 1.5x комиссии

# ========================
# DATA STRUCTURES & REGEX
# ========================
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)
TABLE_NAME_RE = re.compile(r"Название стола:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
START_END_RE = re.compile(
    r"Начало:\s*([0-9/:\s]+?)\s+By.+?Окончание:\s*([0-9/:\s]+)",
    re.IGNORECASE | re.DOTALL
)

RING_HINT_RE = re.compile(r"\bPPSR\b|PLO|OFC|NLH|Bomb Pot|Ante|3-1|HU\b|Heads", re.IGNORECASE)
TOUR_HINT_RE = re.compile(r"\bPPST\b|Бай-ин:|satellite|pko|mko\b|SNG\b|MTT\b", re.IGNORECASE)
STAKES_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)")

# DB column names
COL_WEEK = "Номер недели"
COL_PLAYER_ID = "ID игрока"
COL_COUNTRY = "Страна/регион"
COL_NICK = "Ник"
COL_IGN = "Игровое имя"
COL_AGENT = "Агент"
COL_AGENT_ID = "ID агента"
COL_SUPER_AGENT = "Супер-агент"
COL_SUPER_AGENT_ID = "ID cупер-агента"

COL_J_TOTAL = "Общий выигрыш игроков + События"
COL_PLAYER_WIN_TOTAL = "Выигрыш игрока Общий"
COL_PLAYER_WIN_RING = "Выигрыш игрока Ring Game"
COL_PLAYER_WIN_MTT = "Выигрыш игрока MTT, SNG"
COL_WIN_JACKPOT = "Выигрыш игрока Jackpot"
COL_WIN_EQUITY = "Выигрыш игрока Выдача эквити"

COL_CLUB_INCOME_TOTAL = "Доход клуба Общий"
COL_CLUB_COMMISSION = "Доход клуба Комиссия"
COL_CLUB_COMM_PPST = "Доход клуба Комиссия (только PPST)"
COL_CLUB_COMM_PPSR = "Доход клуба Комиссия (только PPSR)"
COL_CLUB_COMM_NO_PPST = "Доход клуба Комиссия (без PPST)"
COL_CLUB_COMM_NO_PPSR = "Доход клуба Комиссия (без PPSR)"

EXTRA_PLAYER_WIN_COL_PREFIX = "Выигрыш игрока "


# =========================
# FILE PERSISTENCE LAYER
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
            pass
    return BytesFile(content, name)


def cache_clear(key: str) -> None:
    for p in (_bin_path(key), _meta_path(key)):
        if p.exists():
            p.unlink()


def resolve_file(key: str, uploaded_file):
    if uploaded_file is not None:
        cache_save_uploaded(key, uploaded_file)
        return uploaded_file
    return cache_load_file(key)


# =========================
# HELPERS
# =========================
def detect_delimiter(sample_bytes: bytes) -> str:
    sample = sample_bytes[:8000].decode("utf-8", errors="ignore")
    candidates = [";", ",", "\t"]
    counts = {c: sample.count(c) for c in candidates}
    return max(counts, key=counts.get)


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
    x = s.copy()
    x = x.fillna("").astype(str).str.replace("\u00a0", "", regex=False).str.strip()
    x = x.str.replace(",", ".", regex=False)
    x = x.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    return pd.to_numeric(x, errors="coerce")


def fmt_money(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):.2f}"


def fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x) * 100:.0f}%"


def fmt_bb(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.1f} BB"


def safe_div(a, b):
    if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return a / b


# =========================
# DATABASE LOADER
# =========================
def load_db_any(file_obj) -> pd.DataFrame:
    name = (getattr(file_obj, "name", "") or "").lower()
    content = file_obj.getvalue()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content), sheet_name="Общий")
    else:
        sep = detect_delimiter(content)
        df = pd.read_csv(io.BytesIO(content), sep=sep, encoding="utf-8-sig")

    required = [
        COL_WEEK,
        COL_PLAYER_ID,
        COL_J_TOTAL,
        COL_PLAYER_WIN_TOTAL,
        COL_PLAYER_WIN_RING,
        COL_PLAYER_WIN_MTT,
        COL_CLUB_INCOME_TOTAL,
        COL_CLUB_COMMISSION,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"В DB не найдены обязательные колонки: {missing}")

    out = pd.DataFrame()
    out["_week"] = pd.to_numeric(df[COL_WEEK], errors="coerce").fillna(-1).astype(int)
    out["_player_id"] = pd.to_numeric(df[COL_PLAYER_ID], errors="coerce")
    out = out.dropna(subset=["_player_id"]).copy()
    out["_player_id"] = out["_player_id"].astype(int)

    for src, dst in [
        (COL_COUNTRY, "_country"),
        (COL_NICK, "_nick"),
        (COL_IGN, "_ign"),
        (COL_AGENT, "_agent"),
        (COL_SUPER_AGENT, "_super_agent"),
    ]:
        out[dst] = (
            df.loc[out.index, src].astype(str).fillna("").str.strip()
            if src in df.columns
            else ""
        )

    for src, dst in [(COL_AGENT_ID, "_agent_id"), (COL_SUPER_AGENT_ID, "_super_agent_id")]:
        out[dst] = pd.to_numeric(df.loc[out.index, src], errors="coerce") if src in df.columns else np.nan

    base_num_cols = {
        COL_J_TOTAL: "_j_total",
        COL_PLAYER_WIN_TOTAL: "_p_total",
        COL_PLAYER_WIN_RING: "_p_ring",
        COL_PLAYER_WIN_MTT: "_p_mtt",
        COL_WIN_JACKPOT: "_p_jackpot",
        COL_WIN_EQUITY: "_p_equity",
        COL_CLUB_INCOME_TOTAL: "_club_income_total",
        COL_CLUB_COMMISSION: "_club_comm_total",
        COL_CLUB_COMM_PPST: "_club_comm_ppst",
        COL_CLUB_COMM_PPSR: "_club_comm_ppsr",
        COL_CLUB_COMM_NO_PPST: "_club_comm_no_ppst",
        COL_CLUB_COMM_NO_PPSR: "_club_comm_no_ppsr",
    }
    for src, dst in base_num_cols.items():
        out[dst] = to_float_series(df.loc[out.index, src]) if src in df.columns else np.nan

    extra_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(EXTRA_PLAYER_WIN_COL_PREFIX)]
    already = set(base_num_cols.keys())
    extra_cols = [c for c in extra_cols if c not in already]
    for c in extra_cols:
        norm = "_p_extra__" + re.sub(r"[^a-zA-Z0-9а-яА-Я_]+", "_", c.replace(EXTRA_PLAYER_WIN_COL_PREFIX, "").strip())
        out[norm] = to_float_series(df.loc[out.index, c])

    return out


# =========================
# GAMES PARSER (OPTIMIZED)
# =========================
def _extract_bb_any(*texts: str) -> float:
    for t in texts:
        if not t:
            continue
        m = STAKES_RE.search(str(t).replace(",", "."))
        if m:
            try:
                bb = float(m.group(2))
                if bb > 0:
                    return bb
            except Exception:
                pass
    return np.nan


def _classify_game_type(descriptor: str, table_name: str = "") -> str:
    s = (descriptor or "") + " " + (table_name or "")
    if not s.strip():
        return "UNKNOWN"
    if TOUR_HINT_RE.search(s):
        return "TOURNAMENT"
    if STAKES_RE.search(s):
        return "RING"
    if RING_HINT_RE.search(s):
        return "RING"
    return "UNKNOWN"


def _split_semicolon(line: str) -> list[str]:
    return [p.strip().strip('"') for p in line.split(";")]


def parse_games_pppoker_export(file_obj) -> pd.DataFrame:
    """Parse PPPoker game export with full metadata."""
    text = file_obj.getvalue().decode("utf-8", errors="ignore")
    lines = text.splitlines()

    rows = []
    current = {
        "game_id": None,
        "table_name": "",
        "descriptor": "",
        "game_type": "UNKNOWN",
        "product": "",
        "bb": np.nan,
        "start_time": None,
        "end_time": None,
    }

    header = None
    idx = {}

    for line in lines:
        m = GAME_ID_RE.search(line)
        if m:
            current = {
                "game_id": m.group(1).strip(),
                "table_name": "",
                "descriptor": "",
                "game_type": "UNKNOWN",
                "product": "",
                "bb": np.nan,
                "start_time": None,
                "end_time": None,
            }
            header = None
            idx = {}
            tm = TABLE_NAME_RE.search(line)
            if tm:
                current["table_name"] = tm.group(1).strip()
            continue

        if current["game_id"] is None:
            continue

        se = START_END_RE.search(line)
        if se:
            current["start_time"] = se.group(1).strip()
            current["end_time"] = se.group(2).strip()

        # Descriptor detection
        if ("ID игрока" not in line) and ("Итог" not in line):
            is_desc = False
            if ("PPSR" in line) or ("PPST" in line):
                is_desc = True
            elif TOUR_HINT_RE.search(line):
                is_desc = True
            elif STAKES_RE.search(line):
                is_desc = True

            if is_desc and (not current["descriptor"] or ("PPSR" in line) or ("PPST" in line)):
                current["descriptor"] = line.strip()
                current["product"] = "PPSR" if "PPSR" in line else ("PPST" if "PPST" in line else "")
                current["game_type"] = _classify_game_type(current["descriptor"], current["table_name"])
                if current["game_type"] == "RING":
                    current["bb"] = _extract_bb_any(current["descriptor"], current["table_name"])
                continue

        if "ID игрока" in line:
            header = _split_semicolon(line)

            def find(col):
                return header.index(col) if col in header else None

            idx = {
                "player_id": find("ID игрока"),
                "nick": find("Ник"),
                "ign": find("Игровое имя"),
                "hands": find("Раздачи"),
                "win": find("Выигрыш") or find("Выигрыш игрока"),
                "fee": find("Комиссия"),
            }
            continue

        if header is None:
            continue

        parts = _split_semicolon(line)
        if len(parts) < 2 or "Итог" in line:
            continue

        pid_idx = idx.get("player_id")
        if pid_idx is None or pid_idx >= len(parts):
            continue

        try:
            pid = int(float(str(parts[pid_idx]).replace(",", ".")))
        except Exception:
            continue

        row = {
            "game_id": current["game_id"],
            "game_type": current["game_type"],
            "product": current["product"],
            "table_name": current["table_name"],
            "descriptor": current["descriptor"],
            "bb": current["bb"],
            "start_time": current["start_time"],
            "end_time": current["end_time"],
            "player_id": pid,
            "nick": parts[idx["nick"]] if idx.get("nick") is not None and idx["nick"] < len(parts) else "",
            "ign": parts[idx["ign"]] if idx.get("ign") is not None and idx["ign"] < len(parts) else "",
            "hands": np.nan,
            "win_total": np.nan,
            "win_vs_opponents": np.nan,
            "fee": np.nan,
        }

        if idx.get("hands") is not None and idx["hands"] < len(parts):
            row["hands"] = to_float(parts[idx["hands"]])
        if idx.get("fee") is not None and idx["fee"] < len(parts):
            row["fee"] = to_float(parts[idx["fee"]])

        if current["game_type"] == "RING":
            hidx = idx.get("hands")
            if hidx is not None and hidx + 2 < len(parts):
                try:
                    row["win_total"] = to_float(parts[hidx + 1])
                    row["win_vs_opponents"] = to_float(parts[hidx + 2])
                except Exception:
                    pass
            if pd.isna(row["win_total"]):
                widx = idx.get("win")
                if widx is not None and widx < len(parts):
                    row["win_total"] = to_float(parts[widx])
        else:
            widx = idx.get("win")
            if widx is not None and widx < len(parts):
                row["win_total"] = to_float(parts[widx])

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_type",
                "product",
                "table_name",
                "descriptor",
                "bb",
                "start_time",
                "end_time",
                "player_id",
                "nick",
                "ign",
                "hands",
                "win_total",
                "win_vs_opponents",
                "fee",
            ]
        )

    for c in ["bb", "hands", "win_total", "win_vs_opponents", "fee"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["player_id", "game_id"]).copy()
    df["player_id"] = df["player_id"].astype(int)
    df["game_id"] = df["game_id"].astype(str)

    df.loc[(df["game_type"] == "UNKNOWN") & df["bb"].notna(), "game_type"] = "RING"
    return df


# =========================
# INDEXING & FEATURES
# =========================
def build_games_indexes(games_df: pd.DataFrame) -> dict:
    """Build indexes for fast lookups."""
    indexes = {
        "player_ring_games": {},  # pid -> [game_ids]
        "player_tour_games": {},  # pid -> [game_ids]
        "game_players": {},  # game_id -> [pids]
        "player_pair_ring": {},  # (p1, p2) -> DataFrame
        "player_pair_tour": {},  # (p1, p2) -> DataFrame
        "sessions_ring": {},  # set of unique (p1, p2) in same game
        "sessions_tour": {},
    }

    for _, row in games_df.iterrows():
        pid = int(row["player_id"])
        gid = str(row["game_id"])
        gtype = row["game_type"]

        if gtype == "RING":
            if pid not in indexes["player_ring_games"]:
                indexes["player_ring_games"][pid] = []
            indexes["player_ring_games"][pid].append(gid)
        elif gtype == "TOURNAMENT":
            if pid not in indexes["player_tour_games"]:
                indexes["player_tour_games"][pid] = []
            indexes["player_tour_games"][pid].append(gid)

        if gid not in indexes["game_players"]:
            indexes["game_players"][gid] = []
        indexes["game_players"][gid].append(pid)

    # Build session co-play matrices
    for gid, players in indexes["game_players"].items():
        gtype = games_df[games_df["game_id"] == gid]["game_type"].iloc[0]
        players = sorted(set(players))
        if len(players) >= 2:
            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    p1, p2 = players[i], players[j]
                    pair = tuple(sorted([p1, p2]))
                    if gtype == "RING":
                        indexes["sessions_ring"].setdefault(pair, set()).add(gid)
                    else:
                        indexes["sessions_tour"].setdefault(pair, set()).add(gid)

    return indexes


def extract_pair_stats(games_df: pd.DataFrame, p1: int, p2: int, game_type: str) -> dict:
    """Extract all stats for a pair of players."""
    pair_games = games_df[
        (games_df["game_id"].isin(
            games_df[(games_df["player_id"] == p1) | (games_df["player_id"] == p2)]["game_id"].unique()
        ))
        & (games_df["game_type"] == game_type)
    ]

    # Filter to games where BOTH players are present
    game_ids_with_both = []
    for gid in pair_games["game_id"].unique():
        g = pair_games[pair_games["game_id"] == gid]
        pids = set(g["player_id"].unique())
        if p1 in pids and p2 in pids:
            game_ids_with_both.append(gid)

    if not game_ids_with_both:
        return {
            "games_cnt": 0,
            "p1_total": 0.0,
            "p2_total": 0.0,
            "p1_avg": np.nan,
            "p2_avg": np.nan,
            "p1_win_cnt": 0,
            "p2_win_cnt": 0,
            "p1_win_ratio": 0.0,
            "p2_win_ratio": 0.0,
            "flow_from_p2_to_p1": 0.0,
            "flow_from_p1_to_p2": 0.0,
            "flow_total": 0.0,
            "flow_ratio": np.nan,
            "bb": np.nan,
            "shared_bb": 0.0,
        }

    pair_data = pair_games[pair_games["game_id"].isin(game_ids_with_both)]

    p1_data = pair_data[pair_data["player_id"] == p1]
    p2_data = pair_data[pair_data["player_id"] == p2]

    p1_total = p1_data["win_total"].sum()
    p2_total = p2_data["win_total"].sum()

    p1_wins = (p1_data["win_total"] > 0).sum()
    p2_wins = (p2_data["win_total"] > 0).sum()

    p1_games = len(p1_data)
    p2_games = len(p2_data)

    p1_win_ratio = safe_div(p1_wins, p1_games) or 0.0
    p2_win_ratio = safe_div(p2_wins, p2_games) or 0.0

    # Flow (one-directional transfer)
    flow_p2_to_p1 = max(0, -p2_total)  # p2's losses
    flow_p1_to_p2 = max(0, -p1_total)  # p1's losses
    flow_total = flow_p2_to_p1 + flow_p1_to_p2

    # Flow ratio (which direction)
    flow_ratio = np.nan
    if flow_total > 0:
        flow_ratio = max(flow_p2_to_p1, flow_p1_to_p2) / flow_total

    # BB conversion
    bb_val = pair_data[pair_data["bb"].notna()]["bb"].iloc[0] if not pair_data[pair_data["bb"].notna()].empty else np.nan
    shared_bb = 0.0
    if pd.notna(bb_val) and bb_val > 0:
        shared_bb = flow_total / bb_val

    return {
        "games_cnt": len(game_ids_with_both),
        "p1_total": float(p1_total),
        "p2_total": float(p2_total),
        "p1_avg": float(safe_div(p1_total, p1_games)) or np.nan,
        "p2_avg": float(safe_div(p2_total, p2_games)) or np.nan,
        "p1_win_cnt": int(p1_wins),
        "p2_win_cnt": int(p2_wins),
        "p1_win_ratio": float(p1_win_ratio),
        "p2_win_ratio": float(p2_win_ratio),
        "flow_from_p2_to_p1": float(flow_p2_to_p1),
        "flow_from_p1_to_p2": float(flow_p1_to_p2),
        "flow_total": float(flow_total),
        "flow_ratio": float(flow_ratio) if pd.notna(flow_ratio) else np.nan,
        "bb": float(bb_val) if pd.notna(bb_val) else np.nan,
        "shared_bb": float(shared_bb),
    }


# =========================
# FRAUD DETECTION SCORING
# =========================
def analyze_player(player_id: int, db_df: pd.DataFrame, games_df: pd.DataFrame, indexes: dict) -> tuple:
    """
    Comprehensive player fraud analysis.
    Returns: (score, decision, detailed_reasons)
    """
    score = 0
    reasons = []

    # === SECTION 1: Database (Общий) Analysis ===
    db_player = db_df[db_df["_player_id"] == player_id]
    if db_player.empty:
        return 5, "APPROVE", ["Игрок не найден в базе данных."], {}

    j_total = float(db_player["_j_total"].sum() or 0.0)
    p_total = float(db_player["_p_total"].sum() or 0.0)
    p_ring = float(db_player["_p_ring"].sum() or 0.0)
    p_mtt = float(db_player["_p_mtt"].sum() or 0.0)
    p_jackpot = float(db_player["_p_jackpot"].sum() or 0.0)
    p_equity = float(db_player["_p_equity"].sum() or 0.0)
    club_comm = float(db_player["_club_comm_total"].sum() or 0.0)

    events_delta = p_total - (p_ring + p_mtt + p_jackpot + p_equity)
    weeks = int(db_player["_week"].nunique())
    top_week_share = 0.0

    if weeks > 0:
        weekly = db_player.groupby("_week")["_j_total"].sum()
        if not weekly.empty:
            top_week = weekly.max()
            total = weekly.sum()
            if total > 0:
                top_week_share = top_week / total

    # Signal 1: Excessive concentration in single week
    if top_week_share >= DB_WEEK_CONCENTRATION and weeks >= 2:
        score += 10
        reasons.append(f"DB: Выигрыш концентрируется в одну неделю ({fmt_pct(top_week_share)}) из {weeks} недель.")

    # Signal 2: Suspicious Ring vs MTT ratio
    total_main = p_ring + p_mtt
    if total_main > 0:
        rg_ratio = p_ring / total_main
        if rg_ratio >= DB_RG_VS_MTT_RATIO:
            score += 6
            reasons.append(f"DB: {fmt_pct(rg_ratio)} выигрыша из Ring Game (обычно 50-60%).")

    # Signal 3: Jackpot anomaly
    if club_comm > 0 and p_jackpot > 0:
        jp_ratio = p_jackpot / club_comm
        if jp_ratio >= DB_JACKPOT_SPIKE:
            score += 5
            reasons.append(f"DB: Джекпот аномально высок: {fmt_pct(jp_ratio)} от комиссии.")

    # Signal 4: Events spike (может быть манипуляция через бонусы)
    if total_main > 0:
        events_share = abs(events_delta) / total_main
        if events_share >= 0.3:
            score += 4
            reasons.append(f"DB: Необычные события/дельта: {fmt_money(events_delta)} ({fmt_pct(events_share)}).")

    # === SECTION 2: Ring Game Deep Dive ===
    ring_games = games_df[
        (games_df["player_id"] == player_id) & (games_df["game_type"] == "RING")
    ].copy()

    if not ring_games.empty:
        ring_games_cnt = len(ring_games)

        # Signal 5: Win ratio in Ring
        ring_wins = (ring_games["win_total"] > 0).sum()
        ring_win_ratio = safe_div(ring_wins, ring_games_cnt) or 0.0

        if ring_games_cnt >= RING_MIN_GAMES_MEANINGFUL:
            if ring_win_ratio >= RING_WIN_RATIO_ALERT:
                score += 12
                reasons.append(
                    f"RING: Слишком высокий процент побед {fmt_pct(ring_win_ratio)} "
                    f"({ring_wins} из {ring_games_cnt} игр)."
                )

        # Signal 6: HU dominance
        hu_opponents = defaultdict(lambda: {"games": 0, "wins": 0, "total": 0.0})
        all_game_ids = set(ring_games["game_id"].unique())

        for gid in all_game_ids:
            g = games_df[games_df["game_id"] == gid]
            other_players = g[g["player_id"] != player_id]["player_id"].unique()

            if len(other_players) == 1:  # HU
                opp = int(other_players[0])
                p_row = g[g["player_id"] == player_id].iloc[0]
                hu_opponents[opp]["games"] += 1
                hu_opponents[opp]["total"] += float(p_row["win_total"] or 0.0)
                if float(p_row["win_total"] or 0.0) > 0:
                    hu_opponents[opp]["wins"] += 1

        if hu_opponents:
            top_hu = max(hu_opponents.items(), key=lambda x: x[1]["games"])
            top_opp, top_stats = top_hu
            if top_stats["games"] >= HU_DOMINANCE_MIN_GAMES:
                top_hu_ratio = safe_div(top_stats["wins"], top_stats["games"]) or 0.0
                if top_hu_ratio >= HU_ONE_SIDED_RATIO:
                    score += 18
                    reasons.append(
                        f"RING/HU: Доминирование над одним игроком (ID {top_opp}): "
                        f"{fmt_pct(top_hu_ratio)} побед ({top_stats['wins']}/{top_stats['games']})."
                    )

                if top_stats["total"] > 0:
                    bb_val = ring_games[ring_games["bb"].notna()]["bb"].iloc[0] if not ring_games[ring_games["bb"].notna()].empty else 1.0
                    bb_profit = top_stats["total"] / (bb_val or 1.0)
                    if bb_profit >= HU_PROFIT_VS_ONE_ALERT_BB:
                        score += 8
                        reasons.append(
                            f"RING/HU: Крупный профит против ID {top_opp}: "
                            f"{fmt_bb(bb_profit)} ({fmt_money(top_stats['total'])})."
                        )

        # Signal 7: Multiway dominance
        multiway_wins = 0
        multiway_games_cnt = 0
        for gid in all_game_ids:
            g = games_df[games_df["game_id"] == gid]
            if len(g) >= 3:  # multiway
                multiway_games_cnt += 1
                p_row = g[g["player_id"] == player_id]
                if not p_row.empty and float(p_row["win_total"].iloc[0] or 0.0) > 0:
                    multiway_wins += 1

        if multiway_games_cnt >= MULTIWAY_DOMINANCE_MIN_GAMES:
            mw_ratio = safe_div(multiway_wins, multiway_games_cnt) or 0.0
            if mw_ratio >= MULTIWAY_WIN_RATIO_ALERT:
                score += 6
                reasons.append(
                    f"RING/MULTIWAY: Высокий win rate в многоигровых сессиях "
                    f"{fmt_pct(mw_ratio)} ({multiway_wins}/{multiway_games_cnt})."
                )

        # Signal 8: Pair flow analysis (find worst partner)
        ring_game_ids = set(ring_games["game_id"].unique())
        all_opponents = set()
        for gid in ring_game_ids:
            g = games_df[games_df["game_id"] == gid]
            all_opponents.update(g[g["player_id"] != player_id]["player_id"].unique())

        worst_partner = None
        worst_score = 0

        for opp in all_opponents:
            stats = extract_pair_stats(games_df, player_id, opp, "RING")
            if stats["games_cnt"] < SHARED_SESSION_MIN:
                continue

            # One-sided ratio
            if stats["flow_ratio"] is not np.nan:
                one_sided = stats["flow_ratio"]
                if one_sided >= FLOW_CONSISTENCY_ALERT:
                    opp_score = 10
                    if one_sided >= 0.90:
                        opp_score += 5

                    # Check BB
                    if pd.notna(stats["bb"]) and stats["bb"] > 0:
                        shared_bb = stats["shared_bb"]
                        if shared_bb >= PAIR_FLOW_CRITICAL_BB:
                            opp_score += 15
                        elif shared_bb >= PAIR_FLOW_ALERT_BB:
                            opp_score += 10
                    else:
                        if stats["flow_total"] >= PAIR_FLOW_AMOUNT_ALERT:
                            opp_score += 8

                    if opp_score > worst_score:
                        worst_score = opp_score
                        worst_partner = opp

        if worst_partner is not None:
            score += worst_score
            worst_stats = extract_pair_stats(games_df, player_id, worst_partner, "RING")
            reason_text = (
                f"RING/FLOW: Подозрительный поток с игроком {worst_partner}: "
                f"{fmt_money(worst_stats['flow_total'])} ({fmt_bb(worst_stats['shared_bb'])}), "
                f"однонаправленность {fmt_pct(worst_stats['flow_ratio'])}."
            )
            reasons.append(reason_text)

    # === SECTION 3: Tournament Analysis ===
    tour_games = games_df[
        (games_df["player_id"] == player_id) & (games_df["game_type"] == "TOURNAMENT")
    ]

    if not tour_games.empty:
        tour_games_cnt = len(tour_games)
        tour_wins = (tour_games["win_total"] > 0).sum()
        tour_win_ratio = safe_div(tour_wins, tour_games_cnt) or 0.0

        # Signal 9: Tournament win ratio
        if tour_games_cnt >= 3:
            if tour_win_ratio >= TOUR_WIN_RATIO_ALERT:
                score += 8
                reasons.append(
                    f"TOUR: Высокий win rate {fmt_pct(tour_win_ratio)} "
                    f"({tour_wins} из {tour_games_cnt} турниров)."
                )

        # Signal 10: Single game extreme
        max_win = tour_games["win_total"].max()
        if max_win >= TOUR_EXTREME_SINGLE_GAME:
            score += 6
            reasons.append(f"TOUR: Разовый выигрыш {fmt_money(max_win)} (потенциально аномальный).")

    # === Final scoring ===
    score = int(max(0, min(100, score)))

    if score < T_APPROVE:
        decision = "APPROVE"
    elif score < T_WARN:
        decision = "FAST_CHECK"
    else:
        decision = "MANUAL_REVIEW"

    signals = {
        "db_j_total": j_total,
        "db_p_ring": p_ring,
        "db_p_mtt": p_mtt,
        "db_jackpot": p_jackpot,
        "db_events_delta": events_delta,
        "db_weeks": weeks,
        "db_top_week_share": top_week_share,
        "ring_games_cnt": len(ring_games) if not ring_games.empty else 0,
        "tour_games_cnt": len(tour_games) if not tour_games.empty else 0,
        "ring_win_ratio": ring_win_ratio if not ring_games.empty else np.nan,
        "tour_win_ratio": tour_win_ratio if not tour_games.empty else np.nan,
    }

    return score, decision, reasons, signals


# =========================
# STREAMLIT UI
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    st.title("🔍 " + APP_TITLE)

    # === SIDEBAR: File uploads ===
    with st.sidebar:
        st.header("📁 Загрузка данных")

        st.subheader("1️⃣ Database (Общий)")
        db_uploaded = st.file_uploader(
            "Загрузите Общий.csv (Database неделей)",
            type=["csv", "xlsx"],
            key="db_upload",
        )
        db_file = resolve_file(DB_KEY, db_uploaded)

        st.subheader("2️⃣ Games (История)")
        games_uploaded = st.file_uploader(
            "Загрузите Igry.csv (История игр)",
            type=["csv", "xlsx"],
            key="games_upload",
        )
        games_file = resolve_file(GAMES_KEY, games_uploaded)

        if db_file is not None:
            st.success(f"✅ DB загружена: {db_file.name}")
        if games_file is not None:
            st.success(f"✅ Games загружены: {games_file.name}")

        if st.button("🗑️ Очистить кэш", help="Удалить сохранённые данные и перезагрузить"):
            cache_clear(DB_KEY)
            cache_clear(GAMES_KEY)
            st.rerun()

    # === MAIN: Analysis ===
    if db_file is None or games_file is None:
        st.warning("⚠️ Пожалуйста, загрузите оба файла (Database и Games) для начала анализа.")
        return

    # Load and cache data
    with st.spinner("📂 Загружаю базу данных..."):
        try:
            db_df = load_db_any(db_file)
            st.session_state["db_df"] = db_df
        except Exception as e:
            st.error(f"❌ Ошибка загрузки DB: {e}")
            return

    with st.spinner("🎮 Парсю историю игр..."):
        try:
            games_df = parse_games_pppoker_export(games_file)
            indexes = build_games_indexes(games_df)
            st.session_state["games_df"] = games_df
            st.session_state["indexes"] = indexes
        except Exception as e:
            st.error(f"❌ Ошибка парсинга games: {e}")
            return

    # === Player query ===
    col1, col2 = st.columns([3, 1])
    with col1:
        player_id_input = st.text_input("🆔 ID игрока для проверки:", placeholder="Например: 11005036")
    with col2:
        check_btn = st.button("🔍 Проверить", use_container_width=True)

    if check_btn and player_id_input:
        try:
            player_id = int(player_id_input)
        except ValueError:
            st.error("❌ ID игрока должен быть числом.")
            return

        # Analyze
        db_df = st.session_state.get("db_df")
        games_df = st.session_state.get("games_df")
        indexes = st.session_state.get("indexes")

        if db_df is None or games_df is None:
            st.error("❌ Данные не загружены.")
            return

        with st.spinner(f"🔬 Анализирую игрока {player_id}..."):
            score, decision, reasons, signals = analyze_player(player_id, db_df, games_df, indexes)

        # === RESULT DISPLAY ===
        st.markdown("---")

        # Decision box
        if decision == "APPROVE":
            col_color = "green"
            icon = "✅"
            title = "МОЖНО ПРОВОДИТЬ ВЫПЛАТУ"
            explanation = "По всем доступным метрикам игрок чист. Нет сигналов перелива или сговора."
        elif decision == "FAST_CHECK":
            col_color = "orange"
            icon = "⚠️"
            title = "ТРЕБУЕТСЯ БЫСТРАЯ ПРОВЕРКА"
            explanation = "Обнаружены подозрительные сигналы. Рекомендуется проверить сессии и связи."
        else:
            col_color = "red"
            icon = "🚫"
            title = "ОТПРАВИТЬ В СБ НА РУЧНУЮ ПРОВЕРКУ"
            explanation = (
                "Высокий риск перелива фишек или сговора. "
                "Требуется детальный анализ карт и лог-файлов."
            )

        st.markdown(f"<h2 style='color: {col_color};'>{icon} {title}</h2>", unsafe_allow_html=True)
        st.markdown(f"**{explanation}**")
        st.markdown(f"**Риск-скор: {score}/100**")

        # Detailed signals
        with st.expander("📊 Детальные метрики", expanded=True):
            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                st.metric("Общий выигрыш (DB)", fmt_money(signals.get("db_j_total", 0.0)))
                st.metric("Ring Game выигрыш", fmt_money(signals.get("db_p_ring", 0.0)))

            with col_b:
                st.metric("MTT выигрыш", fmt_money(signals.get("db_p_mtt", 0.0)))
                st.metric("Джекпот", fmt_money(signals.get("db_jackpot", 0.0)))

            with col_c:
                st.metric("Ring Game статистика", signals.get("ring_games_cnt", 0), "игр")
                st.metric("Ring Win %", fmt_pct(signals.get("ring_win_ratio", np.nan)))

            with col_d:
                st.metric("Tournament статистика", signals.get("tour_games_cnt", 0), "турниров")
                st.metric("Tour Win %", fmt_pct(signals.get("tour_win_ratio", np.nan)))

        # Reasons & flags
        with st.expander("⚠️ Обнаруженные сигналы", expanded=True):
            if reasons:
                for i, reason in enumerate(reasons, 1):
                    st.markdown(f"**{i}.** {reason}")
            else:
                st.info("✅ Подозрительных сигналов не обнаружено.")

        # Recommendation
        st.markdown("---")
        st.subheader("📋 Рекомендация менеджеру:")
        if decision == "APPROVE":
            st.success(
                "Игрок может получить выплату без дополнительной проверки. "
                "По данным аналитики риск перелива минимален."
            )
        elif decision == "FAST_CHECK":
            st.warning(
                "Рекомендуется: 1) Проверить партнёрские связи игрока в последних сессиях; "
                "2) Проверить win rate в многоигровых столах; 3) Убедиться в логичности выигрышей."
            )
        else:
            st.error(
                "ОТПРАВИТЬ В DEPARTMENT БЕЗОПАСНОСТИ: Требуется ручной анализ всех каменей игры, "
                "проверка логов и связей с другими игроками. "
                "Возможны следующие нарушения: перелив фишек, скрытый сговор, искусственное раздутие выигрыша."
            )


if __name__ == "__main__":
    main()
