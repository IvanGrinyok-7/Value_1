# app.py
import io
import re
import json
import hashlib
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
APP_TITLE = "PPPoker | anti-fraud — проверка игрока (перелив / collusion)"

CACHE_DIR = Path(".pppoker_app_cache")
CACHE_DIR.mkdir(exist_ok=True)

DB_KEY = "db_file"
GAMES_KEY = "games_file"

# Decision thresholds (консервативно против false-negative)
T_APPROVE = 15
T_FAST_CHECK = 35

# Coverage: если мало данных по играм — не даём APPROVE при значимом профите
MIN_RING_GAMES_FOR_APPROVE = 3
MIN_RING_SESSIONS_FOR_APPROVE = 2

# Ring / pair thresholds (BB-aware)
PAIR_NET_ALERT_RING_BB = 20.0          # было 25
PAIR_NET_CRITICAL_RING_BB = 45.0       # было 50

PAIR_GROSS_ALERT_RING_BB = 100.0       # было 120
PAIR_GROSS_CRITICAL_RING_BB = 220.0    # было 250

PAIR_PARTNER_SHARE_ALERT = 0.55        # было 0.60
PAIR_ONE_SIDED_ALERT = 0.86            # было 0.88
PAIR_DIR_CONSIST_ALERT = 0.75          # было 0.78

# HU dominance
HU_DOMINANCE_MIN_HU = 2
HU_DOMINANCE_RATIO = 0.70              # было 0.75
HU_TOP_PARTNER_SHARE = 0.80

# Co-play concentration
COPLAY_TOP2_SHARE_SUSP = 0.85
MIN_SESSIONS_FOR_COPLAY = 6

# Tournament thresholds (currency)
PAIR_NET_ALERT_TOUR = 60.0
PAIR_GROSS_ALERT_TOUR = 150.0

# Extremes
SINGLE_GAME_WIN_ALERT_TOUR = 150.0

# =========================
# PPPoker export regex
# =========================
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)
TABLE_NAME_RE = re.compile(r"Название стола:\s*(.+?)\s*$", re.IGNORECASE)
START_END_RE = re.compile(r"Начало:\s*([0-9/: \s]+)\s+By.+?Окончание:\s*([0-9/: \s]+)", re.IGNORECASE)

RING_HINT_RE = re.compile(r"\bPPSR\b|PLO|OFC|NLH|Bomb Pot|Ante|3-1|HU\b|Heads", re.IGNORECASE)
TOUR_HINT_RE = re.compile(r"\bPPST\b|Бай-ин:|satellite|pko|mko\b|SNG\b|MTT\b", re.IGNORECASE)
STAKES_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)")

# =========================
# DB columns ("Общее.csv")
# =========================
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

EXTRA_PLAYER_WIN_COL_PREFIX = "Выигрыш игрока "

# =========================
# Persistent file cache (disk)
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
            {"name": name, "bytes": len(content), "saved_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
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
# Helpers
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

def safe_div(a, b):
    if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return a / b

def fmt_money(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NaN"
    return f"{float(v):.2f}"

def fmt_bb(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NaN"
    return f"{float(v):.1f} BB"

def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NaN"
    return f"{float(v) * 100:.0f}%"

def risk_decision(score: int) -> str:
    if score < T_APPROVE:
        return "APPROVE"
    if score < T_FAST_CHECK:
        return "FAST_CHECK"
    return "MANUAL_REVIEW"

def decision_badge(decision: str) -> tuple[str, str]:
    if decision == "APPROVE":
        return "✅ APPROVE", "green"
    if decision == "FAST_CHECK":
        return "🟠 FAST CHECK", "orange"
    return "🔴 MANUAL REVIEW", "red"

def manager_actions(decision: str) -> list[str]:
    if decision == "APPROVE":
        return [
            "Разрешить вывод.",
            "Если сумма крупная — выборочно проверить 1–2 последние сессии.",
        ]
    if decision == "FAST_CHECK":
        return [
            "Быстро проверить топ-партнёра(ов): shared sessions + net/gross + HU.",
            "Если подтверждается паттерн перелива — отправить в СБ.",
        ]
    return [
        "Отправить в СБ на ручную проверку.",
        "Проверить раздачи/карты по ключевым сессиям (особенно HU/3-max).",
    ]

# =========================
# DB loader
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
        COL_WEEK, COL_PLAYER_ID,
        COL_J_TOTAL, COL_PLAYER_WIN_TOTAL, COL_PLAYER_WIN_RING, COL_PLAYER_WIN_MTT,
        COL_CLUB_INCOME_TOTAL, COL_CLUB_COMMISSION
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"В DB не найдены обязательные колонки: {missing}")

    out = pd.DataFrame()
    out["_week"] = pd.to_numeric(df[COL_WEEK], errors="coerce").fillna(-1).astype(int)
    out["_player_id"] = pd.to_numeric(df[COL_PLAYER_ID], errors="coerce")
    out = out.dropna(subset=["_player_id"]).copy()
    out["_player_id"] = out["_player_id"].astype(int)

    # meta text
    for src, dst in [
        (COL_COUNTRY, "_country"),
        (COL_NICK, "_nick"),
        (COL_IGN, "_ign"),
        (COL_AGENT, "_agent"),
        (COL_SUPER_AGENT, "_super_agent"),
    ]:
        out[dst] = df.loc[out.index, src].astype(str).fillna("").str.strip() if src in df.columns else ""

    # meta numeric
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
    }
    for src, dst in base_num_cols.items():
        out[dst] = to_float_series(df.loc[out.index, src]) if src in df.columns else np.nan

    # optional extra wins
    extra_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(EXTRA_PLAYER_WIN_COL_PREFIX)]
    for c in extra_cols:
        if c in base_num_cols:
            continue
        norm = "_p_extra__" + re.sub(r"[^a-zA-Z0-9а-яА-Я_]+", "_", c.replace(EXTRA_PLAYER_WIN_COL_PREFIX, "").strip())
        out[norm] = to_float_series(df.loc[out.index, c])

    return out

def apply_weeks_filter(dbdf: pd.DataFrame, mode: str, last_n: int, week_from: int, week_to: int) -> pd.DataFrame:
    d = dbdf.copy()
    weeks = sorted([w for w in d["_week"].unique().tolist() if w >= 0])
    if mode == "ALL" or not weeks:
        return d
    if mode == "LAST_N":
        maxw = max(weeks)
        minw = maxw - max(0, int(last_n) - 1)
        return d[(d["_week"] >= minw) & (d["_week"] <= maxw)].copy()
    return d[(d["_week"] >= int(week_from)) & (d["_week"] <= int(week_to))].copy()

def db_summary_for_player(db_period: pd.DataFrame, player_id: int):
    d = db_period[db_period["_player_id"] == int(player_id)].copy()
    if d.empty:
        return None, None, None

    num_cols = [c for c in d.columns if c.startswith("_j_") or c.startswith("_p_") or c.startswith("_club_")]
    by_week = d.groupby("_week", as_index=False)[num_cols].sum(min_count=1).sort_values("_week")

    agg = by_week[num_cols].sum(numeric_only=True)
    total_j = float(agg.get("_j_total", 0.0) or 0.0)
    p_total = float(agg.get("_p_total", 0.0) or 0.0)
    p_ring = float(agg.get("_p_ring", 0.0) or 0.0)
    p_mtt = float(agg.get("_p_mtt", 0.0) or 0.0)
    comm_total = float(agg.get("_club_comm_total", 0.0) or 0.0)
    comm_ppsr = float(agg.get("_club_comm_ppsr", 0.0) or 0.0)
    comm_ppst = float(agg.get("_club_comm_ppst", 0.0) or 0.0)

    events_delta = total_j - p_total

    # week concentration
    if by_week.empty or abs(total_j) < 1e-9:
        top_week_share = np.nan
    else:
        top_week_j = float(by_week.sort_values("_j_total", ascending=False).iloc[0]["_j_total"] or 0.0)
        top_week_share = safe_div(top_week_j, total_j)

    meta = d.sort_values("_week").iloc[-1].to_dict()

    summary = {
        "week_cnt": int(by_week.shape[0]),
        "j_total": total_j,
        "p_total": p_total,
        "p_ring": p_ring,
        "p_mtt": p_mtt,
        "events_delta": float(events_delta),
        "comm_total": comm_total,
        "comm_ppsr": comm_ppsr,
        "comm_ppst": comm_ppst,
        "top_week_share": top_week_share,
        "meta": meta,
    }
    return summary, by_week, meta

# =========================
# Games parser
# =========================
def _split_semicolon(line: str) -> list[str]:
    return [p.strip().strip('"') for p in line.split(";")]

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

def parse_games_pppoker_export(file_obj) -> pd.DataFrame:
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

    def find_col(h: list[str], col: str):
        return h.index(col) if col in h else None

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

        # descriptor line
        if ("ID игрока" not in line) and ("Итог" not in line):
            is_desc = False
            if ("PPSR" in line) or ("PPST" in line) or TOUR_HINT_RE.search(line) or STAKES_RE.search(line):
                is_desc = True
            if is_desc and (not current["descriptor"] or ("PPSR" in line) or ("PPST" in line)):
                current["descriptor"] = line.strip()
                current["product"] = "PPSR" if "PPSR" in line else ("PPST" if "PPST" in line else "")
                current["game_type"] = _classify_game_type(current["descriptor"], current["table_name"])
                current["bb"] = _extract_bb_any(current["descriptor"], current["table_name"]) if current["game_type"] == "RING" else np.nan
                continue

        # table header
        if "ID игрока" in line:
            header = _split_semicolon(line)

            pid_i = find_col(header, "ID игрока")
            nick_i = find_col(header, "Ник")
            ign_i = find_col(header, "Игровое имя")
            hands_i = find_col(header, "Раздачи")
            fee_i = find_col(header, "Комиссия")

            # FIX: do NOT use `or` because 0 is falsy
            win_i = find_col(header, "Выигрыш")
            if win_i is None:
                win_i = find_col(header, "Выигрыш игрока")

            idx = {
                "player_id": pid_i,
                "nick": nick_i,
                "ign": ign_i,
                "hands": hands_i,
                "win": win_i,
                "fee": fee_i,
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
            "game_id": str(current["game_id"]),
            "game_type": current["game_type"],
            "product": current["product"],
            "table_name": current["table_name"],
            "descriptor": current["descriptor"],
            "bb": current["bb"],
            "start_time": current["start_time"],
            "end_time": current["end_time"],
            "player_id": pid,
            "nick": (parts[idx["nick"]] if idx.get("nick") is not None and idx["nick"] < len(parts) else ""),
            "ign": (parts[idx["ign"]] if idx.get("ign") is not None and idx["ign"] < len(parts) else ""),
            "hands": np.nan,
            "win_total": np.nan,
            "win_vs_opponents": np.nan,
            "fee": np.nan,
        }

        if idx.get("hands") is not None and idx["hands"] < len(parts):
            row["hands"] = to_float(parts[idx["hands"]])

        if idx.get("fee") is not None and idx["fee"] < len(parts):
            row["fee"] = to_float(parts[idx["fee"]])

        # Ring tables: win_total and win_vs_opponents are usually placed right after "Раздачи"
        if current["game_type"] == "RING":
            hidx = idx.get("hands")
            if hidx is not None and hidx + 2 < len(parts):
                row["win_total"] = to_float(parts[hidx + 1])
                row["win_vs_opponents"] = to_float(parts[hidx + 2])
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
        return pd.DataFrame(columns=[
            "game_id", "game_type", "product", "table_name", "descriptor", "bb",
            "start_time", "end_time", "player_id", "nick", "ign",
            "hands", "win_total", "win_vs_opponents", "fee"
        ])

    for c in ["bb", "hands", "win_total", "win_vs_opponents", "fee"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["player_id", "game_id"]).copy()
    df["player_id"] = df["player_id"].astype(int)
    df["game_id"] = df["game_id"].astype(str)
    df.loc[(df["game_type"] == "UNKNOWN") & df["bb"].notna(), "game_type"] = "RING"
    return df

# =========================
# Sessions + coplay
# =========================
def build_sessions_from_games(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["session_id", "game_type", "players", "players_n"])
    g = (
        games_df.groupby(["game_id", "game_type"])["player_id"]
        .apply(lambda x: sorted(set(int(v) for v in x.dropna().tolist())))
        .reset_index()
        .rename(columns={"game_id": "session_id", "player_id": "players"})
    )
    g["players_n"] = g["players"].apply(len)
    g = g[g["players_n"] >= 2].copy()
    return g

# =========================
# Pair flows (HU exact + multiway proportional)
# =========================
def build_pair_flows_fast(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["from_player", "to_player", "game_type", "amount", "amount_bb", "games_cnt"])

    df = games_df[["game_id", "game_type", "player_id", "bb", "win_total", "win_vs_opponents"]].copy()
    df["_flow_win"] = np.where(
        (df["game_type"] == "RING") & df["win_vs_opponents"].notna(),
        df["win_vs_opponents"],
        df["win_total"],
    )
    df = df[df["_flow_win"].notna()].copy()
    df["_flow_win"] = pd.to_numeric(df["_flow_win"], errors="coerce")
    df = df[df["_flow_win"].notna()].copy()

    flows_amt = {}
    flows_bb = {}
    games_cnt = {}

    for (gid, gtype), part in df.groupby(["game_id", "game_type"], sort=False):
        part = part[["player_id", "_flow_win", "bb"]]
        players = part["player_id"].dropna().astype(int).unique().tolist()
        nplayers = len(players)
        if nplayers < 2:
            continue

        winners = part[part["_flow_win"] > 0]
        losers = part[part["_flow_win"] < 0]
        if winners.empty or losers.empty:
            continue

        bb = float(part["bb"].max()) if gtype == "RING" else np.nan
        bb_ok = (bb > 0) if gtype == "RING" else False

        # HU exact transfer
        if nplayers == 2:
            wrow = winners.sort_values("_flow_win", ascending=False).iloc[0]
            lrow = losers.sort_values("_flow_win", ascending=True).iloc[0]
            wpid = int(wrow["player_id"])
            lpid = int(lrow["player_id"])
            amt = float(min(float(wrow["_flow_win"]), float(-lrow["_flow_win"])))
            if amt > 0:
                key = (lpid, wpid, gtype)
                flows_amt[key] = flows_amt.get(key, 0.0) + amt
                if bb_ok:
                    flows_bb[key] = flows_bb.get(key, 0.0) + float(amt / bb)
                games_cnt[key] = games_cnt.get(key, 0) + 1
            continue

        # multiway approx: allocate each loser's loss to winners proportional to their win
        total_pos = float(winners["_flow_win"].sum())
        if total_pos <= 0:
            continue

        win_pids = winners["player_id"].astype(int).to_numpy()
        win_vals = winners["_flow_win"].to_numpy(dtype=float)
        win_w = win_vals / total_pos

        for lpid, lwin in losers[["player_id", "_flow_win"]].itertuples(index=False):
            lpid = int(lpid)
            loss = float(-lwin)
            if loss <= 0:
                continue
            amts = loss * win_w
            for wpid, amt in zip(win_pids, amts):
                wpid = int(wpid)
                key = (lpid, wpid, gtype)
                flows_amt[key] = flows_amt.get(key, 0.0) + float(amt)
                if bb_ok:
                    flows_bb[key] = flows_bb.get(key, 0.0) + float(amt / bb)
                games_cnt[key] = games_cnt.get(key, 0) + 1

    if not flows_amt:
        return pd.DataFrame(columns=["from_player", "to_player", "game_type", "amount", "amount_bb", "games_cnt"])

    out = pd.DataFrame(
        [{
            "from_player": k[0],
            "to_player": k[1],
            "game_type": k[2],
            "amount": float(v),
            "amount_bb": float(flows_bb.get(k, np.nan)),
            "games_cnt": int(games_cnt[k]),
        } for k, v in flows_amt.items()]
    )
    out = out.groupby(["from_player", "to_player", "game_type"], as_index=False).agg(
        amount=("amount", "sum"),
        amount_bb=("amount_bb", "sum"),
        games_cnt=("games_cnt", "sum"),
    )
    return out

# =========================
# Indexes: coplay, pair stats, ring stats
# =========================
def build_indexes(games_df: pd.DataFrame, sessions_df: pd.DataFrame, flows_df: pd.DataFrame):
    idx = {}

    # --- coplay counters ---
    sessions_by_player = defaultdict(list)
    sessions_n = {}
    coplay_counter = defaultdict(lambda: defaultdict(int))
    coplay_counter_hu = defaultdict(lambda: defaultdict(int))
    coplay_sessions_cnt = defaultdict(int)
    coplay_hu_cnt = defaultdict(int)

    if not sessions_df.empty:
        for sid, gt, players, pn in sessions_df[["session_id", "game_type", "players", "players_n"]].itertuples(index=False):
            sid = str(sid)
            pn = int(pn)
            sessions_n[(gt, sid)] = pn
            pls = [int(x) for x in players]

            for p in pls:
                key = (gt, p)
                sessions_by_player[key].append(sid)
                coplay_sessions_cnt[key] += 1
                if pn == 2:
                    coplay_hu_cnt[key] += 1

            for i in range(len(pls)):
                pi = pls[i]
                di = coplay_counter[(gt, pi)]
                for j in range(len(pls)):
                    if i == j:
                        continue
                    di[pls[j]] += 1

            if pn == 2 and len(pls) == 2:
                a, b = pls[0], pls[1]
                coplay_counter_hu[(gt, a)][b] += 1
                coplay_counter_hu[(gt, b)][a] += 1

    idx["sessions_by_player"] = dict(sessions_by_player)
    idx["sessions_n"] = sessions_n
    idx["coplay_counter"] = {k: dict(v) for k, v in coplay_counter.items()}
    idx["coplay_counter_hu"] = {k: dict(v) for k, v in coplay_counter_hu.items()}
    idx["coplay_sessions_cnt"] = dict(coplay_sessions_cnt)
    idx["coplay_hu_cnt"] = dict(coplay_hu_cnt)

    # --- ring per-player stats (hands/fee/win in BB) ---
    ring_stats = {}
    if not games_df.empty:
        g = games_df[games_df["game_type"] == "RING"].copy()
        g = g[g["player_id"].notna()].copy()
        g["player_id"] = g["player_id"].astype(int)
        g["_hands"] = pd.to_numeric(g["hands"], errors="coerce").fillna(0.0)
        g["_fee"] = pd.to_numeric(g["fee"], errors="coerce").fillna(0.0)
        g["_w"] = np.where(g["win_vs_opponents"].notna(), g["win_vs_opponents"], g["win_total"])
        g["_w"] = pd.to_numeric(g["_w"], errors="coerce").fillna(0.0)
        g["_bb"] = pd.to_numeric(g["bb"], errors="coerce")

        g["_w_bb"] = np.where(g["_bb"].notna() & (g["_bb"] > 0), g["_w"] / g["_bb"], np.nan)
        g["_fee_bb"] = np.where(g["_bb"].notna() & (g["_bb"] > 0), g["_fee"] / g["_bb"], np.nan)

        # faster: manual groupby sums
        for pid, part in g.groupby("player_id", sort=False):
            hands_sum = float(part["_hands"].sum() or 0.0)
            fee_sum = float(part["_fee"].sum() or 0.0)
            w_sum = float(part["_w"].sum() or 0.0)
            w_bb_sum = float(np.nansum(part["_w_bb"].to_numpy(dtype=float)))
            fee_bb_sum = float(np.nansum(part["_fee_bb"].to_numpy(dtype=float)))
            games_cnt = int(part["game_id"].nunique())
            ring_stats[int(pid)] = {
                "ring_games": games_cnt,
                "hands": hands_sum,
                "fee": fee_sum,
                "win": w_sum,
                "win_bb": w_bb_sum,
                "fee_bb": fee_bb_sum,
                "bb_per_100hands": safe_div(w_bb_sum, (hands_sum / 100.0)) if hands_sum > 0 else np.nan,
            }

    idx["ring_stats"] = ring_stats

    # --- pair stats per player (multiple partners) ---
    pair_map = defaultdict(list)
    if not flows_df.empty:
        f = flows_df.copy()
        f["from_player"] = f["from_player"].astype(int)
        f["to_player"] = f["to_player"].astype(int)

        # Build undirected pair to compute net & gross between two players
        tmp = f[["game_type", "from_player", "to_player", "amount", "amount_bb", "games_cnt"]].copy()
        tmp["p"] = tmp[["from_player", "to_player"]].min(axis=1)
        tmp["q"] = tmp[["from_player", "to_player"]].max(axis=1)

        tmp["signed_to_q_amt"] = np.where(tmp["to_player"] == tmp["q"], tmp["amount"], -tmp["amount"])
        tmp["signed_to_q_bb"] = np.where(tmp["to_player"] == tmp["q"], tmp["amount_bb"], -tmp["amount_bb"])

        pair = tmp.groupby(["game_type", "p", "q"], as_index=False).agg(
            net_to_q=("signed_to_q_amt", "sum"),
            net_to_q_bb=("signed_to_q_bb", "sum"),
            gross=("amount", "sum"),
            gross_bb=("amount_bb", "sum"),
            games_cnt=("games_cnt", "sum"),
        )

        # Expand to both directions: for player_id we store net (positive means received)
        vq = pair.rename(columns={"q": "player_id", "p": "partner_id", "net_to_q": "net", "net_to_q_bb": "net_bb"})
        vp = pair.rename(columns={"p": "player_id", "q": "partner_id", "net_to_q": "net", "net_to_q_bb": "net_bb"})
        vp["net"] = -vp["net"]
        vp["net_bb"] = -vp["net_bb"]

        pairs = pd.concat([vq, vp], ignore_index=True)
        pairs["player_id"] = pairs["player_id"].astype(int)
        pairs["partner_id"] = pairs["partner_id"].astype(int)

        # partner share per player
        gross_tot = pairs.groupby(["game_type", "player_id"])["gross"].sum()
        gross_tot_bb = pairs.groupby(["game_type", "player_id"])["gross_bb"].sum()

        for gt, pid, partner, net, net_bb, gross, gross_bb, gcnt in pairs[[
            "game_type", "player_id", "partner_id", "net", "net_bb", "gross", "gross_bb", "games_cnt"
        ]].itertuples(index=False):
            pid = int(pid)
            partner = int(partner)
            gtot = float(gross_tot.get((gt, pid), 0.0) or 0.0)
            gtot_bb = float(gross_tot_bb.get((gt, pid), 0.0) or 0.0)

            pair_map[(gt, pid)].append({
                "partner": partner,
                "net": float(net),
                "net_bb": float(net_bb) if pd.notna(net_bb) else np.nan,
                "gross": float(gross),
                "gross_bb": float(gross_bb) if pd.notna(gross_bb) else np.nan,
                "games_cnt": int(gcnt),
                "partner_share": (float(gross) / gtot) if gtot > 0 else 0.0,
                "partner_share_bb": (float(gross_bb) / gtot_bb) if (gtot_bb > 0 and pd.notna(gross_bb)) else np.nan,
            })

        # sort partners by abs(net_bb) for ring or abs(net) for tour
        for key in list(pair_map.keys()):
            gt, pid = key
            lst = pair_map[key]

            def _rank(x):
                if gt == "RING" and pd.notna(x["net_bb"]):
                    return abs(float(x["net_bb"]))
                return abs(float(x["net"]))

            lst.sort(key=_rank, reverse=True)
            pair_map[key] = lst

    idx["pair_map"] = dict(pair_map)

    return idx

def coplay_features(target_id: int, idx: dict, game_type: str) -> dict:
    key = (game_type, int(target_id))
    sessions_cnt = int(idx.get("coplay_sessions_cnt", {}).get(key, 0))
    hu_sessions = int(idx.get("coplay_hu_cnt", {}).get(key, 0))

    counter = idx.get("coplay_counter", {}).get(key, {})
    partners = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    top2_share = float((partners[0][1] + partners[1][1]) / sessions_cnt) if sessions_cnt and len(partners) >= 2 else 0.0

    huc = idx.get("coplay_counter_hu", {}).get(key, {})
    hups = sorted(huc.items(), key=lambda x: x[1], reverse=True)
    top_hu_partner = int(hups[0][0]) if hups else None
    top_hu_share = float(hups[0][1] / max(1, hu_sessions)) if hups else 0.0

    return {
        "sessions_cnt": sessions_cnt,
        "hu_sessions": hu_sessions,
        "unique_opponents": len(partners),
        "top2_share": top2_share,
        "top_hu_partner": top_hu_partner,
        "top_hu_share": top_hu_share,
    }

def shared_sessions(pid_a: int, pid_b: int, idx: dict, game_type: str, limit: int = 30) -> list[str]:
    a = idx.get("sessions_by_player", {}).get((game_type, int(pid_a)), [])
    b = idx.get("sessions_by_player", {}).get((game_type, int(pid_b)), [])
    if not a or not b:
        return []
    sa = set(a) if len(a) < len(b) else set(b)
    sb = b if len(a) < len(b) else a
    shared = [x for x in sb if x in sa]
    return shared[:limit]

# =========================
# Risk scoring (explainable, multi-partner)
# =========================
def score_player(player_id: int, db_sum: dict, idx: dict) -> tuple[int, str, list[str], dict]:
    pid = int(player_id)
    reasons = []
    score = 0

    ring_cov = idx.get("ring_stats", {}).get(pid, {})
    ring_games = int(ring_cov.get("ring_games", 0) or 0)

    cop_ring = coplay_features(pid, idx, "RING")
    pairs_ring = idx.get("pair_map", {}).get(("RING", pid), [])
    pairs_tour = idx.get("pair_map", {}).get(("TOURNAMENT", pid), [])

    # ---- DB anomalies (мягкие) ----
    j_total = float(db_sum.get("j_total", 0.0) or 0.0)
    events_delta = float(db_sum.get("events_delta", 0.0) or 0.0)
    week_cnt = int(db_sum.get("week_cnt", 0) or 0)
    top_week_share = db_sum.get("top_week_share", np.nan)

    if abs(j_total) >= 800:
        score += 8
        reasons.append(f"DB: большой общий результат {fmt_money(j_total)}.")
    elif abs(j_total) >= 300:
        score += 4
        reasons.append(f"DB: заметный общий результат {fmt_money(j_total)}.")

    if abs(events_delta) >= max(80.0, 0.35 * max(1.0, abs(j_total))):
        score += 6
        reasons.append(f"DB: разница 'Общий выигрыш + События' vs 'Выигрыш игрока общий' = {fmt_money(events_delta)}.")

    if week_cnt >= 3 and pd.notna(top_week_share) and float(top_week_share) >= 0.80 and abs(j_total) >= 300:
        score += 5
        reasons.append(f"DB: концентрация результата в одной неделе {fmt_pct(float(top_week_share))}.")

    # ---- Co-play ----
    if cop_ring["sessions_cnt"] >= MIN_SESSIONS_FOR_COPLAY and cop_ring["top2_share"] >= COPLAY_TOP2_SHARE_SUSP:
        score += 8
        reasons.append(
            f"GAMES: высокая концентрация co-play (топ-2 оппонента дают {fmt_pct(cop_ring['top2_share'])} сессий)."
        )

    hu_ratio = float(cop_ring["hu_sessions"] / max(1, cop_ring["sessions_cnt"]))
    if cop_ring["hu_sessions"] >= HU_DOMINANCE_MIN_HU and hu_ratio >= HU_DOMINANCE_RATIO:
        score += 12
        reasons.append(f"GAMES: доминация HU (HU {cop_ring['hu_sessions']} из {cop_ring['sessions_cnt']} = {fmt_pct(hu_ratio)}).")

    if cop_ring["top_hu_partner"] is not None and cop_ring["hu_sessions"] >= HU_DOMINANCE_MIN_HU and cop_ring["top_hu_share"] >= HU_TOP_PARTNER_SHARE:
        score += 10
        reasons.append(
            f"GAMES: HU почти всегда с одним партнёром {cop_ring['top_hu_partner']} (share {fmt_pct(cop_ring['top_hu_share'])})."
        )

    # ---- Pair risk evaluation: check top 3 partners (Ring) ----
    # Берём максимум риска по нескольким партнёрам, чтобы не пропускать multi-partner перелив.
    strongest_partner = None
    strongest_partner_points = 0

    for p in pairs_ring[:3]:
        partner = int(p["partner"])
        net_bb = p.get("net_bb", np.nan)
        gross_bb = p.get("gross_bb", np.nan)

        # partner_share: prefer BB-share when available
        pshare = p.get("partner_share_bb", np.nan)
        if pd.isna(pshare):
            pshare = float(p.get("partner_share", 0.0) or 0.0)

        shared = shared_sessions(pid, partner, idx, "RING", limit=50)
        shared_cnt = len(shared)
        hu_in_shared = 0
        for sid in shared:
            if idx.get("sessions_n", {}).get(("RING", sid), 0) == 2:
                hu_in_shared += 1
        hu_share = float(hu_in_shared / max(1, shared_cnt))

        local_points = 0
        local_reasons = []

        # Strong dumping: big net to/from one partner in BB
        if pd.notna(net_bb) and abs(float(net_bb)) >= PAIR_NET_CRITICAL_RING_BB:
            local_points += 40
            local_reasons.append(f"RING: крупный net-flow с партнёром {partner}: {fmt_bb(net_bb)} (shared {shared_cnt}).")
        elif pd.notna(net_bb) and abs(float(net_bb)) >= PAIR_NET_ALERT_RING_BB and pshare >= PAIR_PARTNER_SHARE_ALERT:
            local_points += 22
            local_reasons.append(
                f"RING: net-flow с партнёром {partner}: {fmt_bb(net_bb)} при доле оборота {fmt_pct(pshare)} (shared {shared_cnt})."
            )

        # Big turnover even if net not huge: collusion/softplay pattern proxy
        if pd.notna(gross_bb) and float(gross_bb) >= PAIR_GROSS_CRITICAL_RING_BB and pshare >= PAIR_PARTNER_SHARE_ALERT and shared_cnt >= 2:
            local_points += 20
            local_reasons.append(f"RING: высокий gross-turnover с {partner}: {fmt_bb(gross_bb)} и доля {fmt_pct(pshare)}.")
        elif pd.notna(gross_bb) and float(gross_bb) >= PAIR_GROSS_ALERT_RING_BB and pshare >= PAIR_PARTNER_SHARE_ALERT and shared_cnt >= 1:
            local_points += 10
            local_reasons.append(f"RING: заметный gross-turnover с {partner}: {fmt_bb(gross_bb)} и доля {fmt_pct(pshare)}.")

        # HU strengthening: even 1–2 sessions can be enough if HU share is high
        if shared_cnt >= 1 and hu_share >= 0.70 and pshare >= PAIR_PARTNER_SHARE_ALERT:
            local_points += 10
            local_reasons.append(f"RING: много HU в общих сессиях с {partner}: HU-share {fmt_pct(hu_share)}.")

        if local_points > strongest_partner_points:
            strongest_partner_points = local_points
            strongest_partner = partner

        if local_points > 0:
            # accumulate but cap per-partner to avoid score explosion
            score += min(30, local_points)
            reasons.extend(local_reasons)

    # ---- Tournament pairs (weaker evidence) ----
    for p in pairs_tour[:2]:
        partner = int(p["partner"])
        net = float(p.get("net", 0.0) or 0.0)
        gross = float(p.get("gross", 0.0) or 0.0)
        pshare = float(p.get("partner_share", 0.0) or 0.0)
        shared_cnt = len(shared_sessions(pid, partner, idx, "TOURNAMENT", limit=50))

        if abs(net) >= PAIR_NET_ALERT_TOUR and pshare >= 0.60:
            score += 8
            reasons.append(f"TOUR: net-flow {fmt_money(net)} с {partner} при доле {fmt_pct(pshare)} (shared {shared_cnt}).")
        if gross >= PAIR_GROSS_ALERT_TOUR and pshare >= 0.60 and shared_cnt >= 1:
            score += 6
            reasons.append(f"TOUR: gross-turnover {fmt_money(gross)} с {partner} при доле {fmt_pct(pshare)} (shared {shared_cnt}).")

    # ---- Coverage guard (против false-negative) ----
    # Если по играм почти ничего нет — нельзя уверенно approve при заметном результате.
    if ring_games < MIN_RING_GAMES_FOR_APPROVE and abs(j_total) >= 200:
        score = max(score, T_APPROVE)  # минимум FAST_CHECK
        reasons.append(f"COVERAGE: мало Ring данных (ring_games={ring_games}), при этом DB результат {fmt_money(j_total)}.")

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)

    signals = {
        "score": score,
        "decision": decision,
        "player_id": pid,
        "ring_games": ring_games,
        "cop_ring": cop_ring,
        "strongest_partner": strongest_partner,
        "db": {
            "j_total": j_total,
            "events_delta": events_delta,
            "week_cnt": week_cnt,
            "top_week_share": top_week_share,
        },
    }
    return score, decision, reasons, signals

def build_security_message(db_sum: dict, by_week: pd.DataFrame, signals: dict, reasons: list[str]) -> str:
    meta = db_sum.get("meta", {}) if db_sum else {}
    pid = int(signals.get("player_id", 0))
    decision = signals.get("decision", "")
    score = int(signals.get("score", 0))

    msg = []
    msg.append("ANTI-FRAUD CHECK (PPPoker)")
    msg.append(f"Player ID: {pid}")
    msg.append(f"Decision: {decision} | Risk score: {score}/100")
    msg.append("")
    msg.append(f"Nick: {meta.get('_nick', '')} | IGN: {meta.get('_ign', '')} | Country: {meta.get('_country', '')}")
    msg.append(f"Agent: {meta.get('_agent', '')} | AgentID: {meta.get('_agent_id', '')}")
    msg.append("")
    msg.append("DB summary:")
    msg.append(f"- J total: {fmt_money(signals['db']['j_total'])}")
    msg.append(f"- Events delta (J - player total): {fmt_money(signals['db']['events_delta'])}")
    if pd.notna(signals["db"]["top_week_share"]):
        msg.append(f"- Top week share: {fmt_pct(float(signals['db']['top_week_share']))}")
    msg.append("")
    msg.append("Games summary:")
    msg.append(f"- Ring games in export: {signals.get('ring_games', 0)}")
    cr = signals.get("cop_ring", {})
    msg.append(f"- Ring sessions: {cr.get('sessions_cnt', 0)} | HU sessions: {cr.get('hu_sessions', 0)} | Top2 co-play share: {fmt_pct(cr.get('top2_share', 0.0))}")
    if cr.get("top_hu_partner") is not None:
        msg.append(f"- Top HU partner: {cr.get('top_hu_partner')} | HU share: {fmt_pct(cr.get('top_hu_share', 0.0))}")
    if signals.get("strongest_partner") is not None:
        msg.append(f"- Strongest partner by flows: {signals.get('strongest_partner')}")

    msg.append("")
    msg.append("Reasons (top):")
    for r in reasons[:25]:
        msg.append(f"- {r}")

    if by_week is not None and not by_week.empty:
        msg.append("")
        msg.append("DB by week (last rows):")
        tail = by_week.sort_values("_week").tail(8)
        for _, row in tail.iterrows():
            w = int(row["_week"])
            jt = fmt_money(row.get("_j_total", 0.0))
            pr = fmt_money(row.get("_p_ring", 0.0))
            pm = fmt_money(row.get("_p_mtt", 0.0))
            ct = fmt_money(row.get("_club_comm_total", 0.0))
            msg.append(f"- week {w}: J={jt}, Ring={pr}, MTT={pm}, Comm={ct}")

    return "\n".join(msg)

# =========================
# Cached loaders
# =========================
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:12]

@st.cache_data(show_spinner=False)
def cached_load_db_multi(contents: tuple[bytes, ...], names: tuple[str, ...]) -> pd.DataFrame:
    dfs = []
    for c, n in zip(contents, names):
        dfs.append(load_db_any(BytesFile(c, n)))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # drop exact duplicates by (week, player)
    df = df.sort_values(["_week"]).drop_duplicates(subset=["_week", "_player_id"], keep="last")
    return df

@st.cache_data(show_spinner=True)
def cached_games_bundle_multi(contents: tuple[bytes, ...], names: tuple[str, ...]):
    all_games = []
    for c, n in zip(contents, names):
        part = parse_games_pppoker_export(BytesFile(c, n))
        if not part.empty:
            all_games.append(part)
    if not all_games:
        games_df = pd.DataFrame(columns=[
            "game_id", "game_type", "product", "table_name", "descriptor", "bb",
            "start_time", "end_time", "player_id", "nick", "ign",
            "hands", "win_total", "win_vs_opponents", "fee"
        ])
    else:
        games_df = pd.concat(all_games, ignore_index=True)

    sessions_df = build_sessions_from_games(games_df)
    flows_df = build_pair_flows_fast(games_df)
    idx = build_indexes(games_df, sessions_df, flows_df)
    return games_df, sessions_df, flows_df, idx

# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Данные")
    db_uploaded = st.file_uploader("Загрузить Общее (CSV/XLSX). Можно несколько файлов (недели).", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    games_uploaded = st.file_uploader("Загрузить Игры (CSV/TXT-выгрузка PPPoker). Можно несколько файлов.", type=["csv", "txt"], accept_multiple_files=True)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Сохранить как базу", use_container_width=True):
            # сохраняем только первый файл (как у вас), остальные будут в памяти через st.cache_data в рамках сессии
            if db_uploaded:
                cache_save_uploaded(DB_KEY, db_uploaded[0])
            if games_uploaded:
                cache_save_uploaded(GAMES_KEY, games_uploaded[0])
            st.success("Сохранено.")
    with colB:
        if st.button("Очистить базу", use_container_width=True):
            cache_clear(DB_KEY)
            cache_clear(GAMES_KEY)
            st.success("Очищено.")

    st.divider()
    st.subheader("Период проверки (DB)")
    week_mode = st.selectbox("Режим", ["ALL", "LAST_N", "RANGE"], index=1)
    last_n = st.number_input("Последние N недель", min_value=1, max_value=52, value=4, step=1)
    week_from = st.number_input("Неделя от", min_value=0, max_value=999, value=1, step=1)
    week_to = st.number_input("Неделя до", min_value=0, max_value=999, value=1, step=1)

# resolve persisted baseline files
db_base = resolve_file(DB_KEY, None)
games_base = resolve_file(GAMES_KEY, None)

# Build DB contents list
db_contents = []
db_names = []
if db_base is not None:
    db_contents.append(db_base.getvalue())
    db_names.append(getattr(db_base, "name", "db_base"))
if db_uploaded:
    for f in db_uploaded:
        db_contents.append(f.getvalue())
        db_names.append(getattr(f, "name", "db_upload"))

if not db_contents:
     st.info("Загрузите файл 'Общее' (CSV/XLSX). После этого можно (опционально) загрузить 'Игры'.")
    st.stop()

dbdf = cached_load_db_multi(tuple(db_contents), tuple(db_names))
if dbdf.empty:
    st.error("DB пустая или не распарсилась.")
    st.stop()

valid_weeks = sorted([int(w) for w in dbdf["_week"].unique().tolist() if int(w) >= 0])
wmin = min(valid_weeks) if valid_weeks else 0
wmax = max(valid_weeks) if valid_weeks else 0

# Если RANGE выбран — подставим адекватные дефолты
if week_mode == "RANGE":
    if week_from == 1 and week_to == 1 and wmax >= wmin:
        week_from = wmin
        week_to = wmax

db_period = apply_weeks_filter(dbdf, week_mode, int(last_n), int(week_from), int(week_to))

# -------------------------
# LOAD GAMES (optional)
# -------------------------
games_contents = []
games_names = []

if games_base is not None:
    games_contents.append(games_base.getvalue())
    games_names.append(getattr(games_base, "name", "games_base"))

if games_uploaded:
    for f in games_uploaded:
        games_contents.append(f.getvalue())
        games_names.append(getattr(f, "name", "games_upload"))

if games_contents:
    games_df, sessions_df, flows_df, idx = cached_games_bundle_multi(tuple(games_contents), tuple(games_names))
else:
    games_df = pd.DataFrame()
    sessions_df = pd.DataFrame()
    flows_df = pd.DataFrame()
    idx = {
        "sessions_by_player": {},
        "sessions_n": {},
        "coplay_counter": {},
        "coplay_counter_hu": {},
        "coplay_sessions_cnt": {},
        "coplay_hu_cnt": {},
        "ring_stats": {},
        "pair_map": {},
    }

# -------------------------
# Header metrics
# -------------------------
m1, m2, m3, m4 = st.columns(4, gap="small")
m1.metric("DB rows", f"{len(dbdf)}", border=True)
m2.metric("Players in DB", f"{dbdf['_player_id'].nunique()}", border=True)
m3.metric("Games rows", f"{len(games_df)}", border=True)
m4.metric("Flows pairs", f"{len(flows_df)}", border=True)

if not games_contents:
    st.warning("Файл 'Игры' не загружен: детект перелива будет ограничен, решения будут более консервативными (чаще FAST_CHECK).")

st.divider()

# -------------------------
# Cached top risk
# -------------------------
@st.cache_data(show_spinner=True)
def cached_top_risk(db_period_in: pd.DataFrame, idx_in: dict, top_n: int) -> pd.DataFrame:
    players = sorted([int(x) for x in db_period_in["_player_id"].unique().tolist()])
    res = []

    for pid in players:
        db_sum, by_week, _meta = db_summary_for_player(db_period_in, pid)
        if db_sum is None:
            continue
        score, decision, reasons, signals = score_player(pid, db_sum, idx_in)

        res.append({
            "player_id": int(pid),
            "score": int(score),
            "decision": decision,
            "j_total": float(signals["db"]["j_total"]),
            "events_delta": float(signals["db"]["events_delta"]),
            "weeks": int(signals["db"]["week_cnt"]),
            "ring_games": int(signals.get("ring_games", 0) or 0),
            "ring_sessions": int((signals.get("cop_ring", {}) or {}).get("sessions_cnt", 0)),
            "ring_hu_sessions": int((signals.get("cop_ring", {}) or {}).get("hu_sessions", 0)),
            "strongest_partner": signals.get("strongest_partner", None),
            "top_reason": (reasons[0] if reasons else ""),
        })

    if not res:
        return pd.DataFrame()

    out = pd.DataFrame(res)
    out = out.sort_values(["score", "j_total"], ascending=[False, False]).head(int(top_n)).copy()
    return out

# -------------------------
# TABS
# -------------------------
tab_check, tab_top, tab_diag = st.tabs(["Проверка по ID", "Топ риск", "Диагностика"])

with tab_check:
    left, right = st.columns([1.0, 1.6], gap="large")

    with left:
        st.subheader("Проверка игрока")
        default_id = int(db_period["_player_id"].iloc[0]) if len(db_period) else int(dbdf["_player_id"].iloc[0])
        pid = st.number_input("ID игрока", min_value=0, value=int(default_id), step=1)
        run = st.button("Проверить", type="primary", use_container_width=True)

        st.caption("Логика решения: APPROVE / FAST_CHECK / MANUAL_REVIEW (в СБ).")

    with right:
        if not run:
            st.info("Введите ID и нажмите «Проверить».")
            st.stop()

        db_sum, by_week, meta = db_summary_for_player(db_period, int(pid))
        if db_sum is None:
            st.error("Игрок не найден в DB за выбранный период.")
            st.stop()

        score, decision, reasons, signals = score_player(int(pid), db_sum, idx)
        badge_text, badge_color = decision_badge(decision)

        cA, cB, cC, cD = st.columns([1.4, 1.0, 1.0, 1.0], gap="small")
        cA.metric("Решение", badge_text, border=True)
        cB.metric("Risk score", f"{score}/100", border=True)
        cC.metric("Недель в периоде", f"{signals['db']['week_cnt']}", border=True)
        cD.metric("Ring games", f"{signals.get('ring_games', 0)}", border=True)

        if badge_color == "green":
            st.success("Риск по данным низкий — вывод можно разрешить.")
        elif badge_color == "orange":
            st.warning("Есть подозрительные сигналы — рекомендован быстрый чек (или СБ при сомнениях).")
        else:
            st.error("Высокий риск — отправить в СБ на ручную проверку.")

        st.subheader("Действия менеджера")
        for a in manager_actions(decision):
            st.write(f"- {a}")

        st.divider()

        st.subheader("Ключевые цифры (DB)")
        k1, k2, k3, k4 = st.columns(4, gap="small")
        k1.metric(COL_J_TOTAL, fmt_money(signals["db"]["j_total"]), border=True)
        k2.metric(f"{COL_J_TOTAL} - {COL_PLAYER_WIN_TOTAL}", fmt_money(signals["db"]["events_delta"]), border=True)
        if pd.notna(signals["db"]["top_week_share"]):
            k3.metric("Top week share", fmt_pct(float(signals["db"]["top_week_share"])), border=True)
        else:
            k3.metric("Top week share", "NaN", border=True)
        k4.metric("Ring games (export)", f"{signals.get('ring_games', 0)}", border=True)

        st.subheader("DB по неделям")
        st.dataframe(by_week.sort_values("_week", ascending=False), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Партнёры/переводы (Ring)")
        pairs = idx.get("pair_map", {}).get(("RING", int(pid)), [])
        if not pairs:
            st.info("По 'Игры' не найдено парных потоков (flows) для Ring.")
        else:
            rows = []
            for p in pairs[:10]:
                partner = int(p["partner"])
                shared = shared_sessions(int(pid), partner, idx, "RING", limit=50)
                shared_cnt = len(shared)

                net_bb = p.get("net_bb", np.nan)
                gross_bb = p.get("gross_bb", np.nan)

                # prefer BB-share; fallback to currency share
                pshare = p.get("partner_share_bb", np.nan)
                if pd.isna(pshare):
                    pshare = float(p.get("partner_share", 0.0) or 0.0)

                direction = "получает от партнёра" if (pd.notna(net_bb) and float(net_bb) > 0) else "отдаёт партнёру"
                if pd.isna(net_bb):
                    # если BB нет — по currency net
                    direction = "получает от партнёра" if float(p.get("net", 0.0) or 0.0) > 0 else "отдаёт партнёру"

                rows.append({
                    "partner_id": partner,
                    "direction": direction,
                    "net_bb": float(net_bb) if pd.notna(net_bb) else np.nan,
                    "gross_bb": float(gross_bb) if pd.notna(gross_bb) else np.nan,
                    "partner_share": float(pshare),
                    "pair_games_cnt": int(p.get("games_cnt", 0) or 0),
                    "shared_sessions_cnt": int(shared_cnt),
                    "shared_sessions_preview": ", ".join(shared[:8]),
                })

            show = pd.DataFrame(rows)
            if not show.empty:
                show["partner_share"] = show["partner_share"].apply(lambda x: f"{x*100:.0f}%")
                show["net_bb"] = show["net_bb"].apply(lambda x: fmt_bb(x) if pd.notna(x) else "NaN")
                show["gross_bb"] = show["gross_bb"].apply(lambda x: fmt_bb(x) if pd.notna(x) else "NaN")

            st.dataframe(show, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Причины (top)")
        for r in reasons[:30]:
            st.write(f"- {r}")

        st.divider()

        st.subheader("Сообщение для СБ")
        sec_text = build_security_message(db_sum, by_week, signals, reasons)
        st.textarea("Скопировать/отправить", value=sec_text, height=260)
        st.download_button(
            "Скачать .txt для СБ",
            data=sec_text.encode("utf-8"),
            file_name=f"SB_check_{int(pid)}.txt",
            mime="text/plain",
            use_container_width=True,
        )

with tab_top:
    st.subheader("Топ подозрительных за период")
    colA, colB = st.columns([1.0, 1.0], gap="small")
    with colA:
        top_n = st.number_input("Top N", min_value=10, max_value=300, value=50, step=10)
    with colB:
        build = st.button("Построить список", type="primary", use_container_width=True)

    if not build:
        st.info("Нажмите «Построить список».")
        st.stop()

    topdf = cached_top_risk(db_period, idx, int(top_n))
    if topdf.empty:
        st.info("Не удалось построить список (возможно, пустой период).")
        st.stop()

    st.dataframe(topdf, use_container_width=True, hide_index=True)

    csv_bytes = topdf.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Скачать CSV",
        data=csv_bytes,
        file_name="top_risk_players.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tab_diag:
    st.subheader("Диагностика загрузки")
    st.write("- DB weeks:", f"{wmin} .. {wmax}")
    st.write("- Selected period rows:", len(db_period))
    st.write("- Games loaded:", bool(games_contents))
    if games_contents:
        st.write("- Sessions:", len(sessions_df))
        st.write("- Flows:", len(flows_df))

    st.subheader("Сигналы по выборке (для контроля)")
    # покажем 5 первых игроков с их ring_games (если есть)
    sample_players = sorted([int(x) for x in db_period["_player_id"].unique().tolist()])[:5]
    diag_rows = []
    for sp in sample_players:
        rs = idx.get("ring_stats", {}).get(int(sp), {})
        diag_rows.append({
            "player_id": int(sp),
            "ring_games": int(rs.get("ring_games", 0) or 0),
            "hands": float(rs.get("hands", 0.0) or 0.0),
            "win_bb": float(rs.get("win_bb", 0.0) or 0.0),
            "bb_per_100hands": float(rs.get("bb_per_100hands", np.nan)),
        })
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
