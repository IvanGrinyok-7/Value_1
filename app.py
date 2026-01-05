import io
import re
import json
import hashlib
import datetime as dt
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
APP_TITLE = "PPPoker | риск/anti-fraud — проверка игроков (chip dumping / collusion)"
CACHE_DIR = Path(".pppoker_app_cache")
CACHE_DIR.mkdir(exist_ok=True)

DB_KEY = "db_file"
GAMES_KEY = "games_file"

# --- RATING THRESHOLDS ---
T_APPROVE = 25
T_FAST_CHECK = 55

# --- CO-PLAY / COLLUSION ---
MIN_SESSIONS_FOR_COPLAY = 1
COPLAY_TOP2_SHARE_SUSP = 0.85

HU_DOMINANCE_MIN_HU = 2
HU_DOMINANCE_RATIO = 0.75

# --- FLOWS (BB-aware) ---
PAIR_NET_ALERT_RING_BB = 25.0
PAIR_NET_CRITICAL_RING_BB = 50.0

PAIR_GROSS_ALERT_RING_BB = 120.0
PAIR_GROSS_CRITICAL_RING_BB = 250.0

PAIR_ONE_SIDED_ALERT = 0.88
PAIR_DIR_CONSIST_ALERT = 0.78
PAIR_PARTNER_SHARE_ALERT = 0.60

PAIR_MIN_SHARED_SESSIONS_STRONG = 2
PAIR_MIN_PAIR_GAMES_STRONG = 3

# NEW: single-session big transfer should still alert
PAIR_SINGLE_SESSION_NET_ALERT_RING_BB = 30.0     # if shared_sessions==1 but net is huge
PAIR_SINGLE_SESSION_NET_ALERT_TOUR = 80.0        # currency

# Tournaments thresholds in currency
PAIR_NET_ALERT_TOUR = 60.0
PAIR_GROSS_ALERT_TOUR = 150.0

# Regex PPPoker export
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)
TABLE_NAME_RE = re.compile(r"Название стола:\s*(.+?)\s*$", re.IGNORECASE)
START_END_RE = re.compile(r"Начало:\s*([0-9/: \s]+)\s+By.+?Окончание:\s*([0-9/: \s]+)", re.IGNORECASE)

RING_HINT_RE = re.compile(r"\bPPSR\b|PLO|OFC|NLH|Bomb Pot|Ante|3-1|HU\b|Heads", re.IGNORECASE)
TOUR_HINT_RE = re.compile(r"\bPPST\b|Бай-ин:|satellite|pko|mko\b|SNG\b|MTT\b", re.IGNORECASE)
STAKES_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)")

# DB columns (sheet "Общий")
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
# Persistent file cache (как в исходнике)
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
def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


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


def fmt_money(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NaN"
    return f"{float(v):.2f}"


def fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return f"{float(x) * 100:.0f}%"


def fmt_bb(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return f"{float(x):.1f} BB"


def risk_decision(score: int) -> str:
    if score < T_APPROVE:
        return "APPROVE"
    if score < T_FAST_CHECK:
        return "FAST_CHECK"
    return "MANUAL_REVIEW"


# =========================
# DB LOAD (CSV/XLSX)
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

    for src, dst in [
        (COL_COUNTRY, "_country"),
        (COL_NICK, "_nick"),
        (COL_IGN, "_ign"),
        (COL_AGENT, "_agent"),
        (COL_SUPER_AGENT, "_super_agent"),
    ]:
        out[dst] = df.loc[out.index, src].astype(str).fillna("").str.strip() if src in df.columns else ""

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

    out["_src_sha1"] = _sha1_bytes(content)
    out["_src_name"] = getattr(file_obj, "name", "db")
    return out


# =========================
# GAMES PARSER (+BB)
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

        # descriptor detection
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
                current["bb"] = _extract_bb_any(current["descriptor"], current["table_name"]) if current["game_type"] == "RING" else np.nan
                continue

        if "ID игрока" in line:
            header = _split_semicolon(line)

            def find(col):
                try:
                    return header.index(col)
                except ValueError:
                    return None

            # FIX: нельзя делать `a or b` для индекса (0 считается False)
            win_idx = find("Выигрыш")
            if win_idx is None:
                win_idx = find("Выигрыш игрока")

            idx = {
                "player_id": find("ID игрока"),
                "nick": find("Ник"),
                "ign": find("Игровое имя"),
                "hands": find("Раздачи"),
                "win": win_idx,
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

        if current["game_type"] == "RING":
            # Ring export has '...;Раздачи;Выигрыш игрока;...'
            # and in rows we typically have: hands, win_total, win_vs_opponents right after hands.
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
            "game_id", "game_type", "product", "table_name", "descriptor", "bb", "start_time", "end_time",
            "player_id", "nick", "ign", "hands", "win_total", "win_vs_opponents", "fee"
        ])

    for c in ["bb", "hands", "win_total", "win_vs_opponents", "fee"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["player_id", "game_id"]).copy()
    df["player_id"] = df["player_id"].astype(int)
    df["game_id"] = df["game_id"].astype(str)

    df.loc[(df["game_type"] == "UNKNOWN") & df["bb"].notna(), "game_type"] = "RING"

    content = file_obj.getvalue()
    df["_src_sha1"] = _sha1_bytes(content)
    df["_src_name"] = getattr(file_obj, "name", "games")
    return df


# =========================
# SESSIONS / COPLAY GRAPH
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


def build_coplay_edges(sessions_df: pd.DataFrame) -> pd.DataFrame:
    if sessions_df.empty:
        return pd.DataFrame(columns=["player_a", "player_b", "game_type", "shared_sessions", "hu_sessions"])

    agg = defaultdict(lambda: {"shared_sessions": 0, "hu_sessions": 0})
    for sid, gtype, players, pn in sessions_df[["session_id", "game_type", "players", "players_n"]].itertuples(index=False):
        is_hu = (pn == 2)
        for a, b in combinations(players, 2):
            key = (min(a, b), max(a, b), gtype)
            agg[key]["shared_sessions"] += 1
            if is_hu:
                agg[key]["hu_sessions"] += 1

    rows = []
    for (a, b, gtype), v in agg.items():
        rows.append({"player_a": a, "player_b": b, "game_type": gtype,
                     "shared_sessions": v["shared_sessions"], "hu_sessions": v["hu_sessions"]})
    return pd.DataFrame(rows)


# =========================
# FLOWS (HU exact + multiway approx)
# =========================
def build_pair_flows_fast(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["from_player", "to_player", "game_type", "amount", "amount_bb", "games_cnt"])

    df = games_df[["game_id", "game_type", "player_id", "bb", "win_total", "win_vs_opponents"]].copy()

    # for Ring we prefer "vs opponents" (excludes jackpot/equity components)
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

        # multiway approx: allocate each loser's loss to winners proportionally to their win
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
# PLAYER / PAIR METRICS
# =========================
def summarize_db_player(db_df: pd.DataFrame, pid: int) -> dict:
    d = db_df[db_df["_player_id"] == pid].copy()
    if d.empty:
        return {"meta": {"player_id": pid}, "weeks_count": 0}

    meta = {
        "player_id": pid,
        "nick": d["_nick"].replace({"nan": ""}).iloc[-1],
        "ign": d["_ign"].replace({"nan": ""}).iloc[-1],
        "country": d["_country"].replace({"nan": ""}).iloc[-1],
        "agent": d["_agent"].replace({"nan": ""}).iloc[-1],
    }

    weeks_cnt = int(d["_week"].nunique())
    j_total = float(d["_j_total"].sum(skipna=True))
    p_total = float(d["_p_total"].sum(skipna=True))
    p_ring = float(d["_p_ring"].sum(skipna=True))
    p_mtt = float(d["_p_mtt"].sum(skipna=True))
    comm_ppsr = float(d["_club_comm_ppsr"].sum(skipna=True)) if "_club_comm_ppsr" in d.columns else 0.0

    # concentration by week
    by_w = d.groupby("_week")["_j_total"].sum().sort_values(ascending=False)
    top_week_share = np.nan
    if len(by_w) > 0 and abs(by_w.sum()) > 0:
        top_week_share = float(abs(by_w.iloc[0]) / abs(by_w.sum()))

    out = {
        "meta": meta,
        "weeks_count": weeks_cnt,
        "j_total": j_total,
        "p_total": p_total,
        "p_ring": p_ring,
        "p_mtt": p_mtt,
        "events_delta": float(j_total - p_total),
        "top_week_share": top_week_share,
        "comm_ppsr": comm_ppsr,
        "ring_over_comm_ppsr": safe_div(abs(p_ring), max(1.0, comm_ppsr)),
    }
    return out


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def get_player_partners(
    pid: int,
    flows_df: pd.DataFrame,
    coplay_edges_df: pd.DataFrame,
    game_type: str,
    topn: int = 10
) -> pd.DataFrame:
    f = flows_df[flows_df["game_type"] == game_type].copy()
    if f.empty:
        return pd.DataFrame()

    out = f[f["from_player"] == pid][["to_player", "amount", "amount_bb", "games_cnt"]].rename(
        columns={"to_player": "partner", "amount": "out_amount", "amount_bb": "out_bb", "games_cnt": "out_games"}
    )
    inc = f[f["to_player"] == pid][["from_player", "amount", "amount_bb", "games_cnt"]].rename(
        columns={"from_player": "partner", "amount": "in_amount", "amount_bb": "in_bb", "games_cnt": "in_games"}
    )

    m = pd.merge(out, inc, on="partner", how="outer").fillna(0.0)
    m["net"] = m["in_amount"] - m["out_amount"]      # positive => player received from partner
    m["gross"] = m["in_amount"] + m["out_amount"]
    m["pair_games"] = m["in_games"] + m["out_games"]

    if game_type == "RING":
        m["net_bb"] = m["in_bb"] - m["out_bb"]
        m["gross_bb"] = m["in_bb"] + m["out_bb"]
    else:
        m["net_bb"] = np.nan
        m["gross_bb"] = np.nan

    # one-sidedness: max(dir)/gross
    m["one_sidedness"] = m.apply(lambda r: (max(r["in_amount"], r["out_amount"]) / r["gross"]) if r["gross"] > 0 else np.nan, axis=1)

    # partner share in player's gross turnover
    total_gross = float(m["gross"].sum()) if float(m["gross"].sum()) > 0 else np.nan
    m["partner_share"] = m["gross"] / total_gross if pd.notna(total_gross) else np.nan

    # coplay join
    ce = coplay_edges_df[coplay_edges_df["game_type"] == game_type].copy()
    if not ce.empty:
        # normalize pair orientation to (a,b)
        ce["_k"] = ce.apply(lambda r: _pair_key(int(r["player_a"]), int(r["player_b"])), axis=1)
        m["_k"] = m["partner"].apply(lambda p: _pair_key(pid, int(p)))
        m = m.merge(ce[["_k", "shared_sessions", "hu_sessions"]], on="_k", how="left").drop(columns=["_k"])
    else:
        m["shared_sessions"] = np.nan
        m["hu_sessions"] = np.nan

    # ranking: prioritize suspicious by abs(net_bb) for ring, by abs(net) for tour
    if game_type == "RING":
        m["_rank"] = m["net_bb"].abs()
    else:
        m["_rank"] = m["net"].abs()

    m = m.sort_values("_rank", ascending=False).drop(columns=["_rank"]).head(topn)
    return m


# =========================
# SCORING
# =========================
def score_player(db_sum: dict, partners_ring: pd.DataFrame, partners_tour: pd.DataFrame, coverage: dict):
    score = 0
    reasons = []

    pid = int(db_sum["meta"]["player_id"])

    j_tot = float(db_sum.get("j_total", 0.0) or 0.0)
    events_delta = float(db_sum.get("events_delta", 0.0) or 0.0)
    weeks_cnt = int(db_sum.get("weeks_count", 0) or 0)
    top_week_share = db_sum.get("top_week_share", np.nan)

    # --- DB anomalies
    if j_tot >= 800:
        score += 8
        reasons.append(f"DB: крупный плюс по '{COL_J_TOTAL}'.")
    elif j_tot >= 300:
        score += 4
        reasons.append(f"DB: заметный плюс по '{COL_J_TOTAL}'.")

    if abs(events_delta) >= max(80.0, 0.35 * max(1.0, abs(j_tot))):
        score += 6
        reasons.append(f"DB: большая разница '{COL_J_TOTAL}' и '{COL_PLAYER_WIN_TOTAL}' — много событий (джекпот/эквити).")

    if weeks_cnt >= 3 and pd.notna(top_week_share) and float(top_week_share) >= 0.80 and abs(j_tot) >= 300:
        score += 5
        reasons.append("DB: сильная концентрация результата в одной неделе.")

    ring_over_comm_ppsr = db_sum.get("ring_over_comm_ppsr", np.nan)
    p_ring = float(db_sum.get("p_ring", 0.0) or 0.0)
    comm_ppsr = float(db_sum.get("comm_ppsr", 0.0) or 0.0)
    if pd.notna(ring_over_comm_ppsr) and comm_ppsr >= 10.0 and float(ring_over_comm_ppsr) >= 10 and abs(p_ring) >= 200:
        score += 5
        reasons.append(f"DB: '{COL_PLAYER_WIN_RING}' слишком высок относительно '{COL_CLUB_COMM_PPSR}'.")

    # --- Coverage
    ring_games = int(coverage.get("ring_games", 0))
    tour_games = int(coverage.get("tour_games", 0))
    if ring_games + tour_games == 0:
        reasons.append("GAMES: нет покрытия по играм (файл не загружен/парсинг/игрок не найден).")
    else:
        reasons.append(f"GAMES: покрытие есть (Ring игр {ring_games}, турниров {tour_games}).")

    # --- Pair scoring helper (NEW: top-K partners, not only top-1)
    def score_partners(df: pd.DataFrame, label: str):
        nonlocal score, reasons
        if df is None or df.empty:
            return

        for r in df.itertuples(index=False):
            partner = int(r.partner)
            shared = int(r.shared_sessions) if pd.notna(r.shared_sessions) else 0
            pair_games = int(r.pair_games) if pd.notna(r.pair_games) else int(r.out_games + r.in_games)
            one_sided = float(r.one_sidedness) if pd.notna(r.one_sidedness) else np.nan
            share = float(r.partner_share) if pd.notna(r.partner_share) else np.nan

            enough = (shared >= PAIR_MIN_SHARED_SESSIONS_STRONG) or (pair_games >= PAIR_MIN_PAIR_GAMES_STRONG)

            if label == "RING":
                net_bb = float(r.net_bb) if pd.notna(r.net_bb) else np.nan
                gross_bb = float(r.gross_bb) if pd.notna(r.gross_bb) else np.nan

                # critical net
                if pd.notna(net_bb) and abs(net_bb) >= PAIR_NET_CRITICAL_RING_BB:
                    score += 55
                    reasons.append(f"GAMES/RING: критический net-flow {fmt_bb(net_bb)} c партнёром {partner}.")
                # alert net
                elif pd.notna(net_bb) and abs(net_bb) >= PAIR_NET_ALERT_RING_BB and (enough or shared >= 1):
                    score += 25
                    reasons.append(f"GAMES/RING: крупный net-flow {fmt_bb(net_bb)} c партнёром {partner}.")
                # NEW: single-session very big transfer
                elif pd.notna(net_bb) and abs(net_bb) >= PAIR_SINGLE_SESSION_NET_ALERT_RING_BB and shared == 1:
                    score += 18
                    reasons.append(f"GAMES/RING: подозрительно большой net-flow за 1 сессию {fmt_bb(net_bb)} c партнёром {partner}.")

                # gross turnover collusion
                if pd.notna(gross_bb) and gross_bb >= PAIR_GROSS_CRITICAL_RING_BB and shared >= 3 and (pd.isna(share) or share >= PAIR_PARTNER_SHARE_ALERT):
                    score += 25
                    reasons.append(f"GAMES/RING: критический оборот пары {fmt_bb(gross_bb)} c партнёром {partner} (сговор/softplay).")
                elif pd.notna(gross_bb) and gross_bb >= PAIR_GROSS_ALERT_RING_BB and (enough or shared >= 2) and (pd.isna(share) or share >= PAIR_PARTNER_SHARE_ALERT):
                    score += 12
                    reasons.append(f"GAMES/RING: высокий оборот пары {fmt_bb(gross_bb)} c партнёром {partner}.")

            else:
                net = float(r.net)
                gross = float(r.gross)

                if abs(net) >= PAIR_NET_ALERT_TOUR and (enough or shared >= 1):
                    score += 18
                    reasons.append(f"GAMES/TOUR: подозрительный net-flow {fmt_money(net)} c партнёром {partner}.")
                elif abs(net) >= PAIR_SINGLE_SESSION_NET_ALERT_TOUR and shared == 1:
                    score += 12
                    reasons.append(f"GAMES/TOUR: большой net-flow за 1 сессию {fmt_money(net)} c партнёром {partner}.")

                if gross >= PAIR_GROSS_ALERT_TOUR and (enough or shared >= 2):
                    score += 8
                    reasons.append(f"GAMES/TOUR: высокий оборот пары {fmt_money(gross)} c партнёром {partner}.")

            # pattern strengthening
            if pd.notna(one_sided) and one_sided >= PAIR_ONE_SIDED_ALERT and (enough or shared >= 1):
                score += 6
                reasons.append(f"GAMES/{label}: one-sided {fmt_pct(one_sided)} в паре с {partner}.")

            if pd.notna(share) and share >= 0.75 and (enough or shared >= 2):
                score += 6
                reasons.append(f"GAMES/{label}: высокая доля партнёра в обороте {fmt_pct(share)} (возможная связка) с {partner}.")

    # Take top-K partners
    score_partners(partners_ring, "RING")
    score_partners(partners_tour, "TOUR")

    return int(score), reasons


# =========================
# SESSION STATE: incremental loading
# =========================
def _ss_init():
    st.session_state.setdefault("db_all", pd.DataFrame())
    st.session_state.setdefault("games_all", pd.DataFrame())
    st.session_state.setdefault("coplay_edges", pd.DataFrame())
    st.session_state.setdefault("flows", pd.DataFrame())
    st.session_state.setdefault("sessions", pd.DataFrame())
    st.session_state.setdefault("loaded_sha1", set())
    st.session_state.setdefault("dirty_graph", True)


def add_db(file_obj):
    df = load_db_any(file_obj)
    sha1 = df["_src_sha1"].iloc[0] if not df.empty else _sha1_bytes(file_obj.getvalue())
    if sha1 in st.session_state["loaded_sha1"]:
        return False, "Этот DB-файл уже загружен (по SHA1)."
    st.session_state["loaded_sha1"].add(sha1)

    cur = st.session_state["db_all"]
    cur = pd.concat([cur, df], ignore_index=True)
    # dedupe by week+player_id (последний файл побеждает)
    cur = cur.sort_values(["_player_id", "_week"]).drop_duplicates(["_player_id", "_week"], keep="last")
    st.session_state["db_all"] = cur.reset_index(drop=True)
    st.session_state["dirty_graph"] = True
    return True, f"DB добавлен: {getattr(file_obj, 'name', 'db')} (строк: {len(df)})."


def add_games(file_obj):
    df = parse_games_pppoker_export(file_obj)
    sha1 = df["_src_sha1"].iloc[0] if not df.empty else _sha1_bytes(file_obj.getvalue())
    if sha1 in st.session_state["loaded_sha1"]:
        return False, "Этот GAMES-файл уже загружен (по SHA1)."
    st.session_state["loaded_sha1"].add(sha1)

    cur = st.session_state["games_all"]
    cur = pd.concat([cur, df], ignore_index=True)
    cur = cur.drop_duplicates(["game_id", "game_type", "player_id"], keep="last")
    st.session_state["games_all"] = cur.reset_index(drop=True)
    st.session_state["dirty_graph"] = True
    return True, f"GAMES добавлен: {getattr(file_obj, 'name', 'games')} (строк: {len(df)})."


def rebuild_graph_if_needed():
    if not st.session_state.get("dirty_graph", True):
        return
    games_df = st.session_state["games_all"]
    sessions = build_sessions_from_games(games_df)
    coplay_edges = build_coplay_edges(sessions)
    flows = build_pair_flows_fast(games_df)

    st.session_state["sessions"] = sessions
    st.session_state["coplay_edges"] = coplay_edges
    st.session_state["flows"] = flows
    st.session_state["dirty_graph"] = False


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

_ss_init()

with st.sidebar:
    st.subheader("Загрузка данных")
    db_upl = st.file_uploader("DB (Общий) — CSV/XLSX", type=["csv", "xlsx", "xls"])
    if st.button("Добавить DB"):
        if db_upl is None:
            st.warning("Загрузите файл DB.")
        else:
            ok, msg = add_db(db_upl)
            (st.success if ok else st.info)(msg)

    games_upl = st.file_uploader("Игры — PPPoker export (CSV/TXT)", type=["csv", "txt"])
    if st.button("Добавить GAMES"):
        if games_upl is None:
            st.warning("Загрузите файл игр.")
        else:
            ok, msg = add_games(games_upl)
            (st.success if ok else st.info)(msg)

    st.divider()
    st.caption("Сессия: социальная база хранится в session_state. Можно догружать недели по очереди.")
    if st.button("Сбросить сессию"):
        for k in ["db_all", "games_all", "coplay_edges", "flows", "sessions", "loaded_sha1", "dirty_graph"]:
            if k in st.session_state:
                del st.session_state[k]
        _ss_init()
        st.success("Сессия очищена.")

rebuild_graph_if_needed()

colA, colB = st.columns([1, 2])
with colA:
    pid = st.number_input("ID игрока для проверки", min_value=0, step=1, value=0)

with colB:
    st.write("Статус данных:")
    st.write(f"DB строк: {len(st.session_state['db_all'])} | Games строк: {len(st.session_state['games_all'])} | "
             f"Edges: {len(st.session_state['coplay_edges'])} | Flows: {len(st.session_state['flows'])}")

if pid and int(pid) > 0:
    pid = int(pid)
    db_df = st.session_state["db_all"]
    games_df = st.session_state["games_all"]
    coplay_edges = st.session_state["coplay_edges"]
    flows = st.session_state["flows"]

    db_sum = summarize_db_player(db_df, pid)

    # coverage
    gpid = games_df[games_df["player_id"] == pid]
    coverage = {
        "ring_games": int((gpid["game_type"] == "RING").sum()) if not gpid.empty else 0,
        "tour_games": int((gpid["game_type"] == "TOURNAMENT").sum()) if not gpid.empty else 0,
    }

    partners_ring = get_player_partners(pid, flows, coplay_edges, "RING", topn=10)
    partners_tour = get_player_partners(pid, flows, coplay_edges, "TOURNAMENT", topn=10)

    score, reasons = score_player(db_sum, partners_ring, partners_tour, coverage)
    decision = risk_decision(score)

    top = st.container()
    with top:
        st.subheader(f"Диагноз: {decision} | score={score}")
        m = db_sum.get("meta", {})
        st.write(f"Игрок: {pid} | Ник: {m.get('nick','')} | IGN: {m.get('ign','')} | Страна: {m.get('country','')} | Агент: {m.get('agent','')}")
        st.write(f"DB: недель={db_sum.get('weeks_count',0)} | J_total={fmt_money(db_sum.get('j_total',0))} | "
                 f"P_total={fmt_money(db_sum.get('p_total',0))} | events_delta={fmt_money(db_sum.get('events_delta',0))}")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Подозрительные партнёры (Ring)")
        if partners_ring is None or partners_ring.empty:
            st.info("Нет данных по Ring/нет значимых связей.")
        else:
            view = partners_ring.copy()
            view["net_bb"] = view["net_bb"].apply(lambda x: float(x) if pd.notna(x) else np.nan)
            view["gross_bb"] = view["gross_bb"].apply(lambda x: float(x) if pd.notna(x) else np.nan)
            st.dataframe(view, use_container_width=True)

    with right:
        st.markdown("### Подозрительные партнёры (Tournaments)")
        if partners_tour is None or partners_tour.empty:
            st.info("Нет данных по турнирам/нет значимых связей.")
        else:
            st.dataframe(partners_tour, use_container_width=True)

    st.divider()
    st.markdown("### Объяснение скоринга")
    for r in reasons[:40]:
        st.write("• " + r)

else:
    st.info("Введите ID игрока, загрузите DB и/или GAMES (можно догружать недели).")
