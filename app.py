import io
import re
import json
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

# HU dominance (важно для переливов)
HU_DOMINANCE_MIN_HU = 2
HU_DOMINANCE_RATIO = 0.75

# --- FLOWS (BB-aware) ---
PAIR_NET_ALERT_RING_BB = 25.0
PAIR_NET_CRITICAL_RING_BB = 50.0

# Collusion turnover
PAIR_GROSS_ALERT_RING_BB = 120.0
PAIR_GROSS_CRITICAL_RING_BB = 250.0

PAIR_ONE_SIDED_ALERT = 0.88
PAIR_DIR_CONSIST_ALERT = 0.78
PAIR_PARTNER_SHARE_ALERT = 0.60

PAIR_MIN_SHARED_SESSIONS_STRONG = 2
PAIR_MIN_PAIR_GAMES_STRONG = 3

# Tournaments thresholds in currency
PAIR_NET_ALERT_TOUR = 60.0
PAIR_GROSS_ALERT_TOUR = 150.0

# Extremes
SINGLE_GAME_WIN_ALERT_TOUR = 150.0

# Regex PPPoker export
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)
TABLE_NAME_RE = re.compile(r"Название стола:\s*(.+?)\s*$", re.IGNORECASE)
START_END_RE = re.compile(r"Начало:\s*([0-9/:\s]+)\s+By.+?Окончание:\s*([0-9/:\s]+)", re.IGNORECASE)

# Type hints
RING_HINT_RE = re.compile(r"\bPPSR\b|PLO|OFC|NLH|Bomb Pot|Ante|3-1|HU\b|Heads", re.IGNORECASE)
TOUR_HINT_RE = re.compile(r"\bPPST\b|Бай-ин:|satellite|pko|mko\b|SNG\b|MTT\b", re.IGNORECASE)

# Stakes: 0.2/0.4
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
# Persistent file cache
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


def safe_div(a, b):
    if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return a / b


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

        # descriptor detection: PPSR/PPST OR contains stakes OR tour hints
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

    # если type UNKNOWN, но есть ставки — считаем Ring
    df.loc[(df["game_type"] == "UNKNOWN") & df["bb"].notna(), "game_type"] = "RING"
    return df


# =========================
# SESSIONS
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
# FLOWS (HU exact + multiway approx)
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
        [
            {
                "from_player": k[0],
                "to_player": k[1],
                "game_type": k[2],
                "amount": float(v),
                "amount_bb": float(flows_bb.get(k, np.nan)),
                "games_cnt": int(games_cnt[k]),
            }
            for k, v in flows_amt.items()
        ]
    )
    out = out.groupby(["from_player", "to_player", "game_type"], as_index=False).agg(
        amount=("amount", "sum"),
        amount_bb=("amount_bb", "sum"),
        games_cnt=("games_cnt", "sum"),
    )
    return out


# =========================
# INDEXES
# =========================
def build_games_indexes(games_df: pd.DataFrame, sessions_df: pd.DataFrame, flows_df: pd.DataFrame):
    idx = {}

    # player_game_series (dir consistency)
    if games_df.empty:
        idx["player_game_series"] = {}
        idx["extremes"] = {}
    else:
        d = games_df[["game_type", "player_id", "game_id", "win_total", "win_vs_opponents"]].copy()
        d["_flow_win"] = np.where(
            (d["game_type"] == "RING") & d["win_vs_opponents"].notna(),
            d["win_vs_opponents"],
            d["win_total"],
        )
        d = d[d["_flow_win"].notna()].copy()
        d["_flow_win"] = pd.to_numeric(d["_flow_win"], errors="coerce")
        d = d[d["_flow_win"].notna()].copy()
        d["player_id"] = d["player_id"].astype(int)
        d["game_id"] = d["game_id"].astype(str)

        series = {}
        extremes = {}
        for (gt, pid), g in d.groupby(["game_type", "player_id"], sort=False):
            s = pd.Series(g["_flow_win"].to_numpy(dtype=float), index=g["game_id"].to_numpy())
            series[(gt, int(pid))] = s
            if gt == "TOURNAMENT":
                big = s[s >= SINGLE_GAME_WIN_ALERT_TOUR]
                extremes[(gt, int(pid))] = list(big.index[:12]) if not big.empty else []
            else:
                extremes[(gt, int(pid))] = []
        idx["player_game_series"] = series
        idx["extremes"] = extremes

    # sessions inverted + coplay counters (incl HU-specific)
    sessions_by_player = defaultdict(list)
    sessions_n = {}
    coplay_counter = defaultdict(lambda: defaultdict(int))
    coplay_counter_hu = defaultdict(lambda: defaultdict(int))
    coplay_sessions_cnt = defaultdict(int)
    coplay_hu_cnt = defaultdict(int)
    coplay_sh_cnt = defaultdict(int)

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
                if pn <= 3:
                    coplay_sh_cnt[key] += 1

            # all sessions coplay
            for i in range(len(pls)):
                pi = pls[i]
                di = coplay_counter[(gt, pi)]
                for j in range(len(pls)):
                    if i == j:
                        continue
                    di[pls[j]] += 1

            # HU-only coplay
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
    idx["coplay_sh_cnt"] = dict(coplay_sh_cnt)

    # flows -> in/out maps + totals + top pair
    if flows_df.empty:
        idx["in_map"] = {}
        idx["out_map"] = {}
        idx["flow_totals"] = {}
        idx["top_pair"] = {}
        return idx

    f = flows_df.copy()
    f["from_player"] = f["from_player"].astype(int)
    f["to_player"] = f["to_player"].astype(int)

    in_map = {}
    for (gt, to_pid), g in f.groupby(["game_type", "to_player"], sort=False):
        s = g.groupby("from_player")["amount"].sum().sort_values(ascending=False)
        in_map[(gt, int(to_pid))] = s

    out_map = {}
    for (gt, from_pid), g in f.groupby(["game_type", "from_player"], sort=False):
        s = g.groupby("to_player")["amount"].sum().sort_values(ascending=False)
        out_map[(gt, int(from_pid))] = s

    idx["in_map"] = in_map
    idx["out_map"] = out_map

    inflow_total = f.groupby(["game_type", "to_player"])["amount"].sum()
    outflow_total = f.groupby(["game_type", "from_player"])["amount"].sum()
    flow_totals = {}
    for (gt, pid), in_amt in inflow_total.items():
        pid = int(pid)
        out_amt = float(outflow_total.get((gt, pid), 0.0))
        in_amt = float(in_amt)
        gross = in_amt + out_amt
        one_sided = abs(in_amt - out_amt) / gross if gross > 0 else 0.0
        flow_totals[(gt, pid)] = {"in": in_amt, "out": out_amt, "gross": gross, "one_sided": float(one_sided)}
    for (gt, pid), out_amt in outflow_total.items():
        pid = int(pid)
        if (gt, pid) in flow_totals:
            continue
        out_amt = float(out_amt)
        gross = out_amt
        flow_totals[(gt, pid)] = {"in": 0.0, "out": out_amt, "gross": gross, "one_sided": 1.0 if gross > 0 else 0.0}
    idx["flow_totals"] = flow_totals

    # top pair by abs(net_bb) for RING (fallback to abs(net))
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

    vq = pair.rename(columns={"q": "player_id", "p": "partner_id", "net_to_q": "net", "net_to_q_bb": "net_bb"})
    vp = pair.rename(columns={"p": "player_id", "q": "partner_id", "net_to_q": "net", "net_to_q_bb": "net_bb"})
    vp["net"] = -vp["net"]
    vp["net_bb"] = -vp["net_bb"]

    player_pairs = pd.concat([vq, vp], ignore_index=True)
    player_pairs["player_id"] = player_pairs["player_id"].astype(int)
    player_pairs["partner_id"] = player_pairs["partner_id"].astype(int)

    gross_total = player_pairs.groupby(["game_type", "player_id"])["gross"].sum()
    gross_total_bb = player_pairs.groupby(["game_type", "player_id"])["gross_bb"].sum()

    def pair_rank_row(r):
        if r["game_type"] == "RING" and pd.notna(r["net_bb"]):
            return abs(float(r["net_bb"]))
        return abs(float(r["net"]))

    player_pairs["_rank"] = player_pairs.apply(pair_rank_row, axis=1)
    top_rows = player_pairs.sort_values("_rank", ascending=False).groupby(["game_type", "player_id"], as_index=False).head(1)

    top_pair = {}
    for gt, pid, partner, net, net_bb, gross, gross_bb, gcnt, _rank in top_rows[
        ["game_type", "player_id", "partner_id", "net", "net_bb", "gross", "gross_bb", "games_cnt", "_rank"]
    ].itertuples(index=False):
        gt = str(gt)
        pid = int(pid)
        partner = int(partner)
        gtot = float(gross_total.get((gt, pid), 0.0))
        gtot_bb = float(gross_total_bb.get((gt, pid), 0.0))
        top_pair[(gt, pid)] = {
            "partner": partner,
            "net": float(net),
            "net_bb": float(net_bb) if pd.notna(net_bb) else np.nan,
            "gross": float(gross),
            "gross_bb": float(gross_bb) if pd.notna(gross_bb) else np.nan,
            "gross_total": gtot,
            "gross_total_bb": gtot_bb,
            "partner_share": float(gross / gtot) if gtot > 0 else 0.0,
            "partner_share_bb": float(gross_bb / gtot_bb) if (gtot_bb > 0 and pd.notna(gross_bb)) else np.nan,
            "games_cnt": int(gcnt),
        }
    idx["top_pair"] = top_pair
    return idx


# =========================
# FEATURES
# =========================
def coplay_features_fast(target_id: int, idx: dict, game_type: str) -> dict:
    key = (game_type, int(target_id))
    sessions_count = int(idx.get("coplay_sessions_cnt", {}).get(key, 0))
    hu_sessions = int(idx.get("coplay_hu_cnt", {}).get(key, 0))
    sh_sessions = int(idx.get("coplay_sh_cnt", {}).get(key, 0))
    counter = idx.get("coplay_counter", {}).get(key, {})

    partners = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    unique_opponents = len(partners)
    top2_share = float((partners[0][1] + partners[1][1]) / sessions_count) if sessions_count and len(partners) >= 2 else 0.0

    # HU partners
    hu_counter = idx.get("coplay_counter_hu", {}).get(key, {})
    hu_partners = sorted(hu_counter.items(), key=lambda x: x[1], reverse=True)
    top_hu_partner = int(hu_partners[0][0]) if hu_partners else None
    top_hu_share = float(hu_partners[0][1] / max(1, hu_sessions)) if hu_partners else 0.0

    return {
        "sessions_count": sessions_count,
        "unique_opponents": unique_opponents,
        "top2_coplay_share": top2_share,
        "hu_sessions": hu_sessions,
        "sh_sessions": sh_sessions,
        "top_partners": partners[:12],
        "top_hu_partner": top_hu_partner,
        "top_hu_share": top_hu_share,
    }


def _shared_sessions_list(pid_a: int, pid_b: int, idx: dict, game_type: str, limit: int = 20) -> list[str]:
    a = idx.get("sessions_by_player", {}).get((game_type, int(pid_a)), [])
    b = idx.get("sessions_by_player", {}).get((game_type, int(pid_b)), [])
    if not a or not b:
        return []
    if len(a) < len(b):
        sb = set(a)
        shared = [x for x in b if x in sb]
    else:
        sa = set(a)
        shared = [x for x in b if x in sa]
    return shared[:limit]


def pair_ctx_fast(target_id: int, partner_id: int, idx: dict, game_type: str) -> dict:
    shared = _shared_sessions_list(target_id, partner_id, idx, game_type, limit=999999)
    shared_cnt = int(len(shared))

    hu_share = 0.0
    if shared_cnt:
        sess_n = idx.get("sessions_n", {})
        hu = sum(1 for sid in shared if sess_n.get((game_type, sid), 0) == 2)
        hu_share = hu / shared_cnt

    s1 = idx.get("player_game_series", {}).get((game_type, int(target_id)))
    s2 = idx.get("player_game_series", {}).get((game_type, int(partner_id)))
    if s1 is None or s2 is None:
        dir_cons = 0.0
    else:
        a1, a2 = s1.align(s2, join="inner")
        if a1.empty:
            dir_cons = 0.0
        else:
            t = a1.to_numpy(dtype=float)
            p = a2.to_numpy(dtype=float)
            dir1 = float(np.mean((t > 0) & (p < 0)))
            dir2 = float(np.mean((t < 0) & (p > 0)))
            dir_cons = max(dir1, dir2)

    return {"shared_sessions": shared_cnt, "hu_share": float(hu_share), "dir_consistency": float(dir_cons)}


def transfer_features_fast(target_id: int, idx: dict, game_type: str) -> dict:
    pid = int(target_id)

    in_s = idx.get("in_map", {}).get((game_type, pid))
    out_s = idx.get("out_map", {}).get((game_type, pid))
    top_inflows = [(int(k), float(v)) for k, v in in_s.head(12).items()] if in_s is not None and len(in_s) else []
    top_outflows = [(int(k), float(v)) for k, v in out_s.head(12).items()] if out_s is not None and len(out_s) else []

    totals = idx.get("flow_totals", {}).get((game_type, pid), {"in": 0.0, "out": 0.0, "gross": 0.0, "one_sided": 0.0})
    top = idx.get("top_pair", {}).get((game_type, pid))

    if top is None:
        return {
            "top_inflows": top_inflows,
            "top_outflows": top_outflows,
            "top_net_partner": None,
            "top_net": 0.0,
            "top_net_bb": np.nan,
            "top_gross": 0.0,
            "top_gross_bb": np.nan,
            "top_partner_share": 0.0,
            "top_partner_share_bb": np.nan,
            "top_pair_games_cnt": 0,
            "one_sidedness": float(totals.get("one_sided", 0.0)),
            "pair_ctx": {},
            "shared_sessions_preview": [],
        }

    partner = int(top["partner"])
    ctx = pair_ctx_fast(pid, partner, idx, game_type)
    shared_preview = _shared_sessions_list(pid, partner, idx, game_type, limit=20)

    return {
        "top_inflows": top_inflows,
        "top_outflows": top_outflows,
        "top_net_partner": partner,
        "top_net": float(top["net"]),
        "top_net_bb": float(top["net_bb"]) if pd.notna(top["net_bb"]) else np.nan,
        "top_gross": float(top["gross"]),
        "top_gross_bb": float(top["gross_bb"]) if pd.notna(top["gross_bb"]) else np.nan,
        "top_partner_share": float(top["partner_share"]),
        "top_partner_share_bb": float(top["partner_share_bb"]) if pd.notna(top["partner_share_bb"]) else np.nan,
        "top_pair_games_cnt": int(top.get("games_cnt", 0) or 0),
        "one_sidedness": float(totals.get("one_sided", 0.0)),
        "pair_ctx": ctx,
        "shared_sessions_preview": shared_preview,
    }


# =========================
# PERIOD FILTER + DB SUMMARY
# =========================
def apply_weeks_filter(db_df: pd.DataFrame, weeks_mode: str, last_n: int, week_from: int, week_to: int) -> pd.DataFrame:
    d = db_df.copy()
    weeks = sorted([w for w in d["_week"].unique().tolist() if w >= 0])
    if weeks_mode == "Все недели":
        return d
    if weeks_mode == "Последние N недель":
        if not weeks:
            return d
        max_w = max(weeks)
        min_w = max_w - max(0, int(last_n) - 1)
        return d[(d["_week"] >= min_w) & (d["_week"] <= max_w)].copy()
    return d[(d["_week"] >= int(week_from)) & (d["_week"] <= int(week_to))].copy()


def db_summary_for_player(db_period: pd.DataFrame, player_id: int):
    d = db_period[db_period["_player_id"] == int(player_id)].copy()
    if d.empty:
        return None, None

    num_cols = [c for c in d.columns if c.startswith(("_j_", "_p_", "_club_", "_ticket_", "_custom_"))]
    by_week = d.groupby("_week", as_index=False)[num_cols].sum(min_count=1).sort_values("_week")
    agg = by_week[num_cols].sum(numeric_only=True)

    total_j = float(agg.get("_j_total", 0.0) or 0.0)
    p_total = float(agg.get("_p_total", 0.0) or 0.0)
    p_ring = float(agg.get("_p_ring", 0.0) or 0.0)
    p_mtt = float(agg.get("_p_mtt", 0.0) or 0.0)

    comm_total = float(agg.get("_club_comm_total", 0.0) or 0.0)
    comm_ppsr = float(agg.get("_club_comm_ppsr", 0.0) or 0.0)
    comm_ppst = float(agg.get("_club_comm_ppst", 0.0) or 0.0)

    events_delta = float(total_j - p_total)
    poker_profit = float((p_ring or 0.0) + (p_mtt or 0.0))
    ring_over_comm_ppsr = safe_div(p_ring, comm_ppsr) if comm_ppsr > 0 else np.nan

    if by_week.empty:
        top_week_share = np.nan
    else:
        top_row = by_week.sort_values("_j_total", ascending=False).iloc[0]
        top_week_j = float(top_row.get("_j_total", 0.0) or 0.0)
        top_week_share = safe_div(top_week_j, total_j) if total_j != 0 else np.nan

    meta_row = d.sort_values("_week").iloc[-1]
    meta = {
        "player_id": int(player_id),
        "country": str(meta_row.get("_country", "")),
        "nick": str(meta_row.get("_nick", "")),
        "ign": str(meta_row.get("_ign", "")),
        "agent": str(meta_row.get("_agent", "")),
        "agent_id": meta_row.get("_agent_id", np.nan),
        "super_agent": str(meta_row.get("_super_agent", "")),
        "super_agent_id": meta_row.get("_super_agent_id", np.nan),
    }

    summary = {
        "weeks_count": int(len(by_week)),
        "j_total": total_j,
        "p_total": p_total,
        "events_delta": events_delta,
        "p_ring": p_ring,
        "p_mtt": p_mtt,
        "poker_profit": poker_profit,
        "comm_total": comm_total,
        "comm_ppsr": comm_ppsr,
        "comm_ppst": comm_ppst,
        "ring_over_comm_ppsr": ring_over_comm_ppsr,
        "top_week_share": top_week_share,
        "meta": meta,
    }
    return summary, by_week


def agent_match_bonus(db_df: pd.DataFrame, pid_a: int, pid_b: int) -> tuple[int, str | None]:
    a = db_df[db_df["_player_id"] == pid_a].tail(1)
    b = db_df[db_df["_player_id"] == pid_b].tail(1)
    if a.empty or b.empty:
        return 0, None

    a_ag = a.iloc[0].get("_agent_id")
    b_ag = b.iloc[0].get("_agent_id")
    if pd.notna(a_ag) and pd.notna(b_ag) and float(a_ag) == float(b_ag):
        return 6, "Оба игрока у одного агента."

    a_sag = a.iloc[0].get("_super_agent_id")
    b_sag = b.iloc[0].get("_super_agent_id")
    if pd.notna(a_sag) and pd.notna(b_sag) and float(a_sag) == float(b_sag):
        return 4, "Оба игрока у одного суперагента."

    return 0, None


# =========================
# SCORING (chip dumping + collusion)
# =========================
def score_player(
    db_df: pd.DataFrame,
    db_sum: dict,
    cop_ring: dict,
    cop_tour: dict,
    trf_ring: dict,
    trf_tour: dict,
    coverage: dict,
):
    """
    Коррекция логики для снижения false-positive:
    - DB-сигналы НЕ добавляют риск-очки (только пояснения).
    - Co-play сам по себе НЕ добавляет риск-очки (только как усилитель, если уже есть сильные flow-сигналы).
    - Риск повышается только при подтверждённых игровых сигналах + достаточной выборке по паре.
    """
    score = 0
    score_games = 0
    reasons = []

    pid = int(db_sum["meta"]["player_id"])

    # --- DB (INFO only; no points) ---
    j_tot = float(db_sum.get("j_total", 0.0) or 0.0)
    events_delta = float(db_sum.get("events_delta", 0.0) or 0.0)
    weeks_cnt = int(db_sum.get("weeks_count", 0) or 0)
    top_week_share = db_sum.get("top_week_share", np.nan)

    if j_tot >= 800:
        reasons.append(f"DB (info): крупный плюс по '{COL_J_TOTAL}' = {fmt_money(j_tot)}.")
    elif j_tot >= 300:
        reasons.append(f"DB (info): заметный плюс по '{COL_J_TOTAL}' = {fmt_money(j_tot)}.")

    if abs(events_delta) >= max(120.0, 0.50 * max(1.0, abs(j_tot))):
        reasons.append(
            f"DB (info): большая разница '{COL_J_TOTAL}' и '{COL_PLAYER_WIN_TOTAL}' (events) = {fmt_money(events_delta)}."
        )

    if weeks_cnt >= 3 and pd.notna(top_week_share) and float(top_week_share) >= 0.80 and abs(j_tot) >= 300:
        reasons.append("DB (info): сильная концентрация результата в одной неделе.")

    ring_over_comm_ppsr = db_sum.get("ring_over_comm_ppsr", np.nan)
    p_ring = float(db_sum.get("p_ring", 0.0) or 0.0)
    comm_ppsr = float(db_sum.get("comm_ppsr", 0.0) or 0.0)
    if pd.notna(ring_over_comm_ppsr) and comm_ppsr >= 10.0 and float(ring_over_comm_ppsr) >= 10 and abs(p_ring) >= 200:
        reasons.append(f"DB (info): '{COL_PLAYER_WIN_RING}' высок относительно '{COL_CLUB_COMM_PPSR}' (возможна аномалия по рейку).")

    # --- Coverage note (no scoring) ---
    ring_games = int(coverage.get("ring_games", 0))
    tour_games = int(coverage.get("tour_games", 0))
    if ring_games + tour_games == 0:
        reasons.append("GAMES: нет покрытия по играм (файл не загружен/игрок не найден/парсинг не смог).")
    else:
        reasons.append(f"GAMES: покрытие есть (Ring игр {ring_games}, турниров {tour_games}).")

    # --- Flow checks: Ring (BB) + Tournament (currency) ---
    def check_flow(trf: dict, label: str):
        nonlocal score_games, reasons

        partner = trf.get("top_net_partner")
        if partner is None:
            return

        net_val = float(trf.get("top_net", 0.0) or 0.0)
        net_bb = trf.get("top_net_bb", np.nan)
        gross_val = float(trf.get("top_gross", 0.0) or 0.0)
        gross_bb = trf.get("top_gross_bb", np.nan)
        pair_games_cnt = int(trf.get("top_pair_games_cnt", 0) or 0)

        p_share_bb = trf.get("top_partner_share_bb", np.nan)
        partner_share = float(p_share_bb) if pd.notna(p_share_bb) else float(trf.get("top_partner_share", 0.0) or 0.0)

        ctx = trf.get("pair_ctx", {}) or {}
        shared = int(ctx.get("shared_sessions", 0) or 0)
        dir_cons = float(ctx.get("dir_consistency", 0.0) or 0.0)
        hu_share = float(ctx.get("hu_share", 0.0) or 0.0)
        one_sided = float(trf.get("one_sidedness", 0.0) or 0.0)

        # Усиленный критерий "достаточная выборка по паре", чтобы не ловить честные единичные пики
        enough_pair = (shared >= 3) or (pair_games_cnt >= 4)

        if label == "RING":
            # 1) NET — основной признак. Теперь требуем подтверждение объёмом по паре.
            if pd.notna(net_bb):
                absnetbb = abs(float(net_bb))
                if absnetbb >= PAIR_NET_CRITICAL_RING_BB and ((shared >= 2) or (pair_games_cnt >= 3)):
                    score_games += 55
                    reasons.append(f"GAMES/RING: критический net-flow {fmt_bb(net_bb)} с партнёром {partner} (shared={shared}, pair_games={pair_games_cnt}).")
                elif absnetbb >= PAIR_NET_ALERT_RING_BB and enough_pair and (partner_share >= PAIR_PARTNER_SHARE_ALERT):
                    score_games += 25
                    reasons.append(
                        f"GAMES/RING: крупный net-flow {fmt_bb(net_bb)} с партнёром {partner} "
                        f"(share={fmt_pct(partner_share)}, shared={shared}, pair_games={pair_games_cnt})."
                    )
            else:
                absnet = abs(net_val)
                if absnet >= 120 and ((shared >= 2) or (pair_games_cnt >= 3)):
                    score_games += 45
                    reasons.append(f"GAMES/RING: критический net-flow (BB не извлечён) {fmt_money(net_val)} с партнёром {partner}.")
                elif absnet >= 60 and enough_pair and (partner_share >= PAIR_PARTNER_SHARE_ALERT):
                    score_games += 18
                    reasons.append(f"GAMES/RING: крупный net-flow (BB не извлечён) {fmt_money(net_val)} с партнёром {partner}.")

            # 2) GROSS — только при явной "связке" (иначе это может быть просто активная игра)
            if pd.notna(gross_bb):
                gb = float(gross_bb)
                if (gb >= PAIR_GROSS_CRITICAL_RING_BB) and (shared >= 4) and (pair_games_cnt >= 6) and (partner_share >= 0.65):
                    score_games += 25
                    reasons.append(
                        f"GAMES/RING: критический оборот пары {fmt_bb(gross_bb)} с партнёром {partner} "
                        f"(share={fmt_pct(partner_share)}, shared={shared}, pair_games={pair_games_cnt})."
                    )
                elif (gb >= PAIR_GROSS_ALERT_RING_BB) and (shared >= 3) and (pair_games_cnt >= 5) and (partner_share >= 0.65):
                    score_games += 12
                    reasons.append(
                        f"GAMES/RING: высокий оборот пары {fmt_bb(gross_bb)} с партнёром {partner} "
                        f"(share={fmt_pct(partner_share)}, shared={shared}, pair_games={pair_games_cnt})."
                    )

            # 3) Паттерн (one-sided/dir/HU) — только как усилитель при очень сильной связке
            if (shared >= 4) and (pair_games_cnt >= 6) and (partner_share >= 0.70) and (
                (one_sided >= PAIR_ONE_SIDED_ALERT) or (dir_cons >= PAIR_DIR_CONSIST_ALERT) or (hu_share >= 0.75)
            ):
                score_games += 12
                reasons.append(
                    f"GAMES/RING: устойчивый паттерн (one-sided/dir/HU) в паре с партнёром {partner} "
                    f"(HU={fmt_pct(hu_share)}, dir={fmt_pct(dir_cons)}, one-sided={fmt_pct(one_sided)})."
                )

            # agent bonus only if already suspicious in games (not DB)
            if score_games >= T_APPROVE:
                bonus, txt = agent_match_bonus(db_df, int(partner), pid)
                if bonus > 0 and txt:
                    score_games += bonus
                    reasons.append(f"DB: усилитель риска — {txt}")

            # context line
            net_str = fmt_bb(net_bb) if pd.notna(net_bb) else f"{fmt_money(net_val)} (BB не извлечён)"
            gross_str = fmt_bb(gross_bb) if pd.notna(gross_bb) else fmt_money(gross_val)
            reasons.append(
                f"Контекст пары Ring: partner={partner}, net={net_str}, gross={gross_str}, shared={shared}, pair_games={pair_games_cnt}, "
                f"share={fmt_pct(partner_share)}, HU_in_pair={fmt_pct(hu_share)}, dir={fmt_pct(dir_cons)}, one-sided={fmt_pct(one_sided)}."
            )

        else:
            # tournament signals (less precise): также требуем подтверждение объёмом по паре
            absnet = abs(net_val)
            if (absnet >= PAIR_NET_ALERT_TOUR) and ((shared >= 3) or (pair_games_cnt >= 4)) and (partner_share >= 0.65):
                score_games += 12
                reasons.append(f"GAMES/TOURNAMENT: крупный net-flow {fmt_money(net_val)} с партнёром {partner} (share={fmt_pct(partner_share)}).")
            if (gross_val >= PAIR_GROSS_ALERT_TOUR) and ((shared >= 3) or (pair_games_cnt >= 4)) and (partner_share >= 0.70):
                score_games += 8
                reasons.append(f"GAMES/TOURNAMENT: высокий оборот пары {fmt_money(gross_val)} с партнёром {partner} (share={fmt_pct(partner_share)}).")

    check_flow(trf_ring, "RING")
    check_flow(trf_tour, "TOURNAMENT")

    # --- Co-play Ring: теперь только как мягкий усилитель, если уже есть game-risk ---
    # Это снижает false-positive для честных регулярных игроков/пар по HU.
    if score_games >= T_APPROVE:
        if cop_ring.get("sessions_count", 0) >= 10 and cop_ring.get("top2_coplay_share", 0.0) >= 0.95:
            score_games += 4
            reasons.append("GAMES/RING: co-play усилитель — топ‑2 оппонента встречаются почти всегда (добавлен небольшой бонус к риску).")

        hu_ratio = (cop_ring.get("hu_sessions", 0) / max(1, cop_ring.get("sessions_count", 0)))
        if cop_ring.get("hu_sessions", 0) >= 8 and hu_ratio >= 0.85 and (cop_ring.get("top_hu_share", 0.0) >= 0.90):
            score_games += 4
            reasons.append("GAMES/RING: co-play усилитель — высокая HU‑доля и почти всегда один HU‑партнёр (добавлен небольшой бонус к риску).")

    score = int(max(0, min(100, score_games)))
    decision = risk_decision(score)

    if decision == "APPROVE":
        manager_text = "Рекомендация: можно проводить (нет подтверждённых сильных сигналов перелива/сговора по играм)."
    elif decision == "FAST_CHECK":
        manager_text = "Рекомендация: пауза и быстрая проверка (есть сигналы по играм — проверить связку/сессии)."
    else:
        manager_text = "Рекомендация: отправить в СБ (высокий риск по играм — нужна ручная проверка)."

    signals = {
        "coverage_ring_games": int(coverage.get("ring_games", 0)),
        "coverage_tour_games": int(coverage.get("tour_games", 0)),

        "coplay_ring_sessions": int(cop_ring.get("sessions_count", 0)),
        "coplay_ring_unique": int(cop_ring.get("unique_opponents", 0)),
        "coplay_ring_top2_share": float(cop_ring.get("top2_coplay_share", 0.0) or 0.0),
        "coplay_ring_hu_sessions": int(cop_ring.get("hu_sessions", 0)),
        "coplay_ring_top_hu_partner": cop_ring.get("top_hu_partner"),
        "coplay_ring_top_hu_share": float(cop_ring.get("top_hu_share", 0.0) or 0.0),

        "ring_top_partner": trf_ring.get("top_net_partner"),
        "ring_net": float(trf_ring.get("top_net", 0.0) or 0.0),
        "ring_net_bb": trf_ring.get("top_net_bb", np.nan),
        "ring_gross": float(trf_ring.get("top_gross", 0.0) or 0.0),
        "ring_gross_bb": trf_ring.get("top_gross_bb", np.nan),
        "ring_partner_share": float(trf_ring.get("top_partner_share", 0.0) or 0.0),
        "ring_partner_share_bb": trf_ring.get("top_partner_share_bb", np.nan),
        "ring_pair_games_cnt": int(trf_ring.get("top_pair_games_cnt", 0) or 0),
        "ring_one_sided": float(trf_ring.get("one_sidedness", 0.0) or 0.0),
        "ring_shared_sessions": int((trf_ring.get("pair_ctx", {}) or {}).get("shared_sessions", 0) or 0),
        "ring_hu_share": float((trf_ring.get("pair_ctx", {}) or {}).get("hu_share", 0.0) or 0.0),
        "ring_dir_cons": float((trf_ring.get("pair_ctx", {}) or {}).get("dir_consistency", 0.0) or 0.0),
        "ring_shared_sessions_preview": trf_ring.get("shared_sessions_preview", []) or [],

        "db_j_total": float(db_sum.get("j_total", 0.0) or 0.0),
        "db_p_ring": float(db_sum.get("p_ring", 0.0) or 0.0),
        "db_p_mtt": float(db_sum.get("p_mtt", 0.0) or 0.0),
        "db_events_delta": float(db_sum.get("events_delta", 0.0) or 0.0),
        "db_weeks": int(db_sum.get("weeks_count", 0) or 0),
        "db_top_week_share": float(db_sum.get("top_week_share", np.nan)) if pd.notna(db_sum.get("top_week_share", np.nan)) else np.nan,
    }

    return score, decision, manager_text, reasons, signals
# =========================
# CACHE
# =========================
@st.cache_data(show_spinner=False)
def cached_load_db_multi(contents: tuple[bytes, ...], names: tuple[str, ...]):
    dfs = []
    for c, n in zip(contents, names):
        dfs.append(load_db_any(BytesFile(c, n)))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


@st.cache_data(show_spinner=True)
def cached_games_bundle_multi(contents: tuple[bytes, ...], names: tuple[str, ...]):
    all_games = []
    for c, n in zip(contents, names):
        part = parse_games_pppoker_export(BytesFile(c, n))
        if not part.empty:
            all_games.append(part)

    if not all_games:
        games_df = pd.DataFrame(
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
    else:
        games_df = pd.concat(all_games, ignore_index=True)

    sessions_df = build_sessions_from_games(games_df)
    flows_df = build_pair_flows_fast(games_df)
    idx = build_games_indexes(games_df, sessions_df, flows_df)
    return games_df, sessions_df, flows_df, idx


@st.cache_data(show_spinner=False)
def cached_top_suspicious(db_period: pd.DataFrame, idx: dict, top_n: int):
    """
    В ТОП попадают только игроки с risk_score >= T_APPROVE.
    Это защищает честных игроков от попадания в список риска при слабых/неподтверждённых сигналах.
    """
    players = sorted(db_period["_player_id"].unique().tolist())
    res = []

    for pid in players:
        db_sum, _ = db_summary_for_player(db_period, int(pid))
        if db_sum is None:
            continue

        ring_s = idx.get("player_game_series", {}).get(("RING", int(pid)))
        tour_s = idx.get("player_game_series", {}).get(("TOURNAMENT", int(pid)))
        coverage = {
            "ring_games": int(len(ring_s)) if ring_s is not None else 0,
            "tour_games": int(len(tour_s)) if tour_s is not None else 0,
        }

        cop_ring = coplay_features_fast(int(pid), idx, "RING")
        cop_tour = coplay_features_fast(int(pid), idx, "TOURNAMENT")
        trf_ring = transfer_features_fast(int(pid), idx, "RING")
        trf_tour = transfer_features_fast(int(pid), idx, "TOURNAMENT")

        score, decision, _, _, signals = score_player(db_period, db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)

        # ФИЛЬТР: честные/неподозрительные (ниже порога) в ТОП не попадают
        if int(score) < int(T_APPROVE):
            continue

        net_bb = signals.get("ring_net_bb", np.nan)
        share = signals.get("ring_partner_share_bb", np.nan)
        if pd.isna(share):
            share = signals.get("ring_partner_share", 0.0)

        res.append(
            {
                "player_id": int(pid),
                "risk_score": int(score),
                "decision": decision,
                "db_total": float(signals.get("db_j_total", 0.0)),
                "ring_games": int(signals.get("coverage_ring_games", 0)),
                "ring_sessions": int(signals.get("coplay_ring_sessions", 0)),
                "ring_hu_sessions": int(signals.get("coplay_ring_hu_sessions", 0)),
                "top_partner_ring": signals.get("ring_top_partner"),
                "net_ring": float(signals.get("ring_net", 0.0)),
                "net_ring_bb": float(net_bb) if pd.notna(net_bb) else np.nan,
                "partner_share": float(share) if pd.notna(share) else np.nan,
                "dir_cons": float(signals.get("ring_dir_cons", 0.0)),
                "one_sided": float(signals.get("ring_one_sided", 0.0)),
            }
        )

    out = pd.DataFrame(res)
    if out.empty:
        return out
    out = out.sort_values(["risk_score", "db_total"], ascending=[False, False]).head(int(top_n)).copy()
    return out


# =========================
# UI HELPERS
# =========================
def decision_badge(decision: str) -> tuple[str, str]:
    if decision == "APPROVE":
        return "МОЖНО ПРОВОДИТЬ", "green"
    if decision == "FAST_CHECK":
        return "БЫСТРАЯ ПРОВЕРКА", "orange"
    return "РУЧНАЯ ПРОВЕРКА СБ", "red"


def manager_actions(decision: str) -> list[str]:
    if decision == "APPROVE":
        return [
            "Можно проводить выплату.",
            "Если сумма крупная — выборочно открыть вкладку 'Игры' и проверить net/gross (BB) с топ‑партнёром.",
        ]
    if decision == "FAST_CHECK":
        return [
            "Поставить выплату на паузу.",
            "Проверить связку: net (BB), gross (BB), HU-долю, число совместных сессий и игр пары.",
            "Если net ≥ 50 BB или gross ≥ 250 BB (при share ≥ 60% и shared ≥ 2–3) — отправить в СБ.",
        ]
    return [
        "Не проводить выплату автоматически.",
        "Отправить в СБ: ID игрока, топ‑партнёр, net/gross (BB), shared session_id (первые 20), скрин результата.",
        "Запросить HH/историю рук (особенно HU/3-max с этим партнёром) и депозиты/кэшауты.",
    ]


def render_signal_row(label: str, value: str, status: str):
    if status == "bad":
        st.error(f"{label}: {value}")
    elif status == "warn":
        st.warning(f"{label}: {value}")
    else:
        st.success(f"{label}: {value}")


def build_manager_summary(pid: int, decision: str, score: int) -> str:
    if decision == "APPROVE":
        return f"ID {pid}: Можно проводить. Риск {score}/100. Сильных антифрод‑сигналов по текущим данным не видно."
    if decision == "FAST_CHECK":
        return f"ID {pid}: Пауза и быстрая проверка. Риск {score}/100. Есть сигналы — проверить связку/сессии."
    return f"ID {pid}: В СБ. Риск {score}/100. Высокая вероятность перелива/сговора — нужна ручная проверка."


def build_security_message(pid: int, decision: str, score: int, weeks_mode: str, week_from: int, week_to: int, signals: dict) -> str:
    ring_partner = signals.get("ring_top_partner")
    ring_net_bb = signals.get("ring_net_bb", np.nan)
    ring_net = signals.get("ring_net", 0.0)
    ring_gross_bb = signals.get("ring_gross_bb", np.nan)
    ring_shared = signals.get("ring_shared_sessions", 0)
    ring_dir = signals.get("ring_dir_cons", 0.0)
    ring_one = signals.get("ring_one_sided", 0.0)
    shared_ids = signals.get("ring_shared_sessions_preview", []) or []

    net_str = f"{ring_net_bb:.1f} BB" if pd.notna(ring_net_bb) else f"{ring_net:.2f} (BB не извлечён)"
    gross_str = f"{ring_gross_bb:.1f} BB" if pd.notna(ring_gross_bb) else f"{signals.get('ring_gross', 0.0):.2f}"

    period_str = f"{weeks_mode}"
    if weeks_mode == "Диапазон недель":
        period_str += f" (недели {week_from}–{week_to})"

    msg = []
    msg.append("ЗАПРОС НА ПРОВЕРКУ (anti-fraud)")
    msg.append(f"Игрок: {pid}")
    msg.append(f"Решение системы: {decision} / Risk score: {score}/100")
    msg.append(f"Период: {period_str}")
    msg.append("")
    msg.append("Общие данные (период):")
    msg.append(f"- '{COL_J_TOTAL}': {signals.get('db_j_total', 0.0):.2f}")
    msg.append(f"- '{COL_PLAYER_WIN_RING}': {signals.get('db_p_ring', 0.0):.2f}; '{COL_PLAYER_WIN_MTT}': {signals.get('db_p_mtt', 0.0):.2f}")
    msg.append(f"- '{COL_J_TOTAL}' - '{COL_PLAYER_WIN_TOTAL}': {signals.get('db_events_delta', 0.0):.2f}")
    msg.append("")
    msg.append("Игры (покрытие):")
    msg.append(f"- Ring игр: {signals.get('coverage_ring_games', 0)}, Tour игр: {signals.get('coverage_tour_games', 0)}")
    msg.append(
        f"- Ring HU: {signals.get('coplay_ring_hu_sessions', 0)}, top HU partner: {signals.get('coplay_ring_top_hu_partner')}, "
        f"share: {signals.get('coplay_ring_top_hu_share', 0.0)*100:.0f}%"
    )
    msg.append("")
    msg.append("Топ-пара Ring (если есть):")
    msg.append(f"- Партнёр: {ring_partner}")
    msg.append(f"- Net: {net_str}; Gross: {gross_str}")
    msg.append(f"- Shared sessions: {ring_shared}; dir: {ring_dir*100:.0f}%; one-sided: {ring_one*100:.0f}%")
    if shared_ids:
        msg.append("- Примеры session_id (первые 20):")
        msg.extend([f"  {x}" for x in shared_ids])

    return "\n".join(msg)


def copy_to_clipboard_button(text: str, label: str = "Скопировать", height: int = 46):
    safe = (text or "").replace("\\", "\\\\").replace("`", "\\`")
    html = f"""
    <div style="display:flex; gap:10px; align-items:center;">
      <button id="copy_btn" style="
        padding:8px 12px; border-radius:8px; border:1px solid #444;
        background:#111; color:#fff; cursor:pointer;">
        {label}
      </button>
      <span id="copy_status" style="font-size:12px; opacity:0.85;"></span>
    </div>
    <script>
      const textToCopy = `{safe}`;
      const btn = document.getElementById("copy_btn");
      const status = document.getElementById("copy_status");

      async function copyModern() {{
        await navigator.clipboard.writeText(textToCopy);
      }}

      function copyFallback() {{
        const ta = document.createElement("textarea");
        ta.value = textToCopy;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }}

      btn.addEventListener("click", async () => {{
        try {{
          if (navigator.clipboard && navigator.clipboard.writeText) {{
            await copyModern();
          }} else {{
            copyFallback();
          }}
          status.innerText = "Скопировано";
          setTimeout(() => status.innerText = "", 1200);
        }} catch (e) {{
          try {{
            copyFallback();
            status.innerText = "Скопировано";
            setTimeout(() => status.innerText = "", 1200);
          }} catch (e2) {{
            status.innerText = "Не удалось — скачай .txt";
          }}
        }}
      }});
    </script>
    """
    components.html(html, height=height)


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Файлы")
    st.caption("Загрузи основные файлы и при необходимости добавь дополнительные недели — они суммируются автоматически.")

    db_up = st.file_uploader("1) Общие данные (основной файл Excel/CSV)", type=["xlsx", "xls", "csv"], key="db_uploader")
    db_up_extra = st.file_uploader(
        "2) Общие данные (доп. недели, можно несколько файлов)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="db_uploader_extra",
    )

    games_up = st.file_uploader("3) Игры (основной файл TXT/CSV export)", type=["txt", "csv"], key="games_uploader")
    games_up_extra = st.file_uploader(
        "4) Игры (доп. недели, можно несколько файлов)",
        type=["txt", "csv"],
        accept_multiple_files=True,
        key="games_uploader_extra",
    )

    c1, c2, c3 = st.columns(3)
    if c1.button("Очистить DB", use_container_width=True):
        cache_clear(DB_KEY)
        st.rerun()
    if c2.button("Очистить Games", use_container_width=True):
        cache_clear(GAMES_KEY)
        st.rerun()
    if c3.button("Очистить всё", use_container_width=True):
        cache_clear(DB_KEY)
        cache_clear(GAMES_KEY)
        st.rerun()

    db_file = resolve_file(DB_KEY, db_up)
    games_file = resolve_file(GAMES_KEY, games_up)

    st.divider()
    st.header("Период из DB")
    weeks_mode = st.selectbox("Как фильтровать недели", ["Все недели", "Последние N недель", "Диапазон недель"])
    last_n = st.number_input("N (если выбран режим 'Последние N недель')", min_value=1, value=4, step=1)

if (db_file is None) and (not db_up_extra):
    st.info("Загрузи общий файл, чтобы начать проверку.")
    st.stop()

# Load DB (main + extra)
db_contents = []
db_names = []
if db_file is not None:
    db_contents.append(db_file.getvalue())
    db_names.append(getattr(db_file, "name", "db"))

for f in (db_up_extra or []):
    try:
        db_contents.append(f.getvalue())
        db_names.append(getattr(f, "name", "db_extra"))
    except Exception:
        pass

db_df = cached_load_db_multi(tuple(db_contents), tuple(db_names))
if db_df.empty:
    st.error("Не удалось загрузить 'Общие данные' (пустая таблица после обработки).")
    st.stop()

valid_weeks = sorted([w for w in db_df["_week"].unique().tolist() if w >= 0])
w_min = min(valid_weeks) if valid_weeks else 0
w_max = max(valid_weeks) if valid_weeks else 0

week_from = w_min
week_to = w_max
if weeks_mode == "Диапазон недель":
    with st.sidebar:
        week_from = st.number_input("Неделя от", value=w_min, step=1)
        week_to = st.number_input("Неделя до", value=w_max, step=1)

db_period = apply_weeks_filter(db_df, weeks_mode, int(last_n), int(week_from), int(week_to))

# Load Games bundle (optional) (main + extra)
games_df = pd.DataFrame()
sessions_df = pd.DataFrame()
flows_df = pd.DataFrame()
idx = {
    "player_game_series": {},
    "extremes": {},
    "sessions_by_player": {},
    "sessions_n": {},
    "coplay_counter": {},
    "coplay_counter_hu": {},
    "coplay_sessions_cnt": {},
    "coplay_hu_cnt": {},
    "coplay_sh_cnt": {},
    "in_map": {},
    "out_map": {},
    "flow_totals": {},
    "top_pair": {},
}

games_contents = []
games_names = []
if games_file is not None:
    games_contents.append(games_file.getvalue())
    games_names.append(getattr(games_file, "name", "games"))

for f in (games_up_extra or []):
    try:
        games_contents.append(f.getvalue())
        games_names.append(getattr(f, "name", "games_extra"))
    except Exception:
        pass

if games_contents:
    games_df, sessions_df, flows_df, idx = cached_games_bundle_multi(tuple(games_contents), tuple(games_names))

m1, m2, m3, m4 = st.columns(4, gap="small")
m1.metric("Общие данные: строк", f"{len(db_df)}", border=True)
m2.metric("Общие данные: игроков", f"{db_df['_player_id'].nunique()}", border=True)
m3.metric("Игры: строк", f"{len(games_df)}", border=True)
m4.metric("Парные потоки", f"{len(flows_df)}", border=True)

st.divider()

tab_check, tab_top = st.tabs(["Проверка игрока по ID", "Список риска (ТОП)"])

with tab_check:
    left, _ = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Ввод ID")
        default_id = int(db_period["_player_id"].iloc[0]) if len(db_period) else int(db_df["_player_id"].iloc[0])
        pid = st.number_input("ID игрока", min_value=0, value=default_id, step=1)
        run = st.button("Проверить", type="primary", use_container_width=True)

        st.divider()
        st.subheader("Как читать результат")
        st.markdown(
            "- МОЖНО ПРОВОДИТЬ: по текущим данным нет сильных сигналов перелива.\n"
            "- БЫСТРАЯ ПРОВЕРКА: есть сигналы, нужна короткая проверка деталей.\n"
            "- РУЧНАЯ ПРОВЕРКА СБ: высокий риск (перелив/сговор вероятны)."
        )

    if not run:
        st.stop()

    db_sum, by_week = db_summary_for_player(db_period, int(pid))
    if db_sum is None:
        st.error("Игрок не найден в общем файле за выбранный период.")
        st.stop()

    ring_s = idx.get("player_game_series", {}).get(("RING", int(pid)))
    tour_s = idx.get("player_game_series", {}).get(("TOURNAMENT", int(pid)))
    coverage = {
        "ring_games": int(len(ring_s)) if ring_s is not None else 0,
        "tour_games": int(len(tour_s)) if tour_s is not None else 0,
    }

    cop_ring = coplay_features_fast(int(pid), idx, "RING")
    cop_tour = coplay_features_fast(int(pid), idx, "TOURNAMENT")
    trf_ring = transfer_features_fast(int(pid), idx, "RING")
    trf_tour = transfer_features_fast(int(pid), idx, "TOURNAMENT")

    score, decision, manager_text, reasons, signals = score_player(db_period, db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)
    badge_text, badge_color = decision_badge(decision)

    st.subheader("Решение")
    cA, cB, cC = st.columns([1.4, 1, 1], gap="small")
    cA.metric("Статус", badge_text, border=True)
    cB.metric("Risk score", f"{score}/100", border=True)
    cC.metric("Период (недели)", f"{signals['db_weeks']}", border=True)

    st.progress(score / 100)

    if badge_color == "green":
        st.success(manager_text)
    elif badge_color == "orange":
        st.warning(manager_text)
    else:
        st.error(manager_text)

    st.subheader("Что делать сейчас")
    for x in manager_actions(decision):
        st.write(f"- {x}")

    st.divider()
    st.subheader("Краткое резюме по выплате")
    summary_line = build_manager_summary(int(pid), decision, int(score))
    st.info(summary_line)

    st.subheader("Сообщение для СБ (если нужно)")
    sec_text = build_security_message(
        pid=int(pid),
        decision=decision,
        score=int(score),
        weeks_mode=weeks_mode,
        week_from=int(week_from),
        week_to=int(week_to),
        signals=signals,
    )

    if decision in ("FAST_CHECK", "MANUAL_REVIEW"):
        st.text_area("Текст (можно править перед отправкой)", value=sec_text, height=260, key="sec_msg_area")
        copy_to_clipboard_button(st.session_state.get("sec_msg_area", sec_text), label="Скопировать в чат СБ")
        st.download_button(
            "Скачать как .txt",
            data=(st.session_state.get("sec_msg_area", sec_text)).encode("utf-8"),
            file_name=f"SB_check_{pid}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.caption("Сообщение для СБ скрыто, потому что решение: 'Можно проводить'.")
        if st.checkbox("Показать всё равно", value=False):
            st.text_area("Текст (можно править перед отправкой)", value=sec_text, height=260, key="sec_msg_area")
            copy_to_clipboard_button(st.session_state.get("sec_msg_area", sec_text), label="Скопировать в чат СБ")
            st.download_button(
                "Скачать как .txt",
                data=(st.session_state.get("sec_msg_area", sec_text)).encode("utf-8"),
                file_name=f"SB_check_{pid}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.divider()
    details_tab, db_tab, games_tab = st.tabs(["Объяснение (почему так)", "Общие данные (операции)", "Игры (сговор/перелив)"])

    with details_tab:
        st.subheader("Почему система так решила")
        for r in reasons[:80]:
            st.write(f"- {r}")

    with db_tab:
        meta = db_sum.get("meta", {})
        st.subheader("Профиль игрока")
        c1, c2, c3, c4 = st.columns(4, gap="small")
        c1.metric("ID", str(meta.get("player_id", "")), border=True)
        c2.metric("Ник", str(meta.get("nick", ""))[:30], border=True)
        c3.metric("Игровое имя", str(meta.get("ign", ""))[:30], border=True)
        c4.metric("Страна/регион", str(meta.get("country", ""))[:30], border=True)

        st.subheader("Операции за период")
        d1, d2, d3, d4 = st.columns(4, gap="small")
        d1.metric(f"'{COL_J_TOTAL}'", fmt_money(signals["db_j_total"]), border=True)
        d2.metric(f"'{COL_PLAYER_WIN_RING}'", fmt_money(signals["db_p_ring"]), border=True)
        d3.metric(f"'{COL_PLAYER_WIN_MTT}'", fmt_money(signals["db_p_mtt"]), border=True)
        d4.metric(f"'{COL_J_TOTAL}' - '{COL_PLAYER_WIN_TOTAL}'", fmt_money(signals["db_events_delta"]), border=True)

        st.subheader("По неделям (суммы)")
        st.dataframe(by_week.sort_values("_week", ascending=False), use_container_width=True)

    with games_tab:
        st.subheader("Сговор/перелив по Играм")
        g1, g2, g3, g4 = st.columns(4, gap="small")
        g1.metric("Ring игр", str(signals["coverage_ring_games"]), border=True)
        g2.metric("Ring сессий (co-play)", str(signals["coplay_ring_sessions"]), border=True)
        g3.metric("HU сессий (Ring)", str(signals["coplay_ring_hu_sessions"]), border=True)
        g4.metric("Top HU share", fmt_pct(signals.get("coplay_ring_top_hu_share", 0.0)), border=True)

        st.subheader("Топ‑пара (Ring)")
        if signals["ring_top_partner"] is None:
            st.info("Топ‑пара не выделилась по текущей выборке.")
        else:
            share = signals.get("ring_partner_share_bb", np.nan)
            if pd.isna(share):
                share = signals.get("ring_partner_share", 0.0)

            rows = [
                {"Показатель": "Партнёр (ID)", "Значение": str(signals["ring_top_partner"])},
                {"Показатель": "Совместных сессий", "Значение": str(signals["ring_shared_sessions"])},
                {"Показатель": "Игр пары", "Значение": str(signals["ring_pair_games_cnt"])},
                {"Показатель": "Net-flow", "Значение": fmt_bb(signals["ring_net_bb"]) if pd.notna(signals["ring_net_bb"]) else fmt_money(signals["ring_net"])},
                {"Показатель": "Gross (оборот пары)", "Значение": fmt_bb(signals["ring_gross_bb"]) if pd.notna(signals["ring_gross_bb"]) else fmt_money(signals["ring_gross"])},
                {"Показатель": "Доля оборота с партнёром", "Значение": fmt_pct(share)},
                {"Показатель": "HU доля внутри пары", "Значение": fmt_pct(signals["ring_hu_share"])},
                {"Показатель": "Повторяемость направления (dir)", "Значение": fmt_pct(signals["ring_dir_cons"])},
                {"Показатель": "Односторонность потоков", "Значение": fmt_pct(signals["ring_one_sided"])},
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with tab_top:
    st.subheader("Список риска (ТОП)")
    st.caption(f"Показываются только игроки с risk_score ≥ {T_APPROVE}.")

    colA, colB = st.columns([1, 1])
    with colA:
        top_n = st.number_input("Сколько показать", min_value=10, max_value=300, value=50, step=10)
    with colB:
        build = st.button("Построить ТОП", type="primary", use_container_width=True)

    if not build:
        st.stop()

    top_df = cached_top_suspicious(db_period, idx, int(top_n))
    if top_df.empty:
        st.info("Нет игроков выше порога риска за выбранный период.")
        st.stop()

    show = top_df.copy()
    show["partner_share"] = show["partner_share"].apply(lambda x: f"{float(x)*100:.0f}%" if pd.notna(x) else "NaN")
    show["dir_cons"] = show["dir_cons"].apply(lambda x: f"{float(x)*100:.0f}%" if pd.notna(x) else "NaN")
    show["one_sided"] = show["one_sided"].apply(lambda x: f"{float(x)*100:.0f}%" if pd.notna(x) else "NaN")

    st.dataframe(show, use_container_width=True)
    csv_bytes = top_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Скачать ТОП (CSV)", data=csv_bytes, file_name="top_risk.csv", mime="text/csv")
