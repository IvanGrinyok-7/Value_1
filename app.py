import io
import re
import json
import datetime as dt
import traceback
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

def decision_badge(decision: str) -> tuple[str, str]:
    if decision == "APPROVE":
        return "✅ APPROVE", "green"
    if decision == "FAST_CHECK":
        return "🟠 FAST CHECK", "orange"
    return "🔴 MANUAL REVIEW", "red"

def manager_actions(decision: str) -> list[str]:
    if decision == "APPROVE":
        return ["Разрешить вывод.", "Если сумма крупная — выборочно проверить 1–2 последние сессии."]
    if decision == "FAST_CHECK":
        return ["Быстро проверить топ‑партнёра: shared sessions + net/gross + HU.", "Если подтверждается перелив — в СБ."]
    return ["Отправить в СБ на ручную проверку.", "Проверить HH по ключевым сессиям (особенно HU/3‑max)."]

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
                return header.index(col) if col in header else None

            # ---------- FIX: never use `or` with indices ----------
            win_idx = find("Выигрыш")
            if win_idx is None:
                win_idx = find("Выигрыш игрока")
            # -----------------------------------------------------

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
# INDEXES (ваш текущий подход)
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
        gt = str(gt); pid = int(pid); partner = int(partner)
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
# SCORING (ваша логика, без падений)
# =========================
def score_player_period(db_df: pd.DataFrame, db_sum: dict, cop_ring: dict, cop_tour: dict, trf_ring: dict, trf_tour: dict, coverage: dict):
    score = 0
    reasons = []

    pid = int(db_sum["meta"]["player_id"])

    j_tot = float(db_sum.get("j_total", 0.0) or 0.0)
    events_delta = float(db_sum.get("events_delta", 0.0) or 0.0)
    weeks_cnt = int(db_sum.get("weeks_count", 0) or 0)
    top_week_share = db_sum.get("top_week_share", np.nan)

    if j_tot >= 800:
        score += 8
        reasons.append(f"DB: крупный плюс по '{COL_J_TOTAL}' (выплаты обычно идут с плюсовых).")
    elif j_tot >= 300:
        score += 4
        reasons.append(f"DB: заметный плюс по '{COL_J_TOTAL}' (контроль усилен).")

    if abs(events_delta) >= max(80.0, 0.35 * max(1.0, abs(j_tot))):
        score += 6
        reasons.append(f"DB: большая разница '{COL_J_TOTAL}' и '{COL_PLAYER_WIN_TOTAL}' — много 'событий' (джекпот/эквити).")

    if weeks_cnt >= 3 and pd.notna(top_week_share) and float(top_week_share) >= 0.80 and abs(j_tot) >= 300:
        score += 5
        reasons.append("DB: сильная концентрация результата в одной неделе.")

    ring_over_comm_ppsr = db_sum.get("ring_over_comm_ppsr", np.nan)
    p_ring = float(db_sum.get("p_ring", 0.0) or 0.0)
    comm_ppsr = float(db_sum.get("comm_ppsr", 0.0) or 0.0)
    if pd.notna(ring_over_comm_ppsr) and comm_ppsr >= 10.0 and float(ring_over_comm_ppsr) >= 10 and abs(p_ring) >= 200:
        score += 5
        reasons.append(f"DB: '{COL_PLAYER_WIN_RING}' слишком высок относительно '{COL_CLUB_COMM_PPSR}'.")

    ring_games = int(coverage.get("ring_games", 0))
    tour_games = int(coverage.get("tour_games", 0))
    if ring_games + tour_games == 0:
        reasons.append("GAMES: нет покрытия по играм (файл не загружен/игрок не найден/парсинг не смог).")
    else:
        reasons.append(f"GAMES: покрытие есть (Ring игр {ring_games}, турниров {tour_games}).")

    # Co-play Ring
    if cop_ring["sessions_count"] >= MIN_SESSIONS_FOR_COPLAY:
        if cop_ring["sessions_count"] >= 6 and cop_ring["unique_opponents"] <= 5:
            score += 6
            reasons.append("GAMES/RING: узкий пул оппонентов при достаточном числе сессий.")

        if cop_ring["top2_coplay_share"] >= COPLAY_TOP2_SHARE_SUSP and cop_ring["sessions_count"] >= 6:
            score += 6
            reasons.append("GAMES/RING: топ‑2 оппонента встречаются почти всегда (связка).")

        hu_ratio = cop_ring["hu_sessions"] / max(1, cop_ring["sessions_count"])
        if cop_ring["hu_sessions"] >= HU_DOMINANCE_MIN_HU and hu_ratio >= HU_DOMINANCE_RATIO:
            score += 8
            reasons.append("GAMES/RING: доминирует HU (типичный формат перелива).")

        if cop_ring.get("top_hu_partner") is not None and cop_ring["hu_sessions"] >= HU_DOMINANCE_MIN_HU and cop_ring.get("top_hu_share", 0.0) >= 0.80:
            score += 8
            reasons.append(f"GAMES/RING: HU почти всегда с одним партнёром ({cop_ring['top_hu_partner']}).")

    def check_flow(trf: dict, label: str):
        nonlocal score, reasons
        partner = trf.get("top_net_partner")
        if partner is None:
            return

        net_val = float(trf.get("top_net", 0.0) or 0.0)
        net_bb = trf.get("top_net_bb", np.nan)
        gross_val = float(trf.get("top_gross", 0.0) or 0.0)
        gross_bb = trf.get("top_gross_bb", np.nan)
        pair_games_cnt = int(trf.get("top_pair_games_cnt", 0) or 0)

        pshare_bb = trf.get("top_partner_share_bb", np.nan)
        partner_share = float(pshare_bb) if pd.notna(pshare_bb) else float(trf.get("top_partner_share", 0.0) or 0.0)

        ctx = trf.get("pair_ctx", {}) or {}
        shared = int(ctx.get("shared_sessions", 0) or 0)
        dir_cons = float(ctx.get("dir_consistency", 0.0) or 0.0)
        hu_share = float(ctx.get("hu_share", 0.0) or 0.0)
        one_sided = float(trf.get("one_sidedness", 0.0) or 0.0)

        enough = (shared >= PAIR_MIN_SHARED_SESSIONS_STRONG) or (pair_games_cnt >= PAIR_MIN_PAIR_GAMES_STRONG)

        if label == "RING":
            # net (BB)
            if pd.notna(net_bb):
                abs_net_bb = abs(float(net_bb))
                if abs_net_bb >= PAIR_NET_CRITICAL_RING_BB:
                    score += 55
                    reasons.append(f"GAMES/RING: критичный net-flow {fmt_bb(net_bb)} с партнёром {partner}.")
                elif abs_net_bb >= PAIR_NET_ALERT_RING_BB and enough:
                    score += 25
                    reasons.append(f"GAMES/RING: net-flow {fmt_bb(net_bb)} с партнёром {partner} (enough shared/games).")
            else:
                if abs(net_val) >= 120:
                    score += 45
                    reasons.append(f"GAMES/RING: net-flow {fmt_money(net_val)} с партнёром {partner} (нет BB, fallback).")
                elif abs(net_val) >= 60 and enough:
                    score += 18
                    reasons.append(f"GAMES/RING: net-flow {fmt_money(net_val)} с партнёром {partner} (fallback).")

            # turnover (gross BB)
            if pd.notna(gross_bb):
                gbb = float(gross_bb)
                if gbb >= PAIR_GROSS_CRITICAL_RING_BB and shared >= 3 and partner_share >= PAIR_PARTNER_SHARE_ALERT:
                    score += 25
                    reasons.append(f"GAMES/RING: очень высокий gross {fmt_bb(gross_bb)} с {partner} при доле {fmt_pct(partner_share)}.")
                elif gbb >= PAIR_GROSS_ALERT_RING_BB and enough and partner_share >= PAIR_PARTNER_SHARE_ALERT:
                    score += 12
                    reasons.append(f"GAMES/RING: высокий gross {fmt_bb(gross_bb)} с {partner} при доле {fmt_pct(partner_share)}.")

            # patterns
            if enough and partner_share >= PAIR_PARTNER_SHARE_ALERT and (one_sided >= PAIR_ONE_SIDED_ALERT or dir_cons >= PAIR_DIR_CONSIST_ALERT or hu_share >= 0.70):
                score += 12
                reasons.append(f"GAMES/RING: one-sided/dir/HU паттерн с {partner} (share {fmt_pct(partner_share)}).")

            # add agent/super-agent bonus only if already suspicious
            if score >= T_APPROVE:
                bonus, txt = agent_match_bonus(db_df, int(partner), pid)
                if bonus > 0 and txt:
                    score += bonus
                    reasons.append(f"DB: {txt}")

        else:
            # TOURNAMENT
            if abs(net_val) >= PAIR_NET_ALERT_TOUR and enough:
                score += 12
                reasons.append(f"GAMES/TOUR: net-flow {fmt_money(net_val)} с {partner}.")
            if gross_val >= PAIR_GROSS_ALERT_TOUR and enough and partner_share >= 0.60:
                score += 8
                reasons.append(f"GAMES/TOUR: gross-turnover {fmt_money(gross_val)} с {partner} (share {fmt_pct(partner_share)}).")

    check_flow(trf_ring, "RING")
    check_flow(trf_tour, "TOURNAMENT")

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)

    signals = {
        "player_id": pid,
        "coverageringgames": int(coverage.get("ring_games", 0)),
        "coveragetourgames": int(coverage.get("tour_games", 0)),
        "coplayringsessions": int(cop_ring.get("sessions_count", 0)),
        "coplayringunique": int(cop_ring.get("unique_opponents", 0)),
        "coplayringtop2share": float(cop_ring.get("top2_coplay_share", 0.0) or 0.0),
        "coplayringhusessions": int(cop_ring.get("hu_sessions", 0)),
        "coplayringtophupartner": cop_ring.get("top_hu_partner"),
        "coplayringtophushare": float(cop_ring.get("top_hu_share", 0.0) or 0.0),
        "ringtoppartner": trf_ring.get("top_net_partner"),
        "ringnet": float(trf_ring.get("top_net", 0.0) or 0.0),
        "ringnetbb": trf_ring.get("top_net_bb", np.nan),
        "ringgross": float(trf_ring.get("top_gross", 0.0) or 0.0),
        "ringgrossbb": trf_ring.get("top_gross_bb", np.nan),
        "ringpartnershare": float(trf_ring.get("top_partner_share", 0.0) or 0.0),
        "ringpartnersharebb": trf_ring.get("top_partner_share_bb", np.nan),
        "ringpairgamescnt": int(trf_ring.get("top_pair_games_cnt", 0) or 0),
        "ringonesided": float(trf_ring.get("one_sidedness", 0.0) or 0.0),
        "ringsharedsessions": int((trf_ring.get("pair_ctx", {}) or {}).get("shared_sessions", 0) or 0),
        "ringhushare": float((trf_ring.get("pair_ctx", {}) or {}).get("hu_share", 0.0) or 0.0),
        "ringdircons": float((trf_ring.get("pair_ctx", {}) or {}).get("dir_consistency", 0.0) or 0.0),
        "ringsharedsessionspreview": trf_ring.get("shared_sessions_preview", []) or [],
        "dbjtotal": float(db_sum.get("j_total", 0.0) or 0.0),
        "dbpring": float(db_sum.get("p_ring", 0.0) or 0.0),
        "dbpmtt": float(db_sum.get("p_mtt", 0.0) or 0.0),
        "dbeventsdelta": float(db_sum.get("events_delta", 0.0) or 0.0),
        "dbweeks": int(db_sum.get("weeks_count", 0) or 0),
        "dbtopweekshare": float(db_sum.get("top_week_share")) if pd.notna(db_sum.get("top_week_share", np.nan)) else np.nan,
    }

    manager_text = (
        "Низкий риск по данным (разрешить вывод)." if decision == "APPROVE"
        else "Есть подозрительные сигналы (быстрый чек/при сомнениях — СБ)." if decision == "FAST_CHECK"
        else "Высокий риск (СБ/ручная проверка)."
    )
    return score, decision, manager_text, reasons, signals

def build_security_message(pid: int, decision: str, score: int, weeks_mode: str, week_from: int, week_to: int, signals: dict) -> str:
    period = weeks_mode if weeks_mode != "Период (от‑до)" else f"Период {week_from}–{week_to}"
    msg = []
    msg.append("ANTI-FRAUD CHECK (PPPoker)")
    msg.append(f"Player ID: {pid}")
    msg.append(f"Decision: {decision} | Risk score: {score}/100")
    msg.append(f"Period: {period}")
    msg.append("")
    msg.append("DB:")
    msg.append(f"- J total: {fmt_money(signals.get('dbjtotal', 0.0))}")
    msg.append(f"- Ring: {fmt_money(signals.get('dbpring', 0.0))} | MTT: {fmt_money(signals.get('dbpmtt', 0.0))}")
    msg.append(f"- Events delta: {fmt_money(signals.get('dbeventsdelta', 0.0))}")
    if pd.notna(signals.get("dbtopweekshare", np.nan)):
        msg.append(f"- Top week share: {fmt_pct(signals.get('dbtopweekshare', 0.0))}")
    msg.append("")
    msg.append("GAMES (Ring):")
    msg.append(f"- Ring games: {signals.get('coverageringgames', 0)} | Tour games: {signals.get('coveragetourgames', 0)}")
    msg.append(f"- HU sessions: {signals.get('coplayringhusessions', 0)} | Top HU partner: {signals.get('coplayringtophupartner', None)} | Top HU share: {fmt_pct(signals.get('coplayringtophushare', 0.0))}")
    msg.append(f"- Top partner: {signals.get('ringtoppartner', None)}")
    netbb = signals.get("ringnetbb", np.nan)
    grossbb = signals.get("ringgrossbb", np.nan)
    if pd.notna(netbb):
        msg.append(f"- Net (BB): {fmt_bb(netbb)} | Gross (BB): {fmt_bb(grossbb) if pd.notna(grossbb) else 'NaN'}")
    else:
        msg.append(f"- Net: {fmt_money(signals.get('ringnet', 0.0))} | Gross: {fmt_money(signals.get('ringgross', 0.0))}")
    msg.append(f"- Partner share: {fmt_pct(signals.get('ringpartnershare', 0.0))}")
    msg.append(f"- Shared sessions: {signals.get('ringsharedsessions', 0)} | HU-share in pair: {fmt_pct(signals.get('ringhushare', 0.0))} | Dir-cons: {fmt_pct(signals.get('ringdircons', 0.0))}")
    if signals.get("ringsharedsessionspreview"):
        msg.append("- Shared session ids (preview): " + ", ".join([str(x) for x in signals.get("ringsharedsessionspreview", [])]))
    return "\n".join(msg)

# =========================
# Cached loaders
# =========================
@st.cache_data(show_spinner=False)
def cached_load_db_multi(contents: tuple[bytes, ...], names: tuple[str, ...]) -> pd.DataFrame:
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
        games_df = pd.DataFrame(columns=[
            "game_id", "game_type", "product", "table_name", "descriptor", "bb", "start_time", "end_time",
            "player_id", "nick", "ign", "hands", "win_total", "win_vs_opponents", "fee"
        ])
    else:
        games_df = pd.concat(all_games, ignore_index=True)

    sessions_df = build_sessions_from_games(games_df)
    flows_df = build_pair_flows_fast(games_df)
    idx = build_games_indexes(games_df, sessions_df, flows_df)
    return games_df, sessions_df, flows_df, idx

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Загрузка")
    st.caption("Файлы сохраняются в кэше сервиса (persist).")

    db_up = st.file_uploader("1) Общее (Excel/CSV)", type=["xlsx", "xls", "csv"], key="db_uploader")
    db_up_extra = st.file_uploader("2) Общее (доп. недели)", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="db_uploader_extra")

    games_up = st.file_uploader("3) Игры (TXT/CSV export)", type=["txt", "csv"], key="games_uploader")
    games_up_extra = st.file_uploader("4) Игры (доп. файлы)", type=["txt", "csv"], accept_multiple_files=True, key="games_uploader_extra")

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

    st.divider()
    st.header("Период (DB)")
    weeksmode = st.selectbox("Режим", ["Все недели", "Последние N недель", "Период (от‑до)"], index=1)
    lastn = st.number_input("N", min_value=1, value=4, step=1)
    weekfrom = st.number_input("От недели", value=1, step=1)
    weekto = st.number_input("До недели", value=1, step=1)

# persist main db/games
dbfile = resolve_file(DB_KEY, db_up)
gamesfile = resolve_file(GAMES_KEY, games_up)

# collect DB files
db_contents = []
db_names = []
if dbfile is not None:
    db_contents.append(dbfile.getvalue())
    db_names.append(getattr(dbfile, "name", "db"))

for f in (db_up_extra or []):
    try:
        db_contents.append(f.getvalue())
        db_names.append(getattr(f, "name", "db_extra"))
    except Exception:
        pass

if not db_contents:
    st.info("Загрузите файл 'Общее' (DB).")
    st.stop()

# load DB
try:
    dbdf = cached_load_db_multi(tuple(db_contents), tuple(db_names))
except Exception as e:
    st.error(f"Ошибка чтения DB: {e}")
    st.code(traceback.format_exc())
    st.stop()

if dbdf.empty:
    st.error("DB пустая или не распарсилась.")
    st.stop()

# period filter
valid_weeks = sorted([int(w) for w in dbdf["_week"].unique().tolist() if int(w) >= 0])
wmin = min(valid_weeks) if valid_weeks else 0
wmax = max(valid_weeks) if valid_weeks else 0
if weeksmode == "Период (от‑до)" and weekfrom == 1 and weekto == 1 and wmax >= wmin:
    weekfrom, weekto = wmin, wmax

dbperiod = apply_weeks_filter(dbdf, weeksmode, int(lastn), int(weekfrom), int(weekto))

# collect games files
games_contents = []
games_names = []
if gamesfile is not None:
    games_contents.append(gamesfile.getvalue())
    games_names.append(getattr(gamesfile, "name", "games"))
for f in (games_up_extra or []):
    try:
        games_contents.append(f.getvalue())
        games_names.append(getattr(f, "name", "games_extra"))
    except Exception:
        pass

games_df = pd.DataFrame()
sessions_df = pd.DataFrame()
flows_df = pd.DataFrame()
idx = {}

if games_contents:
    try:
        games_df, sessions_df, flows_df, idx = cached_games_bundle_multi(tuple(games_contents), tuple(games_names))
    except Exception as e:
        st.error(f"Ошибка чтения/парсинга Games: {e}")
        st.code(traceback.format_exc())
        st.stop()
else:
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

# header metrics
m1, m2, m3, m4 = st.columns(4, gap="small")
m1.metric("DB rows", f"{len(dbdf)}", border=True)
m2.metric("Players in DB", f"{dbdf['_player_id'].nunique()}", border=True)
m3.metric("Games rows", f"{len(games_df)}", border=True)
m4.metric("Flows pairs", f"{len(flows_df)}", border=True)

st.divider()

tab_check, tab_top = st.tabs(["Проверка ID", "Топ риск"])

with tab_check:
    left, right = st.columns([1.0, 1.8], gap="large")

    with left:
        st.subheader("Проверка")
        default_id = int(dbperiod["_player_id"].iloc[0]) if len(dbperiod) else int(dbdf["_player_id"].iloc[0])
        pid = st.number_input("ID игрока", min_value=0, value=int(default_id), step=1)
        run = st.button("Проверить", type="primary", use_container_width=True)

    with right:
        if not run:
            st.info("Введите ID и нажмите «Проверить».")
            st.stop()

        try:
            dbsum, byweek = db_summary_for_player(dbperiod, int(pid))
            if dbsum is None:
                st.error("Игрок не найден в DB за выбранный период.")
                st.stop()

            rings = idx.get("player_game_series", {}).get(("RING", int(pid)))
            tours = idx.get("player_game_series", {}).get(("TOURNAMENT", int(pid)))
            coverage = {
                "ring_games": int(len(rings)) if rings is not None else 0,
                "tour_games": int(len(tours)) if tours is not None else 0,
            }

            copring = coplay_features_fast(int(pid), idx, "RING")
            coptour = coplay_features_fast(int(pid), idx, "TOURNAMENT")
            trfring = transfer_features_fast(int(pid), idx, "RING")
            trftour = transfer_features_fast(int(pid), idx, "TOURNAMENT")

            score, decision, manager_text, reasons, signals = score_player_period(
                dbdf=dbdf,
                db_sum=dbsum,
                cop_ring=copring,
                cop_tour=coptour,
                trf_ring=trfring,
                trf_tour=trftour,
                coverage=coverage,
            )

        except Exception as e:
            st.error(f"Ошибка при проверке ID: {e}")
            st.code(traceback.format_exc())
            st.stop()

        badge_text, badge_color = decision_badge(decision)
        cA, cB, cC = st.columns([1.4, 1.0, 1.0], gap="small")
        cA.metric("Решение", badge_text, border=True)
        cB.metric("Risk", f"{score}/100", border=True)
        cC.metric("DB weeks", f"{signals.get('dbweeks', 0)}", border=True)

        if badge_color == "green":
            st.success(manager_text)
        elif badge_color == "orange":
            st.warning(manager_text)
        else:
            st.error(manager_text)

        st.subheader("Причины")
        for r in reasons[:60]:
            st.write(f"- {r}")

        st.divider()
        st.subheader("Сообщение для СБ")
        sec_text = build_security_message(int(pid), decision, int(score), weeksmode, int(weekfrom), int(weekto), signals)
        st.textarea("Скопировать", value=sec_text, height=260)
        st.download_button(
            "Скачать .txt",
            data=sec_text.encode("utf-8"),
            file_name=f"SB_check_{int(pid)}.txt",
            mime="text/plain",
            use_container_width=True,
        )

with tab_top:
    st.subheader("Топ риск за период (по DB)")
    topn = st.number_input("Top N", min_value=10, max_value=300, value=50, step=10)

    @st.cache_data(show_spinner=True)
    def cached_top_risk(dbperiod_in: pd.DataFrame, idx_in: dict, topn_in: int):
        players = sorted(dbperiod_in["_player_id"].unique().tolist())
        res = []
        for pid in players:
            dbsum, _ = db_summary_for_player(dbperiod_in, int(pid))
            if dbsum is None:
                continue
            rings = idx_in.get("player_game_series", {}).get(("RING", int(pid)))
            tours = idx_in.get("player_game_series", {}).get(("TOURNAMENT", int(pid)))
            coverage = {"ring_games": int(len(rings)) if rings is not None else 0, "tour_games": int(len(tours)) if tours is not None else 0}
            copring = coplay_features_fast(int(pid), idx_in, "RING")
            coptour = coplay_features_fast(int(pid), idx_in, "TOURNAMENT")
            trfring = transfer_features_fast(int(pid), idx_in, "RING")
            trftour = transfer_features_fast(int(pid), idx_in, "TOURNAMENT")

            score, decision, _mgr, reasons, signals = score_player_period(
                dbdf=dbdf,
                db_sum=dbsum,
                cop_ring=copring,
                cop_tour=coptour,
                trf_ring=trfring,
                trf_tour=trftour,
                coverage=coverage,
            )
            res.append({
                "player_id": int(pid),
                "risk_score": int(score),
                "decision": decision,
                "db_total": float(signals.get("dbjtotal", 0.0)),
                "ring_games": int(signals.get("coverageringgames", 0)),
                "ring_hu_sessions": int(signals.get("coplayringhusessions", 0)),
                "top_partner": signals.get("ringtoppartner", None),
                "top_reason": (reasons[0] if reasons else ""),
            })
        out = pd.DataFrame(res)
        if out.empty:
            return out
        return out.sort_values(["risk_score", "db_total"], ascending=[False, False]).head(int(topn_in)).copy()

    topdf = cached_top_risk(dbperiod, idx, int(topn))
    if topdf.empty:
        st.info("Пусто.")
    else:
        st.dataframe(topdf, use_container_width=True, hide_index=True)
        csvbytes = topdf.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Скачать CSV", data=csvbytes, file_name="top_risk.csv", mime="text/csv", use_container_width=True)
