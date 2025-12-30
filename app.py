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
APP_TITLE = "PPPoker | аналитика рисков — anti-fraud (chip dumping / collusion)"
CACHE_DIR = Path(".pppoker_app_cache")
CACHE_DIR.mkdir(exist_ok=True)

DB_KEY = "db_file"
GAMES_KEY = "games_file"

# --- RATING THRESHOLDS ---
T_APPROVE = 25
T_FAST_CHECK = 55

# --- COPLAY ---
MIN_SESSIONS_FOR_COPLAY = 1
COPLAY_TOP2_SHARE_SUSP = 0.80

# --- RING thresholds in BB (только для надёжных HU/3-max flows) ---
PAIR_NET_ALERT_RING_BB = 20.0
PAIR_NET_CRITICAL_RING_BB = 40.0
PAIR_GROSS_ALERT_RING_BB = 80.0

PAIR_ONE_SIDED_ALERT = 0.85
PAIR_DIR_CONSIST_ALERT = 0.78
PAIR_PARTNER_SHARE_ALERT = 0.55
PAIR_MIN_SHARED_SESSIONS_STRONG = 1

# --- 3-max reliability (консервативно) ---
MAX_TABLE_SIZE_FOR_PAIR_INFER = 3
WEIGHT_3MAX_FLOWS = 0.35  # уменьшает вклад 3-max “инференса” в net/gross

# --- Tournament thresholds (валюта) ---
PAIR_NET_ALERT_TOUR = 50.0
SINGLE_GAME_WIN_ALERT_TOUR = 150.0

# Regex PPPoker export
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)
TABLE_NAME_RE = re.compile(r"Название стола:\s*(.+?)\s*$", re.IGNORECASE)
START_END_RE = re.compile(r"Начало:\s*([0-9/:\\s]+)\s+By.+?Окончание:\s*([0-9/:\\s]+)", re.IGNORECASE)

RING_HINT_RE = re.compile(r"\bPPSR\b|PLO|OFC|NLH|Bomb Pot|Ante|3-1|HU\b|Heads", re.IGNORECASE)
TOUR_HINT_RE = re.compile(r"\bPPST\b|Бай-ин:|satellite|pko|mko\b", re.IGNORECASE)

STAKES_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)")

# DB columns
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


def clamp01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0


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
def _classify_game_type(descriptor: str) -> str:
    if not descriptor:
        return "UNKNOWN"
    if TOUR_HINT_RE.search(descriptor):
        return "TOURNAMENT"
    if RING_HINT_RE.search(descriptor):
        return "RING"
    return "UNKNOWN"


def _extract_bb(descriptor: str) -> float:
    if not descriptor:
        return np.nan
    m = STAKES_RE.search(descriptor.replace(",", "."))
    if not m:
        return np.nan
    try:
        bb = float(m.group(2))
        return bb if bb > 0 else np.nan
    except Exception:
        return np.nan


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

        if ("PPSR" in line or "PPST" in line) and ("ID игрока" not in line) and ("Итог" not in line):
            current["descriptor"] = line.strip()
            current["game_type"] = _classify_game_type(current["descriptor"])
            current["bb"] = _extract_bb(current["descriptor"]) if current["game_type"] == "RING" else np.nan
            current["product"] = "PPSR" if "PPSR" in line else ("PPST" if "PPST" in line else "")
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
        return pd.DataFrame(columns=[
            "game_id", "game_type", "product", "table_name", "descriptor", "bb", "start_time", "end_time",
            "player_id", "nick", "ign", "hands", "win_total", "win_vs_opponents", "fee"
        ])

    for c in ["bb", "hands", "win_total", "win_vs_opponents", "fee"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["player_id", "game_id", "game_type"]).copy()
    df["player_id"] = df["player_id"].astype(int)
    df["game_id"] = df["game_id"].astype(str)

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
# FLOWS (консервативно)
# =========================
def _flow_win_col(g: pd.DataFrame) -> pd.Series:
    # Ring: win_vs_opponents (если есть) лучше для “переводов” чем общий итог
    return np.where(
        (g["game_type"] == "RING") & g["win_vs_opponents"].notna(),
        g["win_vs_opponents"],
        g["win_total"],
    )


def build_pair_flows_conservative(games_df: pd.DataFrame, sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Строим парные переводы только там, где это относительно надёжно:
    - HU (2 игрока): перевод однозначен
    - 3-max: перевод оценочный, с пониженным весом
    - 4+: не строим парные переводы (иначе будет много false positive)
    """
    if games_df.empty or sessions_df.empty:
        return pd.DataFrame(columns=["from_player", "to_player", "game_type", "amount", "amount_bb", "games_cnt"])

    s = sessions_df.rename(columns={"session_id": "game_id"})[["game_id", "game_type", "players_n"]].copy()
    g = games_df.merge(s, on=["game_id", "game_type"], how="left")

    g = g[g["players_n"].notna()].copy()
    g["players_n"] = g["players_n"].astype(int)

    g["_flow_win"] = _flow_win_col(g)
    g = g[g["_flow_win"].notna()].copy()
    g["_flow_win"] = pd.to_numeric(g["_flow_win"], errors="coerce")
    g = g[g["_flow_win"].notna()].copy()

    g = g[g["players_n"] <= MAX_TABLE_SIZE_FOR_PAIR_INFER].copy()
    if g.empty:
        return pd.DataFrame(columns=["from_player", "to_player", "game_type", "amount", "amount_bb", "games_cnt"])

    out_rows = []

    for (gid, gt), part in g.groupby(["game_id", "game_type"], sort=False):
        pn = int(part["players_n"].max())
        part = part[["player_id", "_flow_win", "bb"]].copy()
        if part["player_id"].nunique() < 2:
            continue

        winners = part[part["_flow_win"] > 0]
        losers = part[part["_flow_win"] < 0]
        if winners.empty or losers.empty:
            continue

        bb = float(part["bb"].max()) if gt == "RING" else np.nan
        bb_ok = (bb > 0) if gt == "RING" else False

        if pn == 2:
            # точный HU-перевод
            w = winners.sort_values("_flow_win", ascending=False).iloc[0]
            l = losers.sort_values("_flow_win", ascending=True).iloc[0]
            amt = float(w["_flow_win"])
            if amt <= 0:
                continue
            out_rows.append({
                "from_player": int(l["player_id"]),
                "to_player": int(w["player_id"]),
                "game_type": str(gt),
                "amount": amt,
                "amount_bb": float(amt / bb) if bb_ok else np.nan,
                "games_cnt": 1,
                "table_size": 2,
            })
        elif pn == 3:
            # осторожный 3-max “инференс”: распределяем проигрыш по победителям, но уменьшаем вес
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
                amts = (loss * win_w) * WEIGHT_3MAX_FLOWS
                for wpid, amt in zip(win_pids, amts):
                    wpid = int(wpid)
                    if amt <= 0:
                        continue
                    out_rows.append({
                        "from_player": lpid,
                        "to_player": wpid,
                        "game_type": str(gt),
                        "amount": float(amt),
                        "amount_bb": float(amt / bb) if bb_ok else np.nan,
                        "games_cnt": 1,
                        "table_size": 3,
                    })

    if not out_rows:
        return pd.DataFrame(columns=["from_player", "to_player", "game_type", "amount", "amount_bb", "games_cnt"])

    f = pd.DataFrame(out_rows)

    f = f.groupby(["from_player", "to_player", "game_type"], as_index=False).agg(
        amount=("amount", "sum"),
        amount_bb=("amount_bb", "sum"),
        games_cnt=("games_cnt", "sum"),
        min_table_size=("table_size", "min"),
        max_table_size=("table_size", "max"),
    )

    return f


# =========================
# INDEXES (fast lookups)
# =========================
def build_games_indexes(games_df: pd.DataFrame, sessions_df: pd.DataFrame, flows_df: pd.DataFrame):
    idx = {}

    # series per player (для общих экстремумов)
    if games_df.empty:
        idx["player_game_series"] = {}
        idx["extremes"] = {}
    else:
        d = games_df[["game_type", "player_id", "game_id", "win_total", "win_vs_opponents"]].copy()
        d["_flow_win"] = _flow_win_col(d)
        d = d[d["_flow_win"].notna()].copy()
        d["_flow_win"] = pd.to_numeric(d["_flow_win"], errors="coerce")
        d = d[d["_flow_win"].notna()].copy()
        d["player_id"] = d["player_id"].astype(int)
        d["game_id"] = d["game_id"].astype(str)

        series = {}
        extremes = {}
        for (gt, pid), g in d.groupby(["game_type", "player_id"], sort=False):
            s = pd.Series(g["_flow_win"].to_numpy(dtype=float), index=g["game_id"].to_numpy())
            series[(str(gt), int(pid))] = s
            if str(gt) == "TOURNAMENT":
                big = s[s >= SINGLE_GAME_WIN_ALERT_TOUR]
                extremes[(str(gt), int(pid))] = list(big.index[:12]) if not big.empty else []
            else:
                extremes[(str(gt), int(pid))] = []
        idx["player_game_series"] = series
        idx["extremes"] = extremes

    # sessions inverted + coplay counters
    sessions_by_player = defaultdict(list)
    sessions_n = {}
    coplay_counter = defaultdict(lambda: defaultdict(int))
    coplay_sessions_cnt = defaultdict(int)
    coplay_hu_cnt = defaultdict(int)
    coplay_sh_cnt = defaultdict(int)

    if not sessions_df.empty:
        for sid, gt, players, pn in sessions_df[["session_id", "game_type", "players", "players_n"]].itertuples(index=False):
            sid = str(sid)
            gt = str(gt)
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

    idx["sessions_by_player"] = dict(sessions_by_player)
    idx["sessions_n"] = sessions_n
    idx["coplay_counter"] = {k: dict(v) for k, v in coplay_counter.items()}
    idx["coplay_sessions_cnt"] = dict(coplay_sessions_cnt)
    idx["coplay_hu_cnt"] = dict(coplay_hu_cnt)
    idx["coplay_sh_cnt"] = dict(coplay_sh_cnt)

    # flows -> maps + totals + top pair
    if flows_df.empty:
        idx["in_map"] = {}
        idx["out_map"] = {}
        idx["flow_totals"] = {}
        idx["top_pair"] = {}
        return idx

    f = flows_df.copy()
    f["from_player"] = f["from_player"].astype(int)
    f["to_player"] = f["to_player"].astype(int)
    f["game_type"] = f["game_type"].astype(str)

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

    # top pair by abs(net_bb) for RING
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
    player_pairs["game_type"] = player_pairs["game_type"].astype(str)

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
    key = (str(game_type), int(target_id))
    sessions_count = int(idx.get("coplay_sessions_cnt", {}).get(key, 0))
    hu_sessions = int(idx.get("coplay_hu_cnt", {}).get(key, 0))
    sh_sessions = int(idx.get("coplay_sh_cnt", {}).get(key, 0))
    counter = idx.get("coplay_counter", {}).get(key, {})

    partners = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    unique_opponents = len(partners)
    top2_share = float((partners[0][1] + partners[1][1]) / sessions_count) if sessions_count and len(partners) >= 2 else 0.0

    return {
        "sessions_count": sessions_count,
        "unique_opponents": unique_opponents,
        "top2_coplay_share": top2_share,
        "hu_sessions": hu_sessions,
        "sh_sessions": sh_sessions,
        "top_partners": partners[:12],
    }


def _shared_sessions_list(pid_a: int, pid_b: int, idx: dict, game_type: str, limit: int = 20) -> list[str]:
    a = idx.get("sessions_by_player", {}).get((str(game_type), int(pid_a)), [])
    b = idx.get("sessions_by_player", {}).get((str(game_type), int(pid_b)), [])
    if not a or not b:
        return []
    if len(a) < len(b):
        sa = set(a)
        shared = [x for x in b if x in sa]
    else:
        sb = set(b)
        shared = [x for x in a if x in sb]
    return shared[:limit]


def pair_ctx_fast(target_id: int, partner_id: int, idx: dict, game_type: str) -> dict:
    gt = str(game_type)
    shared = _shared_sessions_list(target_id, partner_id, idx, gt, limit=999999)
    shared_cnt = int(len(shared))

    hu_share = 0.0
    if shared_cnt:
        sess_n = idx.get("sessions_n", {})
        hu = sum(1 for sid in shared if int(sess_n.get((gt, sid), 0)) == 2)
        hu_share = hu / shared_cnt

    # направленность по знаку результатов в общих game_id (на HU это работает хорошо)
    s1 = idx.get("player_game_series", {}).get((gt, int(target_id)))
    s2 = idx.get("player_game_series", {}).get((gt, int(partner_id)))
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
    gt = str(game_type)
    pid = int(target_id)

    in_s = idx.get("in_map", {}).get((gt, pid))
    out_s = idx.get("out_map", {}).get((gt, pid))
    top_inflows = [(int(k), float(v)) for k, v in in_s.head(12).items()] if in_s is not None and len(in_s) else []
    top_outflows = [(int(k), float(v)) for k, v in out_s.head(12).items()] if out_s is not None and len(out_s) else []

    totals = idx.get("flow_totals", {}).get((gt, pid), {"in": 0.0, "out": 0.0, "gross": 0.0, "one_sided": 0.0})
    top = idx.get("top_pair", {}).get((gt, pid))

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
            "one_sidedness": float(totals.get("one_sided", 0.0)),
            "pair_ctx": {},
            "shared_sessions_preview": [],
        }

    partner = int(top["partner"])
    ctx = pair_ctx_fast(pid, partner, idx, gt)
    shared_preview = _shared_sessions_list(pid, partner, idx, gt, limit=20)

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
# SCORING (переписано)
# =========================
def score_player(
    db_df: pd.DataFrame,
    db_sum: dict,
    cop_ring: dict,
    cop_tour: dict,
    trf_ring: dict,
    trf_tour: dict,
    coverage: dict,
) -> tuple[int, str, str, list[str], dict]:
    score = 0
    reasons = []
    signals = {}

    # ---- DB signals (мягко, чтобы не банить “по бухгалтерии”) ----
    jtot = float(db_sum.get("j_total", 0.0) or 0.0)
    events_delta = float(db_sum.get("events_delta", 0.0) or 0.0)
    weekscnt = int(db_sum.get("weeks_count", 0) or 0)
    top_week_share = db_sum.get("top_week_share", np.nan)

    signals["db_weeks"] = weekscnt
    signals["db_j_total"] = jtot
    signals["db_events_delta"] = events_delta
    signals["db_p_ring"] = float(db_sum.get("p_ring", 0.0) or 0.0)
    signals["db_p_mtt"] = float(db_sum.get("p_mtt", 0.0) or 0.0)
    signals["db_top_week_share"] = float(top_week_share) if pd.notna(top_week_share) else np.nan

    # Не наказываем за отсутствие игр: это “данных нет”, а не “перелив”
    ring_games = int(coverage.get("ring_games", 0) or 0)
    tour_games = int(coverage.get("tour_games", 0) or 0)
    signals["coverage_ring_games"] = ring_games
    signals["coverage_tour_games"] = tour_games

    if ring_games + tour_games == 0:
        score += 5
        reasons.append("GAMES: Нет игровых данных (в этой выгрузке) — невозможно оценить переводы.")
    else:
        reasons.append(f"GAMES: Ring={ring_games}, Tour={tour_games}.")

    # DB extreme: сильное влияние событий (иногда это “выдача/компенсации”, но в риск можно)
    if jtot != 0 and abs(events_delta) >= max(10.0, 0.25 * abs(jtot)):
        score += 6
        reasons.append("DB: Сильная разница J - O (влияние событий) — проверка источника средств/корректировок.")

    if weekscnt >= 2 and pd.notna(top_week_share) and float(top_week_share) >= 0.70:
        score += 4
        reasons.append("DB: Высокая концентрация результата в одной неделе.")

    ring_over_comm_ppsr = db_sum.get("ring_over_comm_ppsr", np.nan)
    pring = float(db_sum.get("p_ring", 0.0) or 0.0)
    if pd.notna(ring_over_comm_ppsr) and float(ring_over_comm_ppsr) >= 10 and abs(pring) >= 80:
        score += 6
        reasons.append("DB: Ring profit сильно непропорционален rake PPSR — бывает при переливах/подсадках.")

    # ---- Coplay (ключевой, но без фанатизма) ----
    signals["coplay_ring_sessions"] = int(cop_ring.get("sessions_count", 0) or 0)
    signals["coplay_ring_unique_opp"] = int(cop_ring.get("unique_opponents", 0) or 0)
    signals["coplay_ring_top2_share"] = float(cop_ring.get("top2_coplay_share", 0.0) or 0.0)
    signals["coplay_ring_hu_sessions"] = int(cop_ring.get("hu_sessions", 0) or 0)
    signals["coplay_ring_sh_sessions"] = int(cop_ring.get("sh_sessions", 0) or 0)

    if signals["coplay_ring_sessions"] >= 5 and signals["coplay_ring_unique_opp"] <= 2:
        score += 12
        reasons.append("RING: Мало уникальных оппонентов при заметном числе сессий (узкий круг).")

    if signals["coplay_ring_sessions"] >= 6 and signals["coplay_ring_top2_share"] >= COPLAY_TOP2_SHARE_SUSP:
        score += 10
        reasons.append("RING: Топ-2 оппонента занимают слишком большую долю игр (паттерн сговора/перелива).")

    if signals["coplay_ring_hu_sessions"] >= 3 and safe_div(signals["coplay_ring_hu_sessions"], max(1, signals["coplay_ring_sessions"])) >= 0.60:
        score += 8
        reasons.append("RING: Много HU (или почти HU) — типичный канал перелива.")

    # ---- Transfers: используем только консервативные flows (HU + ослабленный 3-max) ----
    def flow_block(trf: dict, label: str):
        nonlocal score, reasons, signals

        partner = trf.get("top_net_partner")
        if partner is None:
            return

        net = float(trf.get("top_net", 0.0) or 0.0)
        net_bb = trf.get("top_net_bb", np.nan)
        gross = float(trf.get("top_gross", 0.0) or 0.0)
        gross_bb = trf.get("top_gross_bb", np.nan)

        partner_share_bb = trf.get("top_partner_share_bb", np.nan)
        partner_share = float(partner_share_bb) if pd.notna(partner_share_bb) else float(trf.get("top_partner_share", 0.0) or 0.0)

        ctx = trf.get("pair_ctx", {}) or {}
        shared = int(ctx.get("shared_sessions", 0) or 0)
        dir_cons = float(ctx.get("dir_consistency", 0.0) or 0.0)
        hu_share = float(ctx.get("hu_share", 0.0) or 0.0)
        one_sided = float(trf.get("one_sidedness", 0.0) or 0.0)

        if label == "RING":
            signals["ring_top_partner"] = int(partner)
            signals["ring_net"] = net
            signals["ring_net_bb"] = float(net_bb) if pd.notna(net_bb) else np.nan
            signals["ring_gross"] = gross
            signals["ring_gross_bb"] = float(gross_bb) if pd.notna(gross_bb) else np.nan
            signals["ring_partner_share"] = float(partner_share)
            signals["ring_shared_sessions"] = shared
            signals["ring_dir_cons"] = float(dir_cons)
            signals["ring_hu_share"] = float(hu_share)
            signals["ring_one_sided"] = float(one_sided)
            signals["ring_shared_sessions_preview"] = trf.get("shared_sessions_preview", []) or []

        # Консервативный “гейт”: если почти нет HU/short, и shared=1, то не эскалируем по net
        # (иначе мультивей-инференс даст лишние MANUAL)
        reliable_enough = (hu_share >= 0.50 and shared >= 2) or (shared >= 4)

        # Net checks
        if label == "RING" and pd.notna(net_bb):
            abs_net = abs(float(net_bb))
            if abs_net >= PAIR_NET_CRITICAL_RING_BB and shared >= 1 and (hu_share >= 0.40 or shared >= 3):
                score += 55
                reasons.append(f"{label}: Критичный net {abs_net:.1f} BB с партнёром {partner}.")
            elif abs_net >= PAIR_NET_ALERT_RING_BB and reliable_enough:
                score += 25
                reasons.append(f"{label}: Высокий net {abs_net:.1f} BB с партнёром {partner}.")
        else:
            abs_net = abs(float(net))
            if abs_net >= PAIR_NET_ALERT_TOUR and reliable_enough:
                score += 10
                reasons.append(f"{label}: Высокий net {abs_net:.2f} с партнёром {partner}.")

        # Gross checks
        if label == "RING" and pd.notna(gross_bb):
            if float(gross_bb) >= PAIR_GROSS_ALERT_RING_BB and shared >= 2 and hu_share >= 0.40:
                score += 10
                reasons.append(f"{label}: Высокий gross {float(gross_bb):.1f} BB с партнёром {partner}.")
        else:
            if gross >= 60 and shared >= 3:
                score += 4
                reasons.append(f"{label}: Высокий gross {gross:.2f} с партнёром {partner}.")

        # Direction consistency
        if dir_cons >= PAIR_DIR_CONSIST_ALERT and shared >= 3 and hu_share >= 0.40:
            score += 12 if label == "RING" else 5
            reasons.append(f"{label}: Направление выигрышей повторяется (dir={dir_cons*100:.0f}%).")

        # Partner share
        if partner_share >= PAIR_PARTNER_SHARE_ALERT and shared >= 3:
            score += 10 if label == "RING" else 5
            reasons.append(f"{label}: Слишком большая доля переводного оборота на одного партнёра (share={partner_share*100:.0f}%).")

        # One-sidedness
        if one_sided >= PAIR_ONE_SIDED_ALERT and shared >= 3 and hu_share >= 0.40:
            score += 8 if label == "RING" else 4
            reasons.append(f"{label}: Переводный оборот сильно односторонний (one-sided={one_sided*100:.0f}%).")

        # Agent link bonus
        bonus, txt = agent_match_bonus(db_df, int(db_sum["meta"]["player_id"]), int(partner))
        if bonus > 0:
            score += bonus
            reasons.append(f"LINK: {txt}")

    flow_block(trf_ring, "RING")
    flow_block(trf_tour, "TOURNAMENT")

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)

    if decision == "APPROVE":
        manager_text = f"ID {db_sum['meta']['player_id']}: Можно проводить. Риск {score}/100. Сильных сигналов перелива по надёжным данным не найдено."
    elif decision == "FAST_CHECK":
        manager_text = f"ID {db_sum['meta']['player_id']}: Пауза и быстрая проверка. Риск {score}/100. Есть сигналы — проверь пару/паттерн игр."
    else:
        manager_text = f"ID {db_sum['meta']['player_id']}: В СБ. Риск {score}/100. Высокая вероятность перелива/сговора (по HU/short и net BB)."

    return score, decision, manager_text, reasons, signals


# =========================
# UI helpers
# =========================
def decision_badge(decision: str) -> tuple[str, str]:
    if decision == "APPROVE":
        return "✅ APPROVE", "green"
    if decision == "FAST_CHECK":
        return "🟠 FAST_CHECK", "orange"
    return "⛔ MANUAL_REVIEW", "red"


def render_signal_row(label: str, value: str, status: str):
    if status == "bad":
        st.error(f"{label}: {value}")
    elif status == "warn":
        st.warning(f"{label}: {value}")
    else:
        st.success(f"{label}: {value}")


def build_security_message(pid: int, decision: str, score: int, weeks_mode: str, week_from: int, week_to: int, signals: dict) -> str:
    ring_partner = signals.get("ring_top_partner")
    ring_net_bb = signals.get("ring_net_bb", np.nan)
    ring_net = signals.get("ring_net", 0.0)
    ring_gross_bb = signals.get("ring_gross_bb", np.nan)
    ring_gross = signals.get("ring_gross", 0.0)
    ring_shared = signals.get("ring_shared_sessions", 0)
    ring_dir = signals.get("ring_dir_cons", 0.0)
    ring_one = signals.get("ring_one_sided", 0.0)
    ring_hu_share = signals.get("ring_hu_share", 0.0)
    shared_ids = signals.get("ring_shared_sessions_preview", []) or []

    net_str = f"{ring_net_bb:.1f} BB" if pd.notna(ring_net_bb) else f"{ring_net:.2f} (BB не извлечён)"
    gross_str = f"{ring_gross_bb:.1f} BB" if pd.notna(ring_gross_bb) else f"{ring_gross:.2f}"

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
    msg.append(f"- J (итог + события): {signals.get('db_j_total', 0.0):.2f}")
    msg.append(f"- Ring: {signals.get('db_p_ring', 0.0):.2f}; MTT/SNG: {signals.get('db_p_mtt', 0.0):.2f}")
    msg.append(f"- J - O (влияние событий): {signals.get('db_events_delta', 0.0):.2f}")
    msg.append("")
    msg.append("Игры (покрытие):")
    msg.append(f"- Ring игр: {signals.get('coverage_ring_games', 0)}, Tour игр: {signals.get('coverage_tour_games', 0)}")
    msg.append(f"- Ring co-play: сессий {signals.get('coplay_ring_sessions', 0)}, HU {signals.get('coplay_ring_hu_sessions', 0)}, топ-2 доля {signals.get('coplay_ring_top2_share', 0.0)*100:.0f}%")
    msg.append("")
    msg.append("Топ-пара Ring (консервативные переводы: HU + ослабленный 3-max):")
    msg.append(f"- Партнёр: {ring_partner}")
    msg.append(f"- Net: {net_str}; Gross: {gross_str}")
    msg.append(f"- Shared sessions: {ring_shared}; HU-share: {ring_hu_share*100:.0f}%; dir: {ring_dir*100:.0f}%; one-sided: {ring_one*100:.0f}%")
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
# TOP RISK TABLE
# =========================
def top_risk_players(db_period: pd.DataFrame, games_df: pd.DataFrame, idx: dict, topn: int = 50) -> pd.DataFrame:
    pids = sorted(db_period["_player_id"].unique().tolist())
    rows = []
    for pid in pids:
        db_sum, _ = db_summary_for_player(db_period, int(pid))
        if db_sum is None:
            continue

        rings = idx.get("player_game_series", {}).get(("RING", int(pid)))
        tours = idx.get("player_game_series", {}).get(("TOURNAMENT", int(pid)))
        coverage = {"ring_games": int(len(rings)) if rings is not None else 0,
                    "tour_games": int(len(tours)) if tours is not None else 0}

        cop_ring = coplay_features_fast(int(pid), idx, "RING")
        cop_tour = coplay_features_fast(int(pid), idx, "TOURNAMENT")
        trf_ring = transfer_features_fast(int(pid), idx, "RING")
        trf_tour = transfer_features_fast(int(pid), idx, "TOURNAMENT")

        score, decision, manager_text, reasons, signals = score_player(
            db_period, db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage
        )

        rows.append({
            "player_id": int(pid),
            "risk_score": int(score),
            "decision": decision,
            "nick": db_sum["meta"].get("nick", ""),
            "ign": db_sum["meta"].get("ign", ""),
            "agent": db_sum["meta"].get("agent", ""),
            "weeks": int(db_sum.get("weeks_count", 0) or 0),
            "j_total": float(db_sum.get("j_total", 0.0) or 0.0),
            "p_ring": float(db_sum.get("p_ring", 0.0) or 0.0),
            "p_mtt": float(db_sum.get("p_mtt", 0.0) or 0.0),
            "ring_games": int(coverage["ring_games"]),
            "ring_top_partner": signals.get("ring_top_partner", None),
            "ring_net_bb": signals.get("ring_net_bb", np.nan),
            "ring_shared_sessions": signals.get("ring_shared_sessions", 0),
            "ring_hu_share": signals.get("ring_hu_share", 0.0),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["risk_score", "j_total"], ascending=[False, False]).head(int(topn)).copy()
    return out


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Файлы")
    up_db = st.file_uploader("DB (Общий) .csv/.xlsx", type=["csv", "xlsx", "xls"], key="up_db")
    up_games = st.file_uploader("Games export .txt/.csv", type=["txt", "csv"], key="up_games")

    colA, colB = st.columns(2)
    if colA.button("Очистить DB cache", use_container_width=True):
        cache_clear(DB_KEY)
    if colB.button("Очистить Games cache", use_container_width=True):
        cache_clear(GAMES_KEY)

    st.divider()
    st.subheader("Фильтр по неделям")
    weeks_mode = st.selectbox("Режим", ["Все недели", "Последние N недель", "Диапазон недель"], index=1)
    last_n = st.number_input("N", min_value=1, max_value=52, value=2, step=1, disabled=(weeks_mode != "Последние N недель"))
    week_from = st.number_input("Неделя от", min_value=0, max_value=999, value=1, step=1, disabled=(weeks_mode != "Диапазон недель"))
    week_to = st.number_input("Неделя до", min_value=0, max_value=999, value=999, step=1, disabled=(weeks_mode != "Диапазон недель"))

    st.divider()
    topn = st.number_input("Топ риск-игроков", min_value=10, max_value=500, value=50, step=10)

db_file = resolve_file(DB_KEY, up_db)
games_file = resolve_file(GAMES_KEY, up_games)

if db_file is None:
    st.info("Загрузи DB-файл (Общий) для старта.")
    st.stop()

# Load DB
try:
    db_df = load_db_any(db_file)
except Exception as e:
    st.error(f"Ошибка чтения DB: {e}")
    st.stop()

db_period = apply_weeks_filter(db_df, weeks_mode, int(last_n), int(week_from), int(week_to))

# Load games
games_df = pd.DataFrame(columns=[
    "game_id","game_type","product","table_name","descriptor","bb","start_time","end_time",
    "player_id","nick","ign","hands","win_total","win_vs_opponents","fee"
])
if games_file is not None:
    try:
        games_df = parse_games_pppoker_export(games_file)
    except Exception as e:
        st.warning(f"Не удалось распарсить Games export: {e}")

sessions_df = build_sessions_from_games(games_df)
flows_df = build_pair_flows_conservative(games_df, sessions_df)
idx = build_games_indexes(games_df, sessions_df, flows_df)

# Header metrics
m1, m2, m3, m4 = st.columns(4, gap="small")
m1.metric("DB строк", f"{len(db_period)}", border=True)
m2.metric("DB игроков", f"{db_period['_player_id'].nunique()}", border=True)
m3.metric("Games строк", f"{len(games_df)}", border=True)
m4.metric("Flows пар", f"{len(flows_df)}", border=True)

st.divider()

tab_check, tab_top = st.tabs(["Проверка ID", "Топ рисков"])

with tab_top:
    st.subheader("Топ риск-игроков (консервативный скоринг)")
    top_df = top_risk_players(db_period, games_df, idx, topn=int(topn))
    if top_df.empty:
        st.info("Нет данных для построения топа (проверь фильтры/выгрузку игр).")
    else:
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        csvbytes = top_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Скачать CSV", data=csvbytes, file_name="top_risk_players.csv", mime="text/csv")

with tab_check:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Проверка игрока")
        default_id = int(db_period["_player_id"].iloc[0]) if len(db_period) else int(db_df["_player_id"].iloc[0])
        pid = st.number_input("ID игрока", min_value=0, value=default_id, step=1)
        run = st.button("Рассчитать риск", type="primary", use_container_width=True)

        st.divider()
        st.caption("Подсказка: чтобы снизить false-positive, flows считаются строго (HU) и осторожно (3-max).")

    if not run:
        st.stop()

    db_sum, by_week = db_summary_for_player(db_period, int(pid))
    if db_sum is None:
        st.error("Игрок не найден в выбранном периоде.")
        st.stop()

    rings = idx.get("player_game_series", {}).get(("RING", int(pid)))
    tours = idx.get("player_game_series", {}).get(("TOURNAMENT", int(pid)))
    coverage = {"ring_games": int(len(rings)) if rings is not None else 0,
                "tour_games": int(len(tours)) if tours is not None else 0}

    cop_ring = coplay_features_fast(int(pid), idx, "RING")
    cop_tour = coplay_features_fast(int(pid), idx, "TOURNAMENT")
    trf_ring = transfer_features_fast(int(pid), idx, "RING")
    trf_tour = transfer_features_fast(int(pid), idx, "TOURNAMENT")

    score, decision, manager_text, reasons, signals = score_player(
        db_period, db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage
    )

    badge_text, badge_color = decision_badge(decision)

    with right:
        st.subheader("Результат")
        cA, cB, cC = st.columns([1.4, 1, 1], gap="small")
        cA.metric("Решение", badge_text, border=True)
        cB.metric("Risk score", f"{score}/100", border=True)
        cC.metric("Недель в периоде", f"{signals.get('db_weeks', 0)}", border=True)

        st.progress(score / 100)

        if badge_color == "green":
            st.success(manager_text)
        elif badge_color == "orange":
            st.warning(manager_text)
        else:
            st.error(manager_text)

        st.subheader("Причины (лог)")
        for r in reasons:
            st.write(f"- {r}")

        st.divider()
        st.subheader("Ключевые сигналы")

        cov_total = int(signals.get("coverage_ring_games", 0)) + int(signals.get("coverage_tour_games", 0))
        if cov_total == 0:
            render_signal_row("Покрытие игр", "0", "warn")
        else:
            render_signal_row("Покрытие игр", f"Ring {signals.get('coverage_ring_games', 0)}, Tour {signals.get('coverage_tour_games', 0)}", "ok")

        render_signal_row("Ring co-play", f"сессий {signals.get('coplay_ring_sessions', 0)}, HU {signals.get('coplay_ring_hu_sessions', 0)}, топ-2 {signals.get('coplay_ring_top2_share', 0.0)*100:.0f}%", "ok")

        partner = signals.get("ring_top_partner", None)
        if partner is None:
            render_signal_row("Топ-пара Ring", "Нет", "ok")
        else:
            net_bb = signals.get("ring_net_bb", np.nan)
            net_str = fmt_bb(net_bb) if pd.notna(net_bb) else fmt_money(signals.get("ring_net", 0.0))
            shared = int(signals.get("ring_shared_sessions", 0))
            hu_share = float(signals.get("ring_hu_share", 0.0))
            dirc = float(signals.get("ring_dir_cons", 0.0))
            ones = float(signals.get("ring_one_sided", 0.0))

            status = "ok"
            if pd.notna(net_bb) and abs(float(net_bb)) >= PAIR_NET_CRITICAL_RING_BB:
                status = "bad"
            elif pd.notna(net_bb) and abs(float(net_bb)) >= PAIR_NET_ALERT_RING_BB:
                status = "warn"

            render_signal_row("Топ-пара Ring", f"партнёр {partner}, net {net_str}, shared {shared}, HU-share {hu_share*100:.0f}%, dir {dirc*100:.0f}%, one-sided {ones*100:.0f}%", status)

        st.divider()
        st.subheader("Сообщение для СБ")
        sec_msg = build_security_message(int(pid), decision, int(score), weeks_mode, int(week_from), int(week_to), signals)
        st.text_area("Текст", sec_msg, height=260)
        copy_to_clipboard_button(sec_msg, label="Скопировать для СБ")

        st.download_button(
            "Скачать .txt",
            data=sec_msg.encode("utf-8"),
            file_name=f"security_request_{pid}.txt",
            mime="text/plain",
        )
