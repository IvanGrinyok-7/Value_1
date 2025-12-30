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
APP_TITLE = "PPPoker Anti-Fraud: общий + игры (FAST + BB normalized)"
CACHE_DIR = Path(".pppoker_app_cache")
CACHE_DIR.mkdir(exist_ok=True)

DB_KEY = "db_file"
GAMES_KEY = "games_file"

T_APPROVE = 25
T_FAST_CHECK = 55

MIN_SESSIONS_FOR_COPLAY = 6
COPLAY_TOP2_SHARE_SUSP = 0.80

# Пороги в BB (ключевое изменение)
PAIR_NET_ALERT_RING_BB = 30.0        # net >= 30 BB
PAIR_GROSS_ALERT_RING_BB = 90.0      # gross >= 90 BB
PAIR_NET_ALERT_TOUR = 60.0           # турниры оставляем в валюте (шумнее)
PAIR_ONE_SIDED_ALERT = 0.85
PAIR_DIR_CONSIST_ALERT = 0.78
PAIR_PARTNER_SHARE_ALERT = 0.55
PAIR_MIN_SHARED_SESSIONS_STRONG = 3  # снизили: для BB-нормализации 3 уже полезно

SINGLE_GAME_WIN_ALERT_RING_BB = 80.0
SINGLE_GAME_WIN_ALERT_TOUR = 150.0

# Regex PPPoker export
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)
TABLE_NAME_RE = re.compile(r"Название стола:\s*(.+?)\s*$", re.IGNORECASE)
START_END_RE = re.compile(r"Начало:\s*([0-9/:\s]+)\s+By.+?Окончание:\s*([0-9/:\s]+)", re.IGNORECASE)

RING_HINT_RE = re.compile(r"\bPPSR\b|PLO|OFC|NLH|Bomb Pot|Ante|3-1|HU\b", re.IGNORECASE)
TOUR_HINT_RE = re.compile(r"\bPPST\b|Бай-ин:|satellite|pko|mko\b", re.IGNORECASE)

# извлечение стейков типа 0.2/0.4, 0.11/0.22 и т.п.
STAKES_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)")

# DB columns (лист "Общий")
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
# CACHE FILE STORE (как раньше)
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
# GAMES PARSER (+extract BB)
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
    """
    Для PPSR строк обычно есть стейк '0.2/0.4'. Берём BB (второе число).
    """
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

        if ("PPSR" in line or "PPST" in line) and ("ID игрока" not in line):
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
                "buyin_pp": find("Бай-ин с PP-фишками"),
                "buyin_ticket": find("Бай-ин с билетом"),
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
            if hidx is not None and hidx + 4 < len(parts):
                row["win_total"] = to_float(parts[hidx + 1])
                row["win_vs_opponents"] = to_float(parts[hidx + 2])
            else:
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
        return pd.DataFrame(columns=["game_id","game_type","product","table_name","descriptor","bb","start_time","end_time","player_id","nick","ign","hands","win_total","win_vs_opponents","fee"])

    for c in ["bb","hands","win_total","win_vs_opponents","fee"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["player_id","game_id","game_type"]).copy()
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
# FLOWS (amount + amount_bb)
# =========================
def build_pair_flows_fast(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["from_player","to_player","game_type","amount","amount_bb","games_cnt"])

    df = games_df[["game_id","game_type","player_id","bb","win_total","win_vs_opponents"]].copy()
    # flow_win: для RING используем "от соперников" если есть
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

    for (gid, gtype), part in df.groupby(["game_id","game_type"], sort=False):
        part = part[["player_id","_flow_win","bb"]]
        if part["player_id"].nunique() < 2:
            continue

        winners = part[part["_flow_win"] > 0]
        losers = part[part["_flow_win"] < 0]
        if winners.empty or losers.empty:
            continue

        total_pos = float(winners["_flow_win"].sum())
        if total_pos <= 0:
            continue

        # bb для игры: берём max(bb) среди строк (обычно одинаковый)
        bb = float(part["bb"].max()) if gtype == "RING" else np.nan
        bb_ok = (bb > 0) if gtype == "RING" else False

        win_pids = winners["player_id"].astype(int).to_numpy()
        win_vals = winners["_flow_win"].to_numpy(dtype=float)
        win_w = win_vals / total_pos

        for lpid, lwin in losers[["player_id","_flow_win"]].itertuples(index=False):
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
        return pd.DataFrame(columns=["from_player","to_player","game_type","amount","amount_bb","games_cnt"])

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
    out = out.groupby(["from_player","to_player","game_type"], as_index=False).agg(
        amount=("amount","sum"),
        amount_bb=("amount_bb","sum"),
        games_cnt=("games_cnt","sum"),
    )
    return out


# =========================
# INDEXES (fast lookups) + TOP PAIR BY abs(net_bb)
# =========================
def build_games_indexes(games_df: pd.DataFrame, sessions_df: pd.DataFrame, flows_df: pd.DataFrame):
    idx = {}

    # player_game_series: flow_win per game_id (для dir consistency)
    if games_df.empty:
        idx["player_game_series"] = {}
        idx["extremes"] = {}
    else:
        d = games_df[["game_type","player_id","game_id","bb","win_total","win_vs_opponents"]].copy()
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
        for (gt, pid), g in d.groupby(["game_type","player_id"], sort=False):
            s = pd.Series(g["_flow_win"].to_numpy(dtype=float), index=g["game_id"].to_numpy())
            series[(gt, int(pid))] = s
            if gt == "RING":
                # extremes в BB: нужен bb на уровне игры -> берём медиану bb по строкам игрока в этой игре
                # для упрощения: extreme по валюте оставим, а в скоринге сравним BB через пары
                extremes[(gt, int(pid))] = []
            else:
                big = s[s >= SINGLE_GAME_WIN_ALERT_TOUR]
                extremes[(gt, int(pid))] = list(big.index[:12]) if not big.empty else []
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
        for sid, gt, players, pn in sessions_df[["session_id","game_type","players","players_n"]].itertuples(index=False):
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

    idx["sessions_by_player"] = dict(sessions_by_player)
    idx["sessions_n"] = sessions_n
    idx["coplay_counter"] = {k: dict(v) for k, v in coplay_counter.items()}
    idx["coplay_sessions_cnt"] = dict(coplay_sessions_cnt)
    idx["coplay_hu_cnt"] = dict(coplay_hu_cnt)
    idx["coplay_sh_cnt"] = dict(coplay_sh_cnt)

    # flows -> in/out maps + top pair
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
    for (gt, to_pid), g in f.groupby(["game_type","to_player"], sort=False):
        s = g.groupby("from_player")["amount"].sum().sort_values(ascending=False)
        in_map[(gt, int(to_pid))] = s

    out_map = {}
    for (gt, from_pid), g in f.groupby(["game_type","from_player"], sort=False):
        s = g.groupby("to_player")["amount"].sum().sort_values(ascending=False)
        out_map[(gt, int(from_pid))] = s

    idx["in_map"] = in_map
    idx["out_map"] = out_map

    # totals for one_sidedness (currency)
    inflow_total = f.groupby(["game_type","to_player"])["amount"].sum()
    outflow_total = f.groupby(["game_type","from_player"])["amount"].sum()
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

    # top pair by abs(net_bb) for RING (fallback to abs(net) если BB нет)
    tmp = f[["game_type","from_player","to_player","amount","amount_bb","games_cnt"]].copy()
    tmp["p"] = tmp[["from_player","to_player"]].min(axis=1)
    tmp["q"] = tmp[["from_player","to_player"]].max(axis=1)

    tmp["signed_to_q_amt"] = np.where(tmp["to_player"] == tmp["q"], tmp["amount"], -tmp["amount"])
    tmp["signed_to_q_bb"] = np.where(tmp["to_player"] == tmp["q"], tmp["amount_bb"], -tmp["amount_bb"])

    pair = tmp.groupby(["game_type","p","q"], as_index=False).agg(
        net_to_q=("signed_to_q_amt","sum"),
        net_to_q_bb=("signed_to_q_bb","sum"),
        gross=("amount","sum"),
        gross_bb=("amount_bb","sum"),
        games_cnt=("games_cnt","sum"),
    )

    vq = pair.rename(columns={"q":"player_id","p":"partner_id","net_to_q":"net","net_to_q_bb":"net_bb"})
    vp = pair.rename(columns={"p":"player_id","q":"partner_id","net_to_q":"net","net_to_q_bb":"net_bb"})
    vp["net"] = -vp["net"]
    vp["net_bb"] = -vp["net_bb"]

    player_pairs = pd.concat([vq, vp], ignore_index=True)
    player_pairs["player_id"] = player_pairs["player_id"].astype(int)
    player_pairs["partner_id"] = player_pairs["partner_id"].astype(int)

    # gross_total per player (в валюте и BB)
    gross_total = player_pairs.groupby(["game_type","player_id"])["gross"].sum()
    gross_total_bb = player_pairs.groupby(["game_type","player_id"])["gross_bb"].sum()

    # рейтинг пары: RING -> abs(net_bb) если есть, иначе abs(net)
    def pair_rank_row(r):
        if r["game_type"] == "RING" and pd.notna(r["net_bb"]):
            return abs(float(r["net_bb"]))
        return abs(float(r["net"]))

    player_pairs["_rank"] = player_pairs.apply(pair_rank_row, axis=1)
    top_rows = player_pairs.sort_values("_rank", ascending=False).groupby(["game_type","player_id"], as_index=False).head(1)

    top_pair = {}
    for gt, pid, partner, net, net_bb, gross, gross_bb, gcnt, _rank in top_rows[["game_type","player_id","partner_id","net","net_bb","gross","gross_bb","games_cnt","_rank"]].itertuples(index=False):
        gt = str(gt); pid = int(pid); partner = int(partner)
        gtot = float(gross_total.get((gt,pid), 0.0))
        gtot_bb = float(gross_total_bb.get((gt,pid), 0.0))
        top_pair[(gt,pid)] = {
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
# FEATURES (fast)
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

    return {
        "sessions_count": sessions_count,
        "unique_opponents": unique_opponents,
        "top2_coplay_share": top2_share,
        "hu_sessions": hu_sessions,
        "sh_sessions": sh_sessions,
        "top_partners": partners[:12],
    }


def pair_ctx_fast(target_id: int, partner_id: int, idx: dict, game_type: str) -> dict:
    a = idx.get("sessions_by_player", {}).get((game_type, int(target_id)), [])
    b = idx.get("sessions_by_player", {}).get((game_type, int(partner_id)), [])

    if not a or not b:
        shared = []
    else:
        sa = set(a) if len(a) <= len(b) else set(b)
        shared = [x for x in (b if len(a) <= len(b) else a) if x in sa]

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
            "one_sidedness": float(totals.get("one_sided", 0.0)),
            "pair_ctx": {},
        }

    partner = int(top["partner"])
    ctx = pair_ctx_fast(pid, partner, idx, game_type)

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
    # Диапазон
    return d[(d["_week"] >= int(week_from)) & (d["_week"] <= int(week_to))].copy()


def db_summary_for_player(db_period: pd.DataFrame, player_id: int):
    d = db_period[db_period["_player_id"] == int(player_id)].copy()
    if d.empty:
        return None, None

    # Суммируем нужные поля
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

    # Топ неделя по концентрации профита
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
    # Проверка совпадения агента для усиления риска
    a = db_df[db_df["_player_id"] == pid_a].tail(1)
    b = db_df[db_df["_player_id"] == pid_b].tail(1)
    if a.empty or b.empty:
        return 0, None
    
    a_ag = a.iloc[0].get("_agent_id")
    b_ag = b.iloc[0].get("_agent_id")
    if pd.notna(a_ag) and pd.notna(b_ag) and float(a_ag) == float(b_ag):
        return 6, "Пара под одним агентом (усилитель риска)."
        
    a_sag = a.iloc[0].get("_super_agent_id")
    b_sag = b.iloc[0].get("_super_agent_id")
    if pd.notna(a_sag) and pd.notna(b_sag) and float(a_sag) == float(b_sag):
        return 4, "Пара под одним суперагентом (усилитель риска)."
        
    return 0, None


# =========================
# SCORING (BB-normalized)
# =========================
def score_player(db_df: pd.DataFrame, db_sum: dict, cop_ring: dict, cop_tour: dict, trf_ring: dict, trf_tour: dict, coverage: dict):
    score = 0
    reasons = []

    # 1. DB Checks
    j_tot = db_sum["j_total"]
    if j_tot <= 0:
        score += 5
        reasons.append("DB: игрок в минусе по J — как 'получатель' перелива менее вероятен.")
    else:
        score += 18
        reasons.append("DB: игрок в плюсе по J — нужен контроль на перелив.")
    
    # Delta J vs O (события)
    if j_tot > 0 and abs(db_sum["events_delta"]) >= max(10.0, 0.25 * j_tot):
        score += 8
        reasons.append("DB: высокая дельта J-O (много непокерных начислений).")

    # Концентрация
    if db_sum["weeks_count"] >= 2 and pd.notna(db_sum["top_week_share"]) and db_sum["top_week_share"] >= 0.60:
        score += 8
        reasons.append("DB: профит концентрирован в одной неделе.")

    # Coverage
    if coverage["ring_games"] + coverage["tour_games"] == 0:
        score += 18
        reasons.append("GAMES: нет игр в файле — проверка невозможна.")
    else:
        reasons.append(f"GAMES: RING={coverage['ring_games']}, TOURNAMENT={coverage['tour_games']}.")

    # 2. Co-play Ring
    if cop_ring["sessions_count"] >= MIN_SESSIONS_FOR_COPLAY:
        if cop_ring["unique_opponents"] <= 5:
            score += 8
            reasons.append("GAMES/RING: мало уникальных оппонентов.")
        if cop_ring["top2_coplay_share"] >= COPLAY_TOP2_SHARE_SUSP:
            score += 8
            reasons.append("GAMES/RING: топ-2 оппонента в большинстве сессий.")
        if cop_ring["hu_sessions"] >= 6 and (cop_ring["hu_sessions"] / cop_ring["sessions_count"]) >= 0.6:
            score += 6
            reasons.append("GAMES/RING: высокая доля HU сессий.")

    # 3. Flow Logic (BB aware)
    def check_flow(trf: dict, label: str):
        nonlocal score
        p = trf.get("top_net_partner")
        if p is None:
            return

        net_val = trf.get("top_net", 0.0)
        net_bb = trf.get("top_net_bb", np.nan)
        gross_val = trf.get("top_gross", 0.0)
        gross_bb = trf.get("top_gross_bb", np.nan)
        
        # Для доли используем BB если есть, иначе валюту
        if pd.notna(trf.get("top_partner_share_bb")):
            p_share = trf.get("top_partner_share_bb")
        else:
            p_share = trf.get("top_partner_share")

        ctx = trf.get("pair_ctx", {})
        shared = ctx.get("shared_sessions", 0)
        dir_cons = ctx.get("dir_consistency", 0.0)
        hu_share = ctx.get("hu_share", 0.0)
        one_sided = trf.get("one_sidedness", 0.0)

        # Info string
        val_str = f"{fmt_money(net_val)}"
        if pd.notna(net_bb):
            val_str += f" ({net_bb:.1f} BB)"
        
        reasons.append(
            f"GAMES/{label}: топ-пара {p}, net≈{val_str}, shared={shared}, "
            f"HU≈{hu_share:.0%}, dir≈{dir_cons:.0%}."
        )

        # Триггеры
        # А) Net flow
        is_high_net = False
        if label == "RING":
            # Если есть BB, смотрим BB порог
            if pd.notna(net_bb):
                if abs(net_bb) >= PAIR_NET_ALERT_RING_BB:
                    is_high_net = True
            # Fallback на валюту (если BB нет или для надёжности) - тут можно оставить только BB если уверены
            # но лучше комбинировать: если BB нет, то старый порог 25 (по умолчанию)
            elif abs(net_val) >= 25.0: 
                 is_high_net = True
        else:
            # Tournaments - валюта
            if abs(net_val) >= PAIR_NET_ALERT_TOUR:
                is_high_net = True

        if is_high_net and shared >= 2:
            score += 25 if label == "RING" else 12
            reasons.append(f"GAMES/{label}: крупный перелив (net-flow) с одним игроком.")

        # Б) Gross turnover
        is_high_gross = False
        if label == "RING":
             if pd.notna(gross_bb) and gross_bb >= PAIR_GROSS_ALERT_RING_BB:
                 is_high_gross = True
        elif gross_val >= 60.0:
             is_high_gross = True
             
        if is_high_gross and shared >= PAIR_MIN_SHARED_SESSIONS_STRONG:
            score += 10 if label == "RING" else 5
            reasons.append(f"GAMES/{label}: аномальный оборот с парой.")

        # В) Доля партнёра
        if p_share >= PAIR_PARTNER_SHARE_ALERT and shared >= PAIR_MIN_SHARED_SESSIONS_STRONG:
            score += 12 if label == "RING" else 6
            reasons.append(f"GAMES/{label}: игра практически только с этим партнёром.")

        # Г) Односторонность + Net
        if one_sided >= PAIR_ONE_SIDED_ALERT and shared >= 2 and is_high_net:
            score += 10 if label == "RING" else 4
            reasons.append(f"GAMES/{label}: односторонний поток (схема слива).")

        # Д) Консистентность
        if dir_cons >= PAIR_DIR_CONSIST_ALERT and shared >= PAIR_MIN_SHARED_SESSIONS_STRONG:
            score += 12 if label == "RING" else 5
            reasons.append(f"GAMES/{label}: стабильный проигрыш/выигрыш в одну сторону.")

        # Е) Agent match
        bonus, txt = agent_match_bonus(db_df, int(p), int(db_sum["meta"]["player_id"]))
        if bonus > 0 and txt:
            score += bonus
            reasons.append(f"DB: {txt}")

    check_flow(trf_ring, "RING")
    check_flow(trf_tour, "TOURNAMENT")

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)
    
    if decision == "APPROVE":
        risk_msg = "Явных признаков сговора нет."
    elif decision == "FAST_CHECK":
        risk_msg = "Есть подозрительные метрики."
    else:
        risk_msg = "Высокий риск: перелив или сговор."

    return score, decision, risk_msg, reasons


# =========================
# STREAMLIT CACHED WRAPPERS
# =========================
@st.cache_data(show_spinner=False)
def cached_load_db(content: bytes, name: str):
    return load_db_any(BytesFile(content, name))

@st.cache_data(show_spinner=True)
def cached_games_bundle(content: bytes, name: str):
    # Парсинг
    games_df = parse_games_pppoker_export(BytesFile(content, name))
    # Структуры
    sessions_df = build_sessions_from_games(games_df)
    flows_df = build_pair_flows_fast(games_df)
    # Индексы
    idx = build_games_indexes(games_df, sessions_df, flows_df)
    return games_df, sessions_df, flows_df, idx

@st.cache_data(show_spinner=False)
def cached_top_suspicious(db_period: pd.DataFrame, idx: dict, top_n: int):
    players = sorted(db_period["_player_id"].unique().tolist())
    res = []
    
    for pid in players:
        db_sum, _ = db_summary_for_player(db_period, int(pid))
        if db_sum is None:
            continue
            
        # Feature gathering
        r_series = idx.get("player_game_series", {}).get(("RING", int(pid)))
        t_series = idx.get("player_game_series", {}).get(("TOURNAMENT", int(pid)))
        cov = {
            "ring_games": len(r_series) if r_series is not None else 0,
            "tour_games": len(t_series) if t_series is not None else 0
        }
        
        c_ring = coplay_features_fast(int(pid), idx, "RING")
        c_tour = coplay_features_fast(int(pid), idx, "TOURNAMENT")
        
        t_ring = transfer_features_fast(int(pid), idx, "RING")
        t_tour = transfer_features_fast(int(pid), idx, "TOURNAMENT")
        
        sc, dec, _, _ = score_player(db_period, db_sum, c_ring, c_tour, t_ring, t_tour, cov)
        
        # Для таблицы добавим читаемые поля
        p_net = t_ring.get("top_net", 0.0)
        p_net_bb = t_ring.get("top_net_bb", np.nan)
        p_share = t_ring.get("top_partner_share", 0.0)
        if pd.notna(t_ring.get("top_partner_share_bb")):
             p_share = t_ring.get("top_partner_share_bb")
             
        res.append({
            "player_id": int(pid),
            "risk_score": sc,
            "decision": dec,
            "db_j_total": db_sum["j_total"],
            "games_ring": cov["ring_games"],
            "top_partner": t_ring.get("top_net_partner"),
            "net_flow": p_net,
            "net_flow_bb": p_net_bb,
            "partner_share": p_share
        })
        
    df = pd.DataFrame(res)
    if df.empty:
        return df
    return df.sort_values(["risk_score", "db_j_total"], ascending=[False, False]).head(int(top_n))


# =========================
# UI MAIN
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Загрузка")
    db_up = st.file_uploader("DB (Excel/CSV)", type=["xlsx","xls","csv"], key="db_u")
    games_up = st.file_uploader("Games (Export)", type=["csv","txt"], key="games_u")
    
    c1, c2 = st.columns(2)
    if c1.button("Clear DB"):
        cache_clear(DB_KEY)
        st.rerun()
    if c2.button("Clear Games"):
        cache_clear(GAMES_KEY)
        st.rerun()
        
    db_file = resolve_file(DB_KEY, db_up)
    games_file = resolve_file(GAMES_KEY, games_up)
    
    st.divider()
    weeks_mode = st.selectbox("Период", ["Все недели", "Последние N недель", "Диапазон недель"])
    last_n = st.number_input("N", min_value=1, value=4)
    
if db_file is None:
    st.info("Загрузите файл DB (общий).")
    st.stop()

# Load DB
db_df = cached_load_db(db_file.getvalue(), getattr(db_file, "name", "db"))
valid_weeks = sorted([w for w in db_df["_week"].unique().tolist() if w >= 0])
w_min = min(valid_weeks) if valid_weeks else 0
w_max = max(valid_weeks) if valid_weeks else 0

week_from = w_min
week_to = w_max
if weeks_mode == "Диапазон недель":
    with st.sidebar:
        week_from = st.number_input("Week From", value=w_min)
        week_to = st.number_input("Week To", value=w_max)

db_period = apply_weeks_filter(db_df, weeks_mode, int(last_n), int(week_from), int(week_to))

# Load Games
games_df = pd.DataFrame()
sessions_df = pd.DataFrame()
flows_df = pd.DataFrame()
idx = {
    "player_game_series": {}, "extremes": {}, "sessions_by_player": {}, "sessions_n": {}, 
    "coplay_counter": {}, "coplay_sessions_cnt": {}, "coplay_hu_cnt": {}, "coplay_sh_cnt": {}, 
    "in_map": {}, "out_map": {}, "flow_totals": {}, "top_pair": {}
}

if games_file is not None:
    g_bytes = games_file.getvalue()
    games_df, sessions_df, flows_df, idx = cached_games_bundle(g_bytes, getattr(games_file, "name", "games"))

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("DB Rows", len(db_df), border=True)
m2.metric("Players", db_df["_player_id"].nunique(), border=True)
m3.metric("Games Rows", len(games_df), border=True)
m4.metric("Pairs Flow", len(flows_df), border=True)

st.divider()

t1, t2 = st.tabs(["Проверка ID", "ТОП Риска"])

with t1:
    c_in, c_info = st.columns([1, 2])
    with c_in:
        def_id = int(db_df["_player_id"].iloc[0]) if not db_df.empty else 0
        pid_in = st.number_input("Player ID", value=def_id, step=1)
        if st.button("Check", type="primary"):
            st.session_state["check_pid"] = pid_in
            
    if "check_pid" in st.session_state:
        pid = st.session_state["check_pid"]
        db_sum, by_week = db_summary_for_player(db_period, pid)
        
        if db_sum is None:
            st.error("Игрок не найден в DB.")
        else:
            # Gather Features
            r_s = idx.get("player_game_series", {}).get(("RING", pid))
            t_s = idx.get("player_game_series", {}).get(("TOURNAMENT", pid))
            cov = {
                "ring_games": len(r_s) if r_s is not None else 0,
                "tour_games": len(t_s) if t_s is not None else 0
            }
            
            cp_r = coplay_features_fast(pid, idx, "RING")
            cp_t = coplay_features_fast(pid, idx, "TOURNAMENT")
            
            tr_r = transfer_features_fast(pid, idx, "RING")
            tr_t = transfer_features_fast(pid, idx, "TOURNAMENT")
            
            score, decision, main_risk, reasons = score_player(db_period, db_sum, cp_r, cp_t, tr_r, tr_t, cov)
            
            # Display
            st.subheader(f"Результат: {decision} ({score}/100)")
            if decision == "APPROVE":
                st.success(main_risk)
            elif decision == "FAST_CHECK":
                st.warning(main_risk)
            else:
                st.error(main_risk)
                
            with st.expander("Детали и причины", expanded=True):
                for r in reasons:
                    st.write(f"- {r}")
                    
            # Details Tabs
            dt1, dt2, dt3 = st.tabs(["DB Info", "Games / Net Flow", "Partners"])
            
            with dt1:
                c1, c2, c3 = st.columns(3)
                c1.metric("J Total", fmt_money(db_sum["j_total"]))
                c2.metric("P Ring", fmt_money(db_sum["p_ring"]))
                c3.metric("Events Delta", fmt_money(db_sum["events_delta"]))
                st.dataframe(by_week, use_container_width=True)
                
            with dt2:
                st.markdown("**Ring Net Flow (Top Partner)**")
                if tr_r["top_net_partner"]:
                    st.write(f"Partner: {tr_r['top_net_partner']}")
                    st.write(f"Net: {tr_r['top_net']:.2f} (BB: {tr_r.get('top_net_bb', np.nan):.1f})")
                    st.write(f"Gross: {tr_r['top_gross']:.2f} (BB: {tr_r.get('top_gross_bb', np.nan):.1f})")
                else:
                    st.info("Нет значимых связей.")
                    
            with dt3:
                if cp_r["top_partners"]:
                    st.write("Top Co-play Partners (Ring):")
                    st.dataframe(pd.DataFrame(cp_r["top_partners"], columns=["Partner", "Sessions"]))

with t2:
    if st.button("Сформировать ТОП"):
        top_df = cached_top_suspicious(db_period, idx, 50)
        if top_df.empty:
            st.info("Пусто.")
        else:
            st.dataframe(top_df, use_container_width=True)
