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
APP_TITLE = "PPPoker Anti-Fraud: общий + игры (analytics v2 FAST)"
CACHE_DIR = Path(".pppoker_app_cache")
CACHE_DIR.mkdir(exist_ok=True)

DB_KEY = "db_file"
GAMES_KEY = "games_file"

T_APPROVE = 25
T_FAST_CHECK = 55

MIN_SESSIONS_FOR_COPLAY = 6
COPLAY_TOP1_SHARE_SUSP = 0.60
COPLAY_TOP2_SHARE_SUSP = 0.80

PAIR_NET_ALERT_RING = 25.0
PAIR_NET_ALERT_TOUR = 60.0
PAIR_GROSS_ALERT_RING = 60.0
PAIR_ONE_SIDED_ALERT = 0.85
PAIR_DIR_CONSIST_ALERT = 0.78
PAIR_PARTNER_SHARE_ALERT = 0.55
PAIR_MIN_SHARED_SESSIONS_STRONG = 4

SINGLE_GAME_WIN_ALERT_RING = 60.0
SINGLE_GAME_WIN_ALERT_TOUR = 150.0

# Regex для твоего экспорта
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)
TABLE_NAME_RE = re.compile(r"Название стола:\s*(.+?)\s*$", re.IGNORECASE)
START_END_RE = re.compile(r"Начало:\s*([0-9/:\s]+)\s+By.+?Окончание:\s*([0-9/:\s]+)", re.IGNORECASE)

RING_HINT_RE = re.compile(r"\bPPSR\b|PLO|OFC|NLH|Bomb Pot|Ante|3-1|HU\b", re.IGNORECASE)
TOUR_HINT_RE = re.compile(r"\bPPST\b|Бай-ин:|satellite|pko|mko\b", re.IGNORECASE)

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

COL_TICKET_VALUE = "Стоимость выигранного билета"
COL_TICKET_BUYIN = "Бай-ин с билетом"
COL_CUSTOM_PRIZE = "Стоимость настраиваемого приза"

EXTRA_PLAYER_WIN_COL_PREFIX = "Выигрыш игрока "


# =========================
# SMALL UTIL
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


def file_hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


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
        COL_TICKET_VALUE: "_ticket_value",
        COL_TICKET_BUYIN: "_ticket_buyin",
        COL_CUSTOM_PRIZE: "_custom_prize",
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
# GAMES PARSER
# =========================
def _classify_game_type(descriptor: str) -> str:
    if descriptor is None:
        return "UNKNOWN"
    if TOUR_HINT_RE.search(descriptor):
        return "TOURNAMENT"
    if RING_HINT_RE.search(descriptor):
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
        "start_time": None,
        "end_time": None,
    }

    header = None
    mode = None
    idx = {}

    def reset_table_state():
        nonlocal header, mode, idx
        header = None
        mode = None
        idx = {}

    for line in lines:
        m = GAME_ID_RE.search(line)
        if m:
            current["game_id"] = m.group(1).strip()
            current["table_name"] = ""
            current["descriptor"] = ""
            current["game_type"] = "UNKNOWN"
            current["product"] = ""
            current["start_time"] = None
            current["end_time"] = None
            reset_table_state()
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
            if "PPSR" in line:
                current["product"] = "PPSR"
            elif "PPST" in line:
                current["product"] = "PPST"
            continue

        if "ID игрока" in line:
            header = _split_semicolon(line)
            mode = current["game_type"]
            if mode == "UNKNOWN":
                mode = "RING" if ("Раздачи" in header or "Выигрыш игрока" in line) else "TOURNAMENT"

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
                "bounty": find("От баунти"),
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
            "start_time": current["start_time"],
            "end_time": current["end_time"],
            "player_id": pid,
            "nick": (parts[idx["nick"]] if idx.get("nick") is not None and idx["nick"] < len(parts) else ""),
            "ign": (parts[idx["ign"]] if idx.get("ign") is not None and idx["ign"] < len(parts) else ""),
            "hands": np.nan,
            "buyin_pp": np.nan,
            "buyin_ticket": np.nan,
            "win_total": np.nan,
            "win_vs_opponents": np.nan,
            "win_jackpot": np.nan,
            "win_equity": np.nan,
            "fee": np.nan,
            "bounty": np.nan,
        }

        if idx.get("hands") is not None and idx["hands"] < len(parts):
            row["hands"] = to_float(parts[idx["hands"]])
        if idx.get("buyin_pp") is not None and idx["buyin_pp"] < len(parts):
            row["buyin_pp"] = to_float(parts[idx["buyin_pp"]])
        if idx.get("buyin_ticket") is not None and idx["buyin_ticket"] < len(parts):
            row["buyin_ticket"] = to_float(parts[idx["buyin_ticket"]])
        if idx.get("fee") is not None and idx["fee"] < len(parts):
            row["fee"] = to_float(parts[idx["fee"]])
        if idx.get("bounty") is not None and idx["bounty"] < len(parts):
            row["bounty"] = to_float(parts[idx["bounty"]])

        if current["game_type"] == "RING":
            hidx = idx.get("hands")
            if hidx is not None and hidx + 4 < len(parts):
                row["win_total"] = to_float(parts[hidx + 1])
                row["win_vs_opponents"] = to_float(parts[hidx + 2])
                row["win_jackpot"] = to_float(parts[hidx + 3])
                row["win_equity"] = to_float(parts[hidx + 4])
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
        return pd.DataFrame(
            columns=[
                "game_id", "game_type", "product", "table_name", "descriptor",
                "start_time", "end_time",
                "player_id", "nick", "ign",
                "hands", "buyin_pp", "buyin_ticket",
                "win_total", "win_vs_opponents", "win_jackpot", "win_equity",
                "fee", "bounty",
            ]
        )

    for c in ["hands", "buyin_pp", "buyin_ticket", "win_total", "win_vs_opponents", "win_jackpot", "win_equity", "fee", "bounty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================
# FAST DERIVED STRUCTURES
# =========================
def _flow_win_value(game_type: str, win_vs_opponents, win_total) -> float:
    if game_type == "RING" and pd.notna(win_vs_opponents):
        return float(win_vs_opponents)
    return float(win_total) if pd.notna(win_total) else np.nan


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


def build_pair_flows_fast(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Тот же смысл, что и build_pair_flows раньше, но:
    - не пересчитывает flow_win через apply по строкам (векторно)
    - избегает лишних копий
    """
    if games_df.empty:
        return pd.DataFrame(columns=["from_player", "to_player", "amount", "game_type", "games_cnt"])

    df = games_df[["game_id", "game_type", "player_id", "win_total", "win_vs_opponents"]].copy()
    df["_flow_win"] = np.where(
        (df["game_type"] == "RING") & df["win_vs_opponents"].notna(),
        df["win_vs_opponents"],
        df["win_total"],
    )
    df = df[df["_flow_win"].notna()].copy()
    df["_flow_win"] = pd.to_numeric(df["_flow_win"], errors="coerce")
    df = df[df["_flow_win"].notna()].copy()

    flows = {}
    games_cnt = {}

    for (gid, gtype), part in df.groupby(["game_id", "game_type"], sort=False):
        part = part[["player_id", "_flow_win"]]
        if part["player_id"].nunique() < 2:
            continue

        winners = part[part["_flow_win"] > 0]
        losers = part[part["_flow_win"] < 0]
        if winners.empty or losers.empty:
            continue

        total_pos = float(winners["_flow_win"].sum())
        if total_pos <= 0:
            continue

        win_pids = winners["player_id"].astype(int).to_numpy()
        win_vals = winners["_flow_win"].to_numpy(dtype=float)

        # нормируем доли победителей 1 раз
        win_weights = win_vals / total_pos

        for lpid, lwin in losers[["player_id", "_flow_win"]].itertuples(index=False):
            lpid = int(lpid)
            loss = float(-lwin)
            if loss <= 0:
                continue
            amts = loss * win_weights
            for wpid, amt in zip(win_pids, amts):
                wpid = int(wpid)
                key = (lpid, wpid, gtype)
                flows[key] = flows.get(key, 0.0) + float(amt)
                games_cnt[key] = games_cnt.get(key, 0) + 1

    if not flows:
        return pd.DataFrame(columns=["from_player", "to_player", "game_type", "amount", "games_cnt"])

    out = pd.DataFrame(
        [{"from_player": k[0], "to_player": k[1], "game_type": k[2], "amount": v, "games_cnt": games_cnt[k]}
         for k, v in flows.items()]
    )
    out = out.groupby(["from_player", "to_player", "game_type"], as_index=False).agg({"amount": "sum", "games_cnt": "sum"})
    return out


def build_games_indexes(games_df: pd.DataFrame, sessions_df: pd.DataFrame, flows_df: pd.DataFrame):
    """
    Главная оптимизация: один раз строим индексы, дальше проверка ID — это быстрые lookup.
    """
    idx = {}

    # --- flow_win series per (game_type, player_id) for dir_consistency/extremes
    if games_df.empty:
        idx["player_game_series"] = {}
        idx["extremes"] = {}
    else:
        df = games_df[["game_type", "player_id", "game_id", "win_total", "win_vs_opponents"]].copy()
        df["_flow_win"] = np.where(
            (df["game_type"] == "RING") & df["win_vs_opponents"].notna(),
            df["win_vs_opponents"],
            df["win_total"],
        )
        df = df[df["_flow_win"].notna()].copy()
        df["_flow_win"] = pd.to_numeric(df["_flow_win"], errors="coerce")
        df = df[df["_flow_win"].notna()].copy()
        # компактнее
        df["player_id"] = df["player_id"].astype(int)
        df["game_id"] = df["game_id"].astype(str)

        player_game_series = {}
        for (gt, pid), g in df.groupby(["game_type", "player_id"], sort=False):
            # Series: index=game_id, value=flow_win
            s = pd.Series(g["_flow_win"].to_numpy(dtype=float), index=g["game_id"].to_numpy())
            player_game_series[(gt, int(pid))] = s

        idx["player_game_series"] = player_game_series

        extremes = {}
        for (gt, pid), s in player_game_series.items():
            thr = SINGLE_GAME_WIN_ALERT_TOUR if gt == "TOURNAMENT" else SINGLE_GAME_WIN_ALERT_RING
            big = s[s >= thr]
            if not big.empty:
                extremes[(gt, pid)] = list(big.index[:12])
            else:
                extremes[(gt, pid)] = []
        idx["extremes"] = extremes

    # --- sessions inverted index: (game_type, player_id) -> list(session_id)
    sessions_by_player = defaultdict(list)
    sessions_n = {}
    if not sessions_df.empty:
        for sid, gt, players, pn in sessions_df[["session_id", "game_type", "players", "players_n"]].itertuples(index=False):
            sid = str(sid)
            sessions_n[(gt, sid)] = int(pn)
            for pid in players:
                sessions_by_player[(gt, int(pid))].append(sid)
    idx["sessions_by_player"] = dict(sessions_by_player)
    idx["sessions_n"] = sessions_n

    # --- coplay partner counters + basic counts (one pass, small k^2 per table)
    coplay_counter = defaultdict(lambda: defaultdict(int))
    coplay_sessions_cnt = defaultdict(int)
    coplay_hu_cnt = defaultdict(int)
    coplay_sh_cnt = defaultdict(int)

    if not sessions_df.empty:
        for sid, gt, players, pn in sessions_df[["session_id", "game_type", "players", "players_n"]].itertuples(index=False):
            pn = int(pn)
            pls = [int(x) for x in players]
            for p in pls:
                key = (gt, p)
                coplay_sessions_cnt[key] += 1
                if pn == 2:
                    coplay_hu_cnt[key] += 1
                if pn <= 3:
                    coplay_sh_cnt[key] += 1

            # пары внутри стола
            for i in range(len(pls)):
                pi = pls[i]
                di = coplay_counter[(gt, pi)]
                for j in range(len(pls)):
                    if i == j:
                        continue
                    pj = pls[j]
                    di[pj] += 1

    idx["coplay_counter"] = {k: dict(v) for k, v in coplay_counter.items()}
    idx["coplay_sessions_cnt"] = dict(coplay_sessions_cnt)
    idx["coplay_hu_cnt"] = dict(coplay_hu_cnt)
    idx["coplay_sh_cnt"] = dict(coplay_sh_cnt)

    # --- flows: быстрые top inflow/outflow списки и топ-пара по abs(net)
    if flows_df.empty:
        idx["in_map"] = {}
        idx["out_map"] = {}
        idx["flow_totals"] = {}
        idx["top_pair"] = {}
        return idx

    f = flows_df.copy()
    f["from_player"] = f["from_player"].astype(int)
    f["to_player"] = f["to_player"].astype(int)

    # in/out maps (Series) для топ-12 списков
    in_map = {}
    for (gt, to_pid), g in f.groupby(["game_type", "to_player"], sort=False):
        s = g.groupby("from_player")["amount"].sum()
        in_map[(gt, int(to_pid))] = s.sort_values(ascending=False)

    out_map = {}
    for (gt, from_pid), g in f.groupby(["game_type", "from_player"], sort=False):
        s = g.groupby("to_player")["amount"].sum()
        out_map[(gt, int(from_pid))] = s.sort_values(ascending=False)

    idx["in_map"] = in_map
    idx["out_map"] = out_map

    inflow_total = f.groupby(["game_type", "to_player"])["amount"].sum()
    outflow_total = f.groupby(["game_type", "from_player"])["amount"].sum()

    # undirected pair net/gross for top pair
    tmp = f[["game_type", "from_player", "to_player", "amount", "games_cnt"]].copy()
    tmp["p"] = tmp[["from_player", "to_player"]].min(axis=1)
    tmp["q"] = tmp[["from_player", "to_player"]].max(axis=1)
    tmp["signed_to_q"] = np.where(tmp["to_player"] == tmp["q"], tmp["amount"], -tmp["amount"])

    pair = (
        tmp.groupby(["game_type", "p", "q"], as_index=False)
        .agg(net_to_q=("signed_to_q", "sum"), gross=("amount", "sum"), games_cnt=("games_cnt", "sum"))
    )

    # view as per-player rows
    vq = pair.rename(columns={"q": "player_id", "p": "partner_id", "net_to_q": "net"})
    vp = pair.rename(columns={"p": "player_id", "q": "partner_id", "net_to_q": "net"})
    vp["net"] = -vp["net"]

    player_pairs = pd.concat([vq, vp], ignore_index=True)
    player_pairs["player_id"] = player_pairs["player_id"].astype(int)
    player_pairs["partner_id"] = player_pairs["partner_id"].astype(int)

    # gross total per player (по всем парам)
    gross_total = player_pairs.groupby(["game_type", "player_id"])["gross"].sum()

    # top pair by abs(net)
    player_pairs["_absnet"] = player_pairs["net"].abs()
    top_rows = (
        player_pairs.sort_values("_absnet", ascending=False)
        .groupby(["game_type", "player_id"], as_index=False)
        .head(1)
    )

    top_pair = {}
    for gt, pid, partner, net, gross, gcnt, _ in top_rows[["game_type", "player_id", "partner_id", "net", "gross", "games_cnt", "_absnet"]].itertuples(index=False):
        gt = str(gt)
        pid = int(pid)
        partner = int(partner)
        gross_tot = float(gross_total.get((gt, pid), 0.0))
        top_pair[(gt, pid)] = {
            "partner": partner,
            "net": float(net),
            "gross": float(gross),
            "gross_total": gross_tot,
            "partner_share": float(gross / gross_tot) if gross_tot > 0 else 0.0,
            "games_cnt": int(gcnt),
        }

    flow_totals = {}
    # totals + one_sidedness
    # one_sidedness = abs(in-out)/(in+out)
    for (gt, pid), in_amt in inflow_total.items():
        pid = int(pid)
        out_amt = float(outflow_total.get((gt, pid), 0.0))
        in_amt = float(in_amt)
        gross = in_amt + out_amt
        one_sided = abs(in_amt - out_amt) / gross if gross > 0 else 0.0
        flow_totals[(gt, pid)] = {"in": in_amt, "out": out_amt, "gross": gross, "one_sided": float(one_sided)}
    # players who only outflow (no inflow)
    for (gt, pid), out_amt in outflow_total.items():
        pid = int(pid)
        if (gt, pid) in flow_totals:
            continue
        out_amt = float(out_amt)
        in_amt = 0.0
        gross = out_amt
        one_sided = 1.0 if gross > 0 else 0.0
        flow_totals[(gt, pid)] = {"in": in_amt, "out": out_amt, "gross": gross, "one_sided": float(one_sided)}

    idx["flow_totals"] = flow_totals
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
    top1 = partners[0][1] if len(partners) >= 1 else 0
    top2 = partners[1][1] if len(partners) >= 2 else 0

    return {
        "sessions_count": sessions_count,
        "unique_opponents": unique_opponents,
        "top1_coplay_share": float(top1 / sessions_count) if sessions_count else 0.0,
        "top2_coplay_share": float((top1 + top2) / sessions_count) if sessions_count else 0.0,
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
        # пересечение списков: быстрее через set меньшего
        if len(a) < len(b):
            sb = set(a)
            shared = [x for x in b if x in sb]
        else:
            sa = set(a)
            shared = [x for x in b if x in sa]

    shared_cnt = int(len(shared))
    if shared_cnt == 0:
        hu_share = 0.0
    else:
        sess_n = idx.get("sessions_n", {})
        hu = 0
        for sid in shared:
            if sess_n.get((game_type, sid), 0) == 2:
                hu += 1
        hu_share = hu / shared_cnt

    # direction consistency по общим game_id
    s1 = idx.get("player_game_series", {}).get((game_type, int(target_id)))
    s2 = idx.get("player_game_series", {}).get((game_type, int(partner_id)))
    if s1 is None or s2 is None:
        dir_cons = 0.0
    else:
        # align по индексу = game_id
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

    s = idx.get("player_game_series", {}).get((game_type, pid))
    target_games = int(len(s)) if s is not None else 0
    target_total_flow_win = float(s.sum()) if s is not None and len(s) else 0.0

    # inflows / outflows топ-12
    in_s = idx.get("in_map", {}).get((game_type, pid))
    out_s = idx.get("out_map", {}).get((game_type, pid))

    top_inflows = []
    if in_s is not None and len(in_s):
        top_inflows = [(int(k), float(v)) for k, v in in_s.head(12).items()]

    top_outflows = []
    if out_s is not None and len(out_s):
        top_outflows = [(int(k), float(v)) for k, v in out_s.head(12).items()]

    totals = idx.get("flow_totals", {}).get((game_type, pid), {"in": 0.0, "out": 0.0, "gross": 0.0, "one_sided": 0.0})
    top = idx.get("top_pair", {}).get((game_type, pid))

    if top is None:
        top_partner = None
        top_net = 0.0
        top_gross = 0.0
        top_partner_share = 0.0
    else:
        top_partner = int(top["partner"])
        top_net = float(top["net"])
        top_gross = float(top["gross"])
        top_partner_share = float(top.get("partner_share", 0.0))

    extremes_list = idx.get("extremes", {}).get((game_type, pid), [])
    extremes = [{"game_id": gid} for gid in extremes_list[:12]] if extremes_list else []

    # pair context только для топ-партнёра
    if top_partner is not None:
        ctx = pair_ctx_fast(pid, top_partner, idx, game_type)
    else:
        ctx = {}

    return {
        "target_games": target_games,
        "target_total_flow_win": target_total_flow_win,
        "top_inflows": top_inflows,
        "top_outflows": top_outflows,
        "top_net_partner": top_partner,
        "top_net": float(top_net),
        "top_gross_with_partner": float(top_gross),
        "top_partner_share": float(top_partner_share),
        "one_sidedness": float(totals.get("one_sided", 0.0)),
        "pair_ctx": ctx,
        "extremes": extremes,
    }


# =========================
# DB PERIOD + SUMMARY (fast)
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
        top_week = None
        top_week_j = 0.0
        top_week_share = np.nan
    else:
        top_row = by_week.sort_values("_j_total", ascending=False).iloc[0]
        top_week = int(top_row["_week"])
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
        "week_min": int(by_week["_week"].min()) if len(by_week) else None,
        "week_max": int(by_week["_week"].max()) if len(by_week) else None,

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

        "top_week": top_week,
        "top_week_j": top_week_j,
        "top_week_share": top_week_share,

        "meta": meta,
    }
    return summary, by_week


def agent_match_bonus(db_df: pd.DataFrame, pid_a: int, pid_b: int) -> tuple[int, str | None]:
    a = db_df[db_df["_player_id"] == pid_a].tail(1)
    b = db_df[db_df["_player_id"] == pid_b].tail(1)
    if a.empty or b.empty:
        return 0, None
    a_agent = a.iloc[0].get("_agent_id")
    b_agent = b.iloc[0].get("_agent_id")
    a_sagent = a.iloc[0].get("_super_agent_id")
    b_sagent = b.iloc[0].get("_super_agent_id")

    if pd.notna(a_agent) and pd.notna(b_agent) and float(a_agent) == float(b_agent):
        return 6, "Пара под одним агентом (усилитель риска)."
    if pd.notna(a_sagent) and pd.notna(b_sagent) and float(a_sagent) == float(b_sagent):
        return 4, "Пара под одним суперагентом (усилитель риска)."
    return 0, None


# =========================
# SCORING (та же логика, быстрые фичи)
# =========================
def score_player(db_df: pd.DataFrame, db_sum: dict, cop_ring: dict, cop_tour: dict, trf_ring: dict, trf_tour: dict, coverage: dict):
    score = 0
    reasons = []

    total_j = db_sum["j_total"]
    poker_profit = db_sum["poker_profit"]
    ring_over_comm_ppsr = db_sum["ring_over_comm_ppsr"]
    top_week_share = db_sum["top_week_share"]
    events_delta = db_sum["events_delta"]

    # DB baseline
    if total_j <= 0:
        score += 5
        reasons.append("DB: игрок в минусе по J — как 'получатель' перелива менее вероятен.")
    else:
        score += 18
        reasons.append("DB: игрок в плюсе по J — нужен контроль на перелив/сговор.")

    if total_j > 0 and abs(events_delta) >= max(10.0, 0.25 * abs(total_j)):
        score += 8
        reasons.append("DB: высокая дельта J-O (существенная роль событий/непокерных компонентов).")

    if poker_profit > 0:
        if db_sum["weeks_count"] >= 2 and pd.notna(top_week_share) and top_week_share >= 0.60:
            score += 8
            reasons.append("DB: профит концентрирован в одной неделе.")

        if pd.notna(ring_over_comm_ppsr) and ring_over_comm_ppsr >= 8 and abs(db_sum["p_ring"]) >= 80:
            score += 8
            reasons.append("DB: очень высокий Ring профит относительно PPSR комиссии (аномалия/перелив/выборка).")

    # Coverage
    if coverage["ring_games"] + coverage["tour_games"] == 0:
        score += 18
        reasons.append("GAMES: игрок не найден в файле игр — проверка перелива по играм невозможна.")
    else:
        reasons.append(f"GAMES: покрытие — RING={coverage['ring_games']}, TOURNAMENT={coverage['tour_games']}.")

    # Co-play
    if cop_ring["sessions_count"] >= MIN_SESSIONS_FOR_COPLAY:
        if cop_ring["unique_opponents"] <= 5:
            score += 8
            reasons.append("GAMES/RING: мало уникальных оппонентов (узкий круг).")
        if cop_ring["top2_coplay_share"] >= COPLAY_TOP2_SHARE_SUSP:
            score += 8
            reasons.append("GAMES/RING: топ‑2 оппонента покрывают большую часть сессий.")
        if cop_ring["hu_sessions"] >= 6 and (cop_ring["hu_sessions"] / max(1, cop_ring["sessions_count"])) >= 0.6:
            score += 6
            reasons.append("GAMES/RING: много HU — усиливает риск при наличии net-flow по паре.")

    def apply_flow_block(trf: dict, label: str, net_alert: float, gross_alert: float):
        nonlocal score, reasons
        partner = trf.get("top_net_partner")
        if partner is None:
            return

        top_net = float(trf.get("top_net", 0.0))
        top_gross = float(trf.get("top_gross_with_partner", 0.0))
        partner_share = float(trf.get("top_partner_share", 0.0))
        one_sided = float(trf.get("one_sidedness", 0.0))
        ctx = trf.get("pair_ctx", {}) or {}
        shared = int(ctx.get("shared_sessions", 0))
        hu_share = float(ctx.get("hu_share", 0.0))
        dir_cons = float(ctx.get("dir_consistency", 0.0))

        direction = "в пользу игрока" if top_net > 0 else "в пользу партнёра"
        reasons.append(
            f"GAMES/{label}: топ‑пара={partner}, net≈{fmt_money(top_net)} ({direction}), gross≈{fmt_money(top_gross)}, "
            f"shared={shared}, HU≈{hu_share:.0%}, dir≈{dir_cons:.0%}."
        )

        if abs(top_net) >= net_alert and shared >= 2:
            score += 22 if label == "RING" else 10
            reasons.append(f"GAMES/{label}: крупный net-flow с одним ID (подозрение на перелив/сговор).")

        if top_gross >= gross_alert and shared >= PAIR_MIN_SHARED_SESSIONS_STRONG:
            score += 10 if label == "RING" else 5
            reasons.append(f"GAMES/{label}: большой оборот пары (много денег проходит через одну пару).")

        if partner_share >= PAIR_PARTNER_SHARE_ALERT and shared >= PAIR_MIN_SHARED_SESSIONS_STRONG:
            score += 12 if label == "RING" else 6
            reasons.append(f"GAMES/{label}: доля оборота с одной парой слишком высокая.")

        if one_sided >= PAIR_ONE_SIDED_ALERT and shared >= 2 and abs(top_net) >= (net_alert / 2):
            score += 10 if label == "RING" else 4
            reasons.append(f"GAMES/{label}: поток односторонний (типичный паттерн перелива).")

        if dir_cons >= PAIR_DIR_CONSIST_ALERT and shared >= PAIR_MIN_SHARED_SESSIONS_STRONG:
            score += 12 if label == "RING" else 5
            reasons.append(f"GAMES/{label}: высокая консистентность направления (один стабильно выигрывает у другого).")

        if hu_share >= 0.6 and shared >= PAIR_MIN_SHARED_SESSIONS_STRONG and abs(top_net) >= (net_alert / 2):
            score += 8 if label == "RING" else 3
            reasons.append(f"GAMES/{label}: много HU в паре + net-flow (сильный сигнал).")

        bonus, txt = agent_match_bonus(db_df, int(partner), int(db_sum["meta"]["player_id"]))
        if bonus > 0 and txt:
            score += bonus
            reasons.append(f"DB: {txt}")

        if trf.get("extremes"):
            score += 6 if label == "RING" else 3
            reasons.append(f"GAMES/{label}: есть экстремальные выигрыши в отдельных играх (триггер на HH/ручную проверку).")

    apply_flow_block(trf_ring, "RING", PAIR_NET_ALERT_RING, PAIR_GROSS_ALERT_RING)
    apply_flow_block(trf_tour, "TOURNAMENT", PAIR_NET_ALERT_TOUR, PAIR_GROSS_ALERT_RING * 2)

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)

    if decision == "APPROVE":
        main_risk = "Явных признаков перелива/сговора по текущей выборке не выявлено."
    elif decision == "FAST_CHECK":
        main_risk = "Есть признаки риска — нужна быстрая проверка (пары/сессии/выплата)."
    else:
        main_risk = "Высокий риск перелива/сговора — обязательна ручная проверка СБ (желательно HH/транзакции)."

    return score, decision, main_risk, reasons


# =========================
# STREAMLIT CACHES (ускорение)
# =========================
@st.cache_data(show_spinner=False)
def cached_load_db(content: bytes, name: str):
    return load_db_any(BytesFile(content, name))


@st.cache_data(show_spinner=True)
def cached_games_bundle(content: bytes, name: str):
    games_df = parse_games_pppoker_export(BytesFile(content, name))
    sessions_df = build_sessions_from_games(games_df)
    flows_df = build_pair_flows_fast(games_df)
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

        # coverage по сериям
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

        score, decision, _, _ = score_player(db_period, db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)

        res.append({
            "player_id": int(pid),
            "risk_score": int(score),
            "decision": decision,
            "db_j_total": float(db_sum["j_total"]),
            "db_p_ring": float(db_sum["p_ring"]),
            "db_p_mtt": float(db_sum["p_mtt"]),
            "db_events_delta": float(db_sum["events_delta"]),
            "games_ring": coverage["ring_games"],
            "games_tour": coverage["tour_games"],
            "coplay_ring_sessions": int(cop_ring["sessions_count"]),
            "flow_ring_top_partner": trf_ring.get("top_net_partner") if trf_ring.get("top_net_partner") is not None else "",
            "flow_ring_net": float(trf_ring.get("top_net", 0.0)),
            "flow_ring_partner_share": float(trf_ring.get("top_partner_share", 0.0)),
            "flow_ring_dir": float(trf_ring.get("pair_ctx", {}).get("dir_consistency", 0.0)),
        })

    out = pd.DataFrame(res)
    if out.empty:
        return out
    out = out.sort_values(["risk_score", "db_j_total"], ascending=[False, False]).head(int(top_n)).copy()
    return out


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Загрузка (2 файла)")

    st.subheader("1) Общий (CSV или XLSX)")
    st.caption("Если XLSX — читается лист 'Общий'.")
    db_up = st.file_uploader("DB", type=["csv", "xlsx", "xls"], key="db_up")

    st.subheader("2) Games (экспорт PPPoker одним файлом)")
    games_up = st.file_uploader("Games", type=["csv", "txt"], key="games_up")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Очистить DB", use_container_width=True):
            cache_clear(DB_KEY)
            st.rerun()
    with c2:
        if st.button("Очистить Games", use_container_width=True):
            cache_clear(GAMES_KEY)
            st.rerun()
    with c3:
        if st.button("Очистить всё", use_container_width=True):
            cache_clear(DB_KEY)
            cache_clear(GAMES_KEY)
            st.rerun()

    db_file = resolve_file(DB_KEY, db_up)
    games_file = resolve_file(GAMES_KEY, games_up)

    st.divider()
    st.subheader("Период (по неделям из DB)")
    weeks_mode = st.selectbox("Фильтр недель", ["Все недели", "Последние N недель", "Диапазон недель"])
    last_n = st.number_input("N (если 'Последние N недель')", min_value=1, value=4, step=1)

if db_file is None:
    st.info("Загрузи общий файл (DB).")
    st.stop()

# --- DB (cached)
db_bytes = db_file.getvalue()
db_df = cached_load_db(db_bytes, getattr(db_file, "name", "db"))
valid_weeks = sorted([w for w in db_df["_week"].unique().tolist() if w >= 0])
w_min = min(valid_weeks) if valid_weeks else 0
w_max = max(valid_weeks) if valid_weeks else 0

# weeks inputs
week_from = w_min
week_to = w_max
if weeks_mode == "Диапазон недель":
    with st.sidebar:
        week_from = st.number_input("Неделя от", value=w_min, step=1)
        week_to = st.number_input("Неделя до", value=w_max, step=1)

db_period = apply_weeks_filter(db_df, weeks_mode, int(last_n), int(week_from), int(week_to))

# --- Games bundle (cached)
games_df = pd.DataFrame()
sessions_df = pd.DataFrame()
flows_df = pd.DataFrame()
games_idx = {}

if games_file is not None:
    g_bytes = games_file.getvalue()
    games_df, sessions_df, flows_df, games_idx = cached_games_bundle(g_bytes, getattr(games_file, "name", "games"))
else:
    games_idx = {"player_game_series": {}, "extremes": {}, "sessions_by_player": {}, "sessions_n": {}, "coplay_counter": {},
                 "coplay_sessions_cnt": {}, "coplay_hu_cnt": {}, "coplay_sh_cnt": {}, "in_map": {}, "out_map": {}, "flow_totals": {}, "top_pair": {}}

m1, m2, m3, m4 = st.columns(4, gap="small")
m1.metric("DB строк", f"{len(db_df)}", border=True)
m2.metric("DB игроков", f"{db_df['_player_id'].nunique()}", border=True)
m3.metric("Games строк", f"{len(games_df)}", border=True)
m4.metric("Pair flows", f"{len(flows_df)}", border=True)

st.divider()
tab1, tab2 = st.tabs(["Проверка по ID", "Топ подозрительных"])

with tab1:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("ID игрока")
        default_id = int(db_df["_player_id"].iloc[0]) if len(db_df) else 0
        player_id = st.number_input("Введите ID", min_value=0, value=default_id, step=1)
        run = st.button("Проверить", type="primary", use_container_width=True)

    with right:
        st.subheader("Что делает fast-версия")
        st.markdown(
            "- Парсинг/flows/sessions индексы считаются 1 раз и кэшируются.\n"
            "- Проверка ID — только быстрые lookup (без повторных groupby/фильтров по game_id)."
        )

    if not run:
        st.stop()

    db_sum, by_week = db_summary_for_player(db_period, int(player_id))
    if db_sum is None:
        st.error("Игрок не найден в DB по выбранному периоду.")
        st.stop()

    ring_s = games_idx.get("player_game_series", {}).get(("RING", int(player_id)))
    tour_s = games_idx.get("player_game_series", {}).get(("TOURNAMENT", int(player_id)))
    coverage = {
        "ring_games": int(len(ring_s)) if ring_s is not None else 0,
        "tour_games": int(len(tour_s)) if tour_s is not None else 0,
    }

    cop_ring = coplay_features_fast(int(player_id), games_idx, "RING")
    cop_tour = coplay_features_fast(int(player_id), games_idx, "TOURNAMENT")
    trf_ring = transfer_features_fast(int(player_id), games_idx, "RING")
    trf_tour = transfer_features_fast(int(player_id), games_idx, "TOURNAMENT")

    score, decision, main_risk, reasons = score_player(db_df, db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)

    st.divider()
    a, b, c = st.columns([1.2, 1, 1], gap="small")
    with a:
        st.subheader("Итог")
    b.metric("Risk score", f"{score}/100", border=True)
    c.metric("Decision", decision, border=True)

    if decision == "APPROVE":
        st.success("ВЫВОД: РАЗРЕШИТЬ")
    elif decision == "FAST_CHECK":
        st.warning("ВЫВОД: БЫСТРАЯ ПРОВЕРКА")
    else:
        st.error("ВЫВОД: РУЧНАЯ ПРОВЕРКА СБ")

    tabs = st.tabs(["Кратко", "DB", "Games", "Пары/источники"])

    with tabs[0]:
        st.subheader("Главный риск")
        st.info(main_risk)
        st.subheader("Причины")
        for r in reasons[:18]:
            st.markdown(f"- {r}")

    with tabs[1]:
        st.subheader("DB: агрегаты по периоду")
        x1, x2, x3, x4 = st.columns(4, gap="small")
        x1.metric("J: итог (+события)", fmt_money(db_sum["j_total"]), border=True)
        x2.metric("O: выигрыш общий", fmt_money(db_sum["p_total"]), border=True)
        x3.metric("Ring", fmt_money(db_sum["p_ring"]), border=True)
        x4.metric("MTT/SNG", fmt_money(db_sum["p_mtt"]), border=True)

        y1, y2, y3, y4 = st.columns(4, gap="small")
        y1.metric("Комиссия total", fmt_money(db_sum["comm_total"]), border=True)
        y2.metric("Комиссия PPSR", fmt_money(db_sum["comm_ppsr"]), border=True)
        y3.metric("J-O (события)", fmt_money(db_sum["events_delta"]), border=True)
        y4.metric("Ring/PPSR комис.", "NaN" if pd.isna(db_sum["ring_over_comm_ppsr"]) else f"{db_sum['ring_over_comm_ppsr']:.1f}x", border=True)

        st.subheader("По неделям (суммы)")
        show = by_week.rename(columns={"_week": "Неделя"}).copy()
        st.dataframe(show.sort_values("Неделя", ascending=False), use_container_width=True)

    with tabs[2]:
        st.subheader("Games: co-play")
        st.markdown(
            f"- RING: сессий={cop_ring['sessions_count']}, HU={cop_ring['hu_sessions']}, ≤3 игроков={cop_ring['sh_sessions']}, "
            f"уникальных оппонентов={cop_ring['unique_opponents']}, топ-1 доля={cop_ring['top1_coplay_share']:.0%}, топ-2 доля={cop_ring['top2_coplay_share']:.0%}."
        )
        st.markdown(
            f"- TOURNAMENT: сессий={cop_tour['sessions_count']}, уникальных оппонентов={cop_tour['unique_opponents']}, "
            f"топ-1 доля={cop_tour['top1_coplay_share']:.0%}."
        )

        st.subheader("Games: net-flow (RING по 'От соперников')")
        st.markdown(
            f"- RING: игр={trf_ring['target_games']}, сумма flow_win={fmt_money(trf_ring['target_total_flow_win'])}, "
            f"топ-партнёр={trf_ring['top_net_partner']}, net≈{fmt_money(trf_ring['top_net'])}, "
            f"gross≈{fmt_money(trf_ring['top_gross_with_partner'])}, доля пары≈{trf_ring['top_partner_share']:.0%}, "
            f"one_sided≈{trf_ring['one_sidedness']:.0%}."
        )
        st.markdown(
            f"- TOURNAMENT: игр={trf_tour['target_games']}, сумма flow_win={fmt_money(trf_tour['target_total_flow_win'])}, "
            f"топ-партнёр={trf_tour['top_net_partner']}, net≈{fmt_money(trf_tour['top_net'])}, "
            f"gross≈{fmt_money(trf_tour['top_gross_with_partner'])}, доля пары≈{trf_tour['top_partner_share']:.0%}, "
            f"one_sided≈{trf_tour['one_sidedness']:.0%}."
        )

    with tabs[3]:
        st.subheader("Co-play партнёры (RING)")
        if cop_ring["top_partners"]:
            st.dataframe(pd.DataFrame(cop_ring["top_partners"], columns=["partner_id", "coplay_sessions"]), use_container_width=True)
        else:
            st.info("Нет данных по партнёрам в RING.")

        st.subheader("Net inflows (RING): кто 'кормит' игрока")
        if trf_ring["top_inflows"]:
            st.dataframe(pd.DataFrame(trf_ring["top_inflows"], columns=["from_player", "amount_to_target"]), use_container_width=True)
        else:
            st.info("Нет выраженных inflow источников.")

        st.subheader("Net outflows (RING): кого игрок 'кормит'")
        if trf_ring["top_outflows"]:
            st.dataframe(pd.DataFrame(trf_ring["top_outflows"], columns=["to_player", "amount_from_target"]), use_container_width=True)
        else:
            st.info("Нет выраженных outflow направлений.")

with tab2:
    st.subheader("Топ подозрительных (по выбранному периоду)")
    st.caption("Это приоритизация для СБ: кого проверять первым. Не является доказательством нарушения.")

    colA, colB = st.columns([1, 1])
    with colA:
        top_n = st.number_input("Сколько показать", min_value=5, max_value=200, value=30, step=5)
    with colB:
        build = st.button("Посчитать ТОП", type="primary", use_container_width=True)

    if not build:
        st.stop()

    top_df = cached_top_suspicious(db_period, games_idx, int(top_n))

    if top_df.empty:
        st.info("Нет данных для построения ТОП (или период пустой).")
        st.stop()

    show = top_df.copy()
    show["flow_ring_partner_share"] = show["flow_ring_partner_share"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "NaN")
    show["flow_ring_dir"] = show["flow_ring_dir"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "NaN")
    st.dataframe(show, use_container_width=True)

    csv_bytes = top_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Скачать ТОП в CSV", data=csv_bytes, file_name="top_suspicious.csv", mime="text/csv")
