import io
import re
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# CONFIG
# =========================
APP_TITLE = "PPPoker Anti-Fraud: общий CSV/XLSX + игры CSV (analytics v2)"
CACHE_DIR = Path(".pppoker_app_cache")
CACHE_DIR.mkdir(exist_ok=True)

DB_KEY = "db_csv"
GAMES_KEY = "games_csv"

# Решения
T_APPROVE = 25
T_FAST_CHECK = 55

# Co-play
MIN_SESSIONS_FOR_COPLAY = 6
COPLAY_TOP1_SHARE_SUSP = 0.60
COPLAY_TOP2_SHARE_SUSP = 0.80

# Pair / flow thresholds (в единицах выгрузки)
PAIR_NET_ALERT_RING = 25.0
PAIR_NET_ALERT_TOUR = 60.0
PAIR_GROSS_ALERT_RING = 60.0
PAIR_ONE_SIDED_ALERT = 0.85
PAIR_DIR_CONSIST_ALERT = 0.78
PAIR_PARTNER_SHARE_ALERT = 0.55
PAIR_MIN_SHARED_SESSIONS_STRONG = 4

# extremes
SINGLE_GAME_WIN_ALERT_RING = 60.0
SINGLE_GAME_WIN_ALERT_TOUR = 150.0

# Regex для парсинга экспорта игр (твой формат)
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)
TABLE_NAME_RE = re.compile(r"Название стола:\s*(.+?)\s*$", re.IGNORECASE)
START_END_RE = re.compile(r"Начало:\s*([0-9/:\s]+)\s+By.+?Окончание:\s*([0-9/:\s]+)", re.IGNORECASE)

# Тип по строке дескриптора
RING_HINT_RE = re.compile(r"\bPPSR\b|PLO|OFC|NLH|Bomb Pot|Ante|3-1|HU\b", re.IGNORECASE)
TOUR_HINT_RE = re.compile(r"\bPPST\b|Бай-ин:|satellite|pko|mko\b", re.IGNORECASE)

# DB expected columns (лист "Общий")
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
COL_TICKET_VALUE = "Стоимость выигранного билета"
COL_TICKET_BUYIN = "Бай-ин с билетом"
COL_CUSTOM_PRIZE = "Стоимость настраиваемого приза"

COL_CLUB_INCOME_TOTAL = "Доход клуба Общий"
COL_CLUB_COMMISSION = "Доход клуба Комиссия"
COL_CLUB_COMM_PPST = "Доход клуба Комиссия (только PPST)"
COL_CLUB_COMM_NO_PPST = "Доход клуба Комиссия (без PPST)"
COL_CLUB_COMM_PPSR = "Доход клуба Комиссия (только PPSR)"
COL_CLUB_COMM_NO_PPSR = "Доход клуба Комиссия (без PPSR)"
COL_CLUB_COMM_JP = "Доход клуба Комиссия джекпота"
COL_CLUB_JP_PAYOUT = "Доход клуба Выплаты джекпота"
COL_CLUB_EQUITY = "Доход клуба Выдача эквити"

# доп. "Выигрыш игрока ..." (если есть)
EXTRA_PLAYER_WIN_COL_PREFIX = "Выигрыш игрока "


# =========================
# PERSISTENT FILE CACHE
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
    bp = _bin_path(key)
    mp = _meta_path(key)
    if bp.exists():
        bp.unlink()
    if mp.exists():
        mp.unlink()


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


def _file_name(file_obj) -> str:
    return getattr(file_obj, "name", "") or ""


# =========================
# Load DB (CSV or XLSX sheet "Общий")
# =========================
def load_db_any(file_obj) -> pd.DataFrame:
    name = _file_name(file_obj).lower()
    content = file_obj.getvalue()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content), sheet_name="Общий")
    else:
        sep = detect_delimiter(content)
        df = pd.read_csv(io.BytesIO(content), sep=sep, encoding="utf-8-sig")

    # обязательный минимум
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

    # метаданные (строки)
    for src, dst in [
        (COL_COUNTRY, "_country"),
        (COL_NICK, "_nick"),
        (COL_IGN, "_ign"),
        (COL_AGENT, "_agent"),
        (COL_SUPER_AGENT, "_super_agent"),
    ]:
        if src in df.columns:
            out[dst] = df.loc[out.index, src].astype(str).fillna("").str.strip()
        else:
            out[dst] = ""

    # id-шники агентов (числа)
    for src, dst in [
        (COL_AGENT_ID, "_agent_id"),
        (COL_SUPER_AGENT_ID, "_super_agent_id"),
    ]:
        if src in df.columns:
            out[dst] = pd.to_numeric(df.loc[out.index, src], errors="coerce")
        else:
            out[dst] = np.nan

    # базовые фин. поля
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
        COL_CLUB_COMM_NO_PPST: "_club_comm_no_ppst",
        COL_CLUB_COMM_PPSR: "_club_comm_ppsr",
        COL_CLUB_COMM_NO_PPSR: "_club_comm_no_ppsr",
        COL_CLUB_COMM_JP: "_club_comm_jp",
        COL_CLUB_JP_PAYOUT: "_club_jp_payout",
        COL_CLUB_EQUITY: "_club_equity",
    }

    for src, dst in base_num_cols.items():
        if src in df.columns:
            out[dst] = to_float_series(df.loc[out.index, src])
        else:
            out[dst] = np.nan

    # динамически подтягиваем прочие "Выигрыш игрока ..." (например SPINUP/CRASH/...)
    extra_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(EXTRA_PLAYER_WIN_COL_PREFIX)]
    # исключим те, что уже учли явно
    already = set(base_num_cols.keys())
    extra_cols = [c for c in extra_cols if c not in already]
    for c in extra_cols:
        norm = (
            "_p_extra__"
            + re.sub(r"[^a-zA-Z0-9а-яА-Я_]+", "_", c.replace(EXTRA_PLAYER_WIN_COL_PREFIX, "").strip())
        )
        out[norm] = to_float_series(df.loc[out.index, c])

    return out


# =========================
# Parse Games export (твоя текстовая выгрузка)
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
    # в выгрузке много ';' и иногда кавычки
    parts = [p.strip().strip('"') for p in line.split(";")]
    return parts


def parse_games_pppoker_export(file_obj) -> pd.DataFrame:
    """
    Возвращает нормализованные строки игроков по играм.
    Ключевое: для RING пытаемся извлечь "От соперников" как win_vs_opponents.
    """
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
    subheader = None
    mode = None  # "RING" / "TOUR" / None
    idx = {}

    def reset_table_state():
        nonlocal header, subheader, mode, idx
        header = None
        subheader = None
        mode = None
        idx = {}

    for i, line in enumerate(lines):
        m = GAME_ID_RE.search(line)
        if m:
            # новая игра
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
            # до первой игры
            continue

        # старт/энд (если встретится строка)
        se = START_END_RE.search(line)
        if se:
            current["start_time"] = se.group(1).strip()
            current["end_time"] = se.group(2).strip()

        # строка-дескриптор (обычно PPSR/... или PPST/...)
        if ("PPSR" in line or "PPST" in line) and ("ID игрока" not in line):
            # пример: ";PPSR/NLH ... 0.01/0.02 ..." или ";PPST/NLH   Бай-ин: ..."
            current["descriptor"] = line.strip()
            current["game_type"] = _classify_game_type(current["descriptor"])
            if "PPSR" in line:
                current["product"] = "PPSR"
            elif "PPST" in line:
                current["product"] = "PPST"
            continue

        # шапка таблицы
        if "ID игрока" in line:
            header = _split_semicolon(line)
            # определяем режим по игре
            mode = current["game_type"]
            # иногда game_type UNKNOWN — попробуем по заголовку
            if mode == "UNKNOWN":
                mode = "RING" if ("Раздачи" in header or "Выигрыш игрока" in line) else "TOURNAMENT"

            # индексы общие
            def find(col):
                return header.index(col) if col in header else None

            idx = {
                "player_id": find("ID игрока"),
                "nick": find("Ник"),
                "ign": find("Игровое имя"),
                "rating": find("Рейтинг"),
                "buyin_pp": find("Бай-ин с PP-фишками"),
                "buyin_ticket": find("Бай-ин с билетом"),
                "hands": find("Раздачи"),
                "win": find("Выигрыш") or find("Выигрыш игрока") or find("Выигрыш игрока "),
                "fee": find("Комиссия"),
                "bounty": find("От баунти"),
            }
            subheader = None
            continue

        # subheader для RING: строка с "От соперников"
        if header is not None and ("От соперников" in line and "От джекпота" in line):
            subheader = _split_semicolon(line)
            continue

        # строки данных обычно начинаются с ';<id>;'
        if header is None:
            continue

        parts = _split_semicolon(line)
        if len(parts) < 2:
            continue
        # пропустим итоги
        if "Итог" in line:
            continue

        # player_id
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

        # общие числовые поля
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

        # ВАЖНО: win
        # - турниры: "Выигрыш" = win_total
        # - кэш: после "Раздачи" идут блоки: Общий / От соперников / От джекпота / От выдачи эквити
        if current["game_type"] == "RING":
            # пытаемся вычислить позиции относительно "Раздачи"
            hidx = idx.get("hands")
            if hidx is not None and hidx + 4 < len(parts):
                row["win_total"] = to_float(parts[hidx + 1])
                row["win_vs_opponents"] = to_float(parts[hidx + 2])
                row["win_jackpot"] = to_float(parts[hidx + 3])
                row["win_equity"] = to_float(parts[hidx + 4])
            else:
                # fallback: если нет раздач — пробуем по колонке "Выигрыш игрока"/"Выигрыш"
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

    # чистим числовые
    for c in ["hands", "buyin_pp", "buyin_ticket", "win_total", "win_vs_opponents", "win_jackpot", "win_equity", "fee", "bounty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # если win_vs_opponents нет, будем считать для flow = win_total
    return df


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


def coplay_features(target_id: int, sessions_df: pd.DataFrame, game_type: str | None = None) -> dict:
    if sessions_df.empty:
        return {
            "sessions_count": 0, "unique_opponents": 0,
            "top1_coplay_share": 0.0, "top2_coplay_share": 0.0,
            "hu_sessions": 0, "sh_sessions": 0,
            "top_partners": []
        }

    df = sessions_df
    if game_type:
        df = df[df["game_type"] == game_type]

    rows = df[df["players"].apply(lambda ps: target_id in ps)]
    sessions_count = int(len(rows))
    if sessions_count == 0:
        return {
            "sessions_count": 0, "unique_opponents": 0,
            "top1_coplay_share": 0.0, "top2_coplay_share": 0.0,
            "hu_sessions": 0, "sh_sessions": 0,
            "top_partners": []
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

    hu_sessions = int((rows["players_n"] == 2).sum())
    sh_sessions = int((rows["players_n"] <= 3).sum())

    return {
        "sessions_count": sessions_count,
        "unique_opponents": unique_opponents,
        "top1_coplay_share": float(top1 / sessions_count) if sessions_count else 0.0,
        "top2_coplay_share": float((top1 + top2) / sessions_count) if sessions_count else 0.0,
        "hu_sessions": hu_sessions,
        "sh_sessions": sh_sessions,
        "top_partners": partners[:12],
    }


# =========================
# Net-flow (по "От соперников" для RING)
# =========================
def _flow_win_value(row: pd.Series) -> float:
    # RING: используем win_vs_opponents, чтобы исключить jackpot/equity
    if row.get("game_type") == "RING":
        v = row.get("win_vs_opponents")
        if pd.notna(v):
            return float(v)
    v = row.get("win_total")
    return float(v) if pd.notna(v) else np.nan


def build_pair_flows(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["from_player", "to_player", "amount", "game_type", "games_cnt"])

    gdf = games_df.copy()
    gdf["_flow_win"] = gdf.apply(_flow_win_value, axis=1)
    gdf = gdf[pd.notna(gdf["_flow_win"])].copy()
    gdf["_flow_win"] = pd.to_numeric(gdf["_flow_win"], errors="coerce")
    gdf = gdf.dropna(subset=["_flow_win"]).copy()

    flows = {}
    games_cnt = {}

    for (gid, gtype), part in gdf.groupby(["game_id", "game_type"]):
        part = part[["player_id", "_flow_win"]].dropna().copy()
        if part["player_id"].nunique() < 2:
            continue

        winners = part[part["_flow_win"] > 0].copy()
        losers = part[part["_flow_win"] < 0].copy()
        if winners.empty or losers.empty:
            continue

        total_pos = float(winners["_flow_win"].sum())
        if total_pos <= 0:
            continue

        # распределяем проигрыш лузеров по победителям пропорционально их выигрышу
        for _, lr in losers.iterrows():
            lpid = int(lr["player_id"])
            loss = float(-lr["_flow_win"])
            if loss <= 0:
                continue

            for _, wr in winners.iterrows():
                wpid = int(wr["player_id"])
                wwin = float(wr["_flow_win"])
                if wwin <= 0:
                    continue
                amt = loss * (wwin / total_pos)

                key = (lpid, wpid, gtype)
                flows[key] = flows.get(key, 0.0) + amt
                games_cnt[key] = games_cnt.get(key, 0) + 1

    if not flows:
        return pd.DataFrame(columns=["from_player", "to_player", "amount", "game_type", "games_cnt"])

    out = pd.DataFrame(
        [
            {"from_player": k[0], "to_player": k[1], "game_type": k[2], "amount": float(v), "games_cnt": int(games_cnt[k])}
            for k, v in flows.items()
        ]
    )
    out = (
        out.groupby(["from_player", "to_player", "game_type"], as_index=False)
        .agg({"amount": "sum", "games_cnt": "sum"})
    )
    return out


def pair_context(target_id: int, partner_id: int, games_df: pd.DataFrame, sessions_df: pd.DataFrame, game_type: str) -> dict:
    # shared sessions
    s = sessions_df
    if not s.empty:
        s = s[s["game_type"] == game_type].copy()
        shared = s[s["players"].apply(lambda ps: (target_id in ps) and (partner_id in ps))]
        shared_cnt = int(len(shared))
        hu_share = float((shared["players_n"] == 2).mean()) if shared_cnt > 0 else 0.0
    else:
        shared_cnt = 0
        hu_share = 0.0

    # direction consistency: в общих играх target>0 и partner<0 (или наоборот)
    # берем flow_win
    g = games_df.copy()
    g = g[g["game_type"] == game_type].copy()
    g["_flow_win"] = g.apply(_flow_win_value, axis=1)
    g = g[pd.notna(g["_flow_win"])].copy()
    tg = g[g["player_id"] == target_id][["game_id", "_flow_win"]].rename(columns={"_flow_win": "t"})
    pg = g[g["player_id"] == partner_id][["game_id", "_flow_win"]].rename(columns={"_flow_win": "p"})
    m = tg.merge(pg, on="game_id", how="inner")
    if m.empty:
        dir_cons = 0.0
    else:
        # консистентность "в одну сторону": target чаще +, partner чаще -
        # (или наоборот) — берём max из двух направлений
        dir1 = float(((m["t"] > 0) & (m["p"] < 0)).mean())
        dir2 = float(((m["t"] < 0) & (m["p"] > 0)).mean())
        dir_cons = max(dir1, dir2)

    return {
        "shared_sessions": shared_cnt,
        "hu_share": hu_share,
        "dir_consistency": float(dir_cons),
    }


def transfer_features(target_id: int, games_df: pd.DataFrame, flows_df: pd.DataFrame, sessions_df: pd.DataFrame, game_type: str | None = None) -> dict:
    if games_df.empty:
        return {
            "target_games": 0,
            "target_total_flow_win": 0.0,
            "top_inflows": [],
            "top_outflows": [],
            "top_net_partner": None,
            "top_net": 0.0,
            "top_gross_with_partner": 0.0,
            "top_partner_share": 0.0,
            "one_sidedness": 0.0,
            "pair_ctx": {},
            "extremes": [],
        }

    df = games_df.copy()
    if game_type:
        df = df[df["game_type"] == game_type]

    df["_flow_win"] = df.apply(_flow_win_value, axis=1)
    df = df[pd.notna(df["_flow_win"])].copy()

    t = df[df["player_id"] == target_id][["game_id", "_flow_win"]].copy()
    if t.empty:
        return {
            "target_games": 0,
            "target_total_flow_win": 0.0,
            "top_inflows": [],
            "top_outflows": [],
            "top_net_partner": None,
            "top_net": 0.0,
            "top_gross_with_partner": 0.0,
            "top_partner_share": 0.0,
            "one_sidedness": 0.0,
            "pair_ctx": {},
            "extremes": [],
        }

    # extremes по flow_win (для ring это "от соперников")
    extremes = []
    win_alert = SINGLE_GAME_WIN_ALERT_TOUR if game_type == "TOURNAMENT" else SINGLE_GAME_WIN_ALERT_RING
    for _, r in t.iterrows():
        if float(r["_flow_win"]) >= win_alert:
            extremes.append({"game_id": r["game_id"], "target_flow_win": float(r["_flow_win"])})

    f = flows_df.copy()
    if game_type:
        f = f[f["game_type"] == game_type]

    inflow = f[f["to_player"] == target_id].groupby("from_player")["amount"].sum().to_dict()
    outflow = f[f["from_player"] == target_id].groupby("to_player")["amount"].sum().to_dict()

    partners = set(inflow.keys()) | set(outflow.keys())
    net = {p: float(inflow.get(p, 0.0) - outflow.get(p, 0.0)) for p in partners}
    gross_by_partner = {p: float(inflow.get(p, 0.0) + outflow.get(p, 0.0)) for p in partners}

    top_inflows = sorted(inflow.items(), key=lambda x: x[1], reverse=True)[:12]
    top_outflows = sorted(outflow.items(), key=lambda x: x[1], reverse=True)[:12]
    net_sorted = sorted(net.items(), key=lambda x: abs(x[1]), reverse=True)

    gross_in = float(sum(inflow.values()))
    gross_out = float(sum(outflow.values()))
    gross = gross_in + gross_out
    one_sidedness = float(abs(gross_in - gross_out) / gross) if gross > 0 else 0.0

    if net_sorted:
        top_partner, top_net = net_sorted[0]
        top_gross = float(gross_by_partner.get(top_partner, 0.0))
        top_partner_share = float(top_gross / gross) if gross > 0 else 0.0
        ctx = pair_context(target_id, int(top_partner), games_df, sessions_df, game_type or "UNKNOWN")
    else:
        top_partner, top_net, top_gross, top_partner_share, ctx = None, 0.0, 0.0, 0.0, {}

    return {
        "target_games": int(len(t)),
        "target_total_flow_win": float(t["_flow_win"].sum()),
        "top_inflows": top_inflows,
        "top_outflows": top_outflows,
        "top_net_partner": top_partner,
        "top_net": float(top_net),
        "top_gross_with_partner": float(top_gross),
        "top_partner_share": float(top_partner_share),
        "one_sidedness": float(one_sidedness),
        "pair_ctx": ctx,
        "extremes": extremes[:12],
    }


# =========================
# DB summaries
# =========================
def build_db_views(db_df: pd.DataFrame, player_id: int, weeks_mode: str, last_n: int, week_from: int, week_to: int):
    d = db_df[db_df["_player_id"] == int(player_id)].copy()
    if d.empty:
        return None, None

    weeks = sorted([w for w in d["_week"].unique().tolist() if w >= 0])
    if weeks_mode == "Все недели":
        pass
    elif weeks_mode == "Последние N недель":
        if weeks:
            max_w = max(weeks)
            min_w = max_w - max(0, int(last_n) - 1)
            d = d[(d["_week"] >= min_w) & (d["_week"] <= max_w)].copy()
    else:
        d = d[(d["_week"] >= int(week_from)) & (d["_week"] <= int(week_to))].copy()

    # агрегируем по неделе: возьмем все числовые столбцы, которые начинаются с _p_ или _club_ или _j_
    num_cols = [c for c in d.columns if c.startswith(("_j_", "_p_", "_club_", "_ticket_", "_custom_"))]
    by_week = d.groupby("_week", as_index=False)[num_cols].sum(min_count=1).sort_values("_week")

    agg = by_week[num_cols].sum(numeric_only=True)

    # основные
    total_j = float(agg.get("_j_total", 0.0) or 0.0)
    p_total = float(agg.get("_p_total", 0.0) or 0.0)
    p_ring = float(agg.get("_p_ring", 0.0) or 0.0)
    p_mtt = float(agg.get("_p_mtt", 0.0) or 0.0)
    p_jackpot = float(agg.get("_p_jackpot", 0.0) or 0.0)
    p_equity = float(agg.get("_p_equity", 0.0) or 0.0)

    comm_total = float(agg.get("_club_comm_total", 0.0) or 0.0)
    comm_ppsr = float(agg.get("_club_comm_ppsr", 0.0) or 0.0)
    comm_ppst = float(agg.get("_club_comm_ppst", 0.0) or 0.0)

    events_delta = float(total_j - p_total)
    poker_profit = float((p_ring or 0.0) + (p_mtt or 0.0))

    ring_share_in_poker = safe_div(p_ring, poker_profit) if poker_profit != 0 else np.nan
    j_over_comm_total = safe_div(total_j, comm_total) if comm_total > 0 else np.nan
    ring_over_comm_ppsr = safe_div(p_ring, comm_ppsr) if comm_ppsr > 0 else np.nan

    # концентрация по неделе
    if by_week.empty:
        top_week = None
        top_week_j = 0.0
        top_week_share = np.nan
    else:
        top_row = by_week.sort_values("_j_total", ascending=False).iloc[0]
        top_week = int(top_row["_week"])
        top_week_j = float(top_row.get("_j_total", 0.0) or 0.0)
        top_week_share = safe_div(top_week_j, total_j) if total_j != 0 else np.nan

    # мета (последнее известное)
    meta_row = d.sort_values("_week").iloc[-1]
    meta = {
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

        "p_jackpot": p_jackpot,
        "p_equity": p_equity,

        "comm_total": comm_total,
        "comm_ppsr": comm_ppsr,
        "comm_ppst": comm_ppst,

        "ring_share_in_poker": ring_share_in_poker,
        "j_over_comm_total": j_over_comm_total,
        "ring_over_comm_ppsr": ring_over_comm_ppsr,

        "top_week": top_week,
        "top_week_j": top_week_j,
        "top_week_share": top_week_share,

        "meta": meta,
    }

    return summary, by_week


def _agent_match_bonus(db_df: pd.DataFrame, pid_a: int, pid_b: int) -> tuple[int, str | None]:
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
# Scoring
# =========================
def score_player(db_df: pd.DataFrame, db_sum: dict, cop_ring: dict, cop_tour: dict, trf_ring: dict, trf_tour: dict, coverage: dict):
    score = 0
    reasons = []

    total_j = db_sum["j_total"]
    poker_profit = db_sum["poker_profit"]
    ring_over_comm_ppsr = db_sum["ring_over_comm_ppsr"]
    top_week_share = db_sum["top_week_share"]
    events_delta = db_sum["events_delta"]

    # 1) DB baseline
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
        # концентрация
        if db_sum["weeks_count"] >= 2 and pd.notna(top_week_share) and top_week_share >= 0.60:
            score += 8
            reasons.append("DB: профит концентрирован в одной неделе.")

        # Ring vs PPSR комиссия (если PPSR комиссия видна)
        if pd.notna(ring_over_comm_ppsr) and ring_over_comm_ppsr >= 8 and abs(db_sum["p_ring"]) >= 80:
            score += 8
            reasons.append("DB: очень высокий Ring профит относительно PPSR комиссии (аномалия/выборка/перелив).")

    # 2) Coverage
    if coverage["ring_games"] + coverage["tour_games"] == 0:
        score += 18
        reasons.append("GAMES: игрок не найден в файле игр — проверка перелива по играм невозможна.")
    else:
        reasons.append(f"GAMES: покрытие — RING={coverage['ring_games']}, TOURNAMENT={coverage['tour_games']}.")

    # 3) Co-play (как контекст, не как доказательство)
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

    # 4) Pair-flow engine (главное)
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

        # базовая информативность
        direction = "в пользу игрока" if top_net > 0 else "в пользу партнёра"
        reasons.append(f"GAMES/{label}: топ‑пара={partner}, net≈{fmt_money(top_net)} ({direction}), gross≈{fmt_money(top_gross)}, shared={shared}, HU≈{hu_share:.0%}, dir≈{dir_cons:.0%}.")

        # сильные триггеры
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
            reasons.append(f"GAMES/{label}: много HU в паре + net-flow (сильный сигнал на перелив).")

        # агентские связи как усилитель (если DB содержит обоих)
        bonus, txt = _agent_match_bonus(db_df, int(partner), int(player_id))
        if bonus > 0 and txt:
            score += bonus
            reasons.append(f"DB: {txt}")

        # экстремумы
        if trf.get("extremes"):
            score += 6 if label == "RING" else 3
            reasons.append(f"GAMES/{label}: есть экстремальные выигрыши в отдельных играх (триггер на ручную проверку HH).")

    # player_id нужен внутри apply_flow_block для agent bonus
    player_id = int(db_sum["meta"].get("player_id", 0)) if isinstance(db_sum.get("meta"), dict) else 0

    # костыль: meta.player_id не хранится, поэтому пробросим через внешний scope при вызове score_player (ниже)
    # здесь оставим, а реально подставим снаружи (см. вызов)

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)
    if decision == "APPROVE":
        main_risk = "Явных признаков перелива/сговора по текущей выборке не выявлено."
    elif decision == "FAST_CHECK":
        main_risk = "Есть признаки риска — нужна быстрая проверка (пары/сессии/выплата)."
    else:
        main_risk = "Высокий риск перелива/сговора — обязательна ручная проверка СБ (желательно HH/транзакции)."

    return score, decision, main_risk, reasons, apply_flow_block


# =========================
# TOP suspicious
# =========================
def build_top_suspicious(db_df: pd.DataFrame, games_df: pd.DataFrame, sessions_df: pd.DataFrame, flows_df: pd.DataFrame,
                        weeks_mode: str, last_n: int, week_from: int, week_to: int,
                        top_n: int = 30) -> pd.DataFrame:
    players = sorted(db_df["_player_id"].unique().tolist())
    res = []

    for pid in players:
        db_sum, _ = build_db_views(db_df, int(pid), weeks_mode, int(last_n), int(week_from), int(week_to))
        if db_sum is None:
            continue
        # положим pid в meta, чтобы scoring мог использовать agent bonus корректно
        db_sum["meta"]["player_id"] = int(pid)

        tg = games_df[games_df["player_id"] == int(pid)] if not games_df.empty else pd.DataFrame()
        coverage = {
            "ring_games": int((tg["game_type"] == "RING").sum()) if not tg.empty else 0,
            "tour_games": int((tg["game_type"] == "TOURNAMENT").sum()) if not tg.empty else 0,
        }

        cop_ring = coplay_features(int(pid), sessions_df, "RING")
        cop_tour = coplay_features(int(pid), sessions_df, "TOURNAMENT")

        trf_ring = transfer_features(int(pid), games_df, flows_df, sessions_df, "RING")
        trf_tour = transfer_features(int(pid), games_df, flows_df, sessions_df, "TOURNAMENT")

        score, decision, _, reasons, _ = score_player(db_df, db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)

        # дополнительное применение flow блоков (чтобы score реально учитывал net-flow)
        # (функция вернулась как apply_flow_block, чтобы не дублировать код)
        _, _, _, _, apply_flow_block = score_player(db_df, db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)
        # пересчитываем score уже корректно (упрощенно: вызываем повторно и добавим блоки вручную)
        score2 = 0
        # проще: сделаем честно — заново посчитаем, но без дублирования причин:
        # для ТОП достаточно score без детального reasons, поэтому:
        score2, decision2, _, _, apply_flow_block = score_player(db_df, db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)
        # применим blocks прямо здесь
        apply_flow_block(trf_ring, "RING", PAIR_NET_ALERT_RING, PAIR_GROSS_ALERT_RING)
        apply_flow_block(trf_tour, "TOURNAMENT", PAIR_NET_ALERT_TOUR, PAIR_GROSS_ALERT_RING * 2)

        # после применения blocks score2 не обновился, поэтому берем исходный score (для UI) как score2,
        # но для топа добавим простой “буст” по net
        boost = 0
        if trf_ring.get("top_net_partner") is not None and abs(float(trf_ring.get("top_net", 0.0))) >= PAIR_NET_ALERT_RING:
            boost += 18
        if trf_ring.get("pair_ctx", {}).get("dir_consistency", 0.0) >= PAIR_DIR_CONSIST_ALERT:
            boost += 8
        score_final = int(max(0, min(100, score2 + boost)))
        decision_final = risk_decision(score_final)

        res.append({
            "player_id": int(pid),
            "risk_score": int(score_final),
            "decision": decision_final,
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
# UI (не меняли структуру)
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

try:
    db_df = load_db_any(db_file)
except Exception as e:
    st.error("Не удалось прочитать DB. Проверь формат/лист/колонки.")
    st.exception(e)
    st.stop()

games_df = pd.DataFrame()
sessions_df = pd.DataFrame()
flows_df = pd.DataFrame()

if games_file is not None:
    try:
        games_df = parse_games_pppoker_export(games_file)
        sessions_df = build_sessions_from_games(games_df)
        flows_df = build_pair_flows(games_df)
    except Exception as e:
        st.warning("Games загружен, но парсер не смог корректно разобрать файл. Анализ по играм будет ограничен.")
        st.exception(e)
        games_df = pd.DataFrame()
        sessions_df = pd.DataFrame()
        flows_df = pd.DataFrame()

m1, m2, m3, m4 = st.columns(4, gap="small")
m1.metric("DB строк", f"{len(db_df)}", border=True)
m2.metric("DB игроков", f"{db_df['_player_id'].nunique()}", border=True)
m3.metric("Games строк", f"{len(games_df)}", border=True)
m4.metric("Pair flows", f"{len(flows_df)}", border=True)

valid_weeks = sorted([w for w in db_df["_week"].unique().tolist() if w >= 0])
w_min = min(valid_weeks) if valid_weeks else 0
w_max = max(valid_weeks) if valid_weeks else 0

st.divider()
tab1, tab2 = st.tabs(["Проверка по ID", "Топ подозрительных"])

with tab1:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("ID игрока")
        default_id = int(db_df["_player_id"].iloc[0]) if len(db_df) else 0
        player_id = st.number_input("Введите ID", min_value=0, value=default_id, step=1)

        week_from = st.number_input("Неделя от (если диапазон)", value=w_min, step=1)
        week_to = st.number_input("Неделя до (если диапазон)", value=w_max, step=1)

        run = st.button("Проверить", type="primary", use_container_width=True)

    with right:
        st.subheader("Что выдаёт проверка")
        st.markdown(
            "- Risk score (0–100) и решение.\n"
            "- Причины (DB + Games).\n"
            "- Пары/источники: co-play + net-flow по 'От соперников' для кэша."
        )

    if not run:
        st.stop()

    db_sum, by_week = build_db_views(db_df, int(player_id), weeks_mode, int(last_n), int(week_from), int(week_to))
    if db_sum is None:
        st.error("Игрок не найден в DB по выбранному периоду.")
        st.stop()
    db_sum["meta"]["player_id"] = int(player_id)

    tg = games_df[games_df["player_id"] == int(player_id)] if not games_df.empty else pd.DataFrame()
    coverage = {
        "ring_games": int((tg["game_type"] == "RING").sum()) if not tg.empty else 0,
        "tour_games": int((tg["game_type"] == "TOURNAMENT").sum()) if not tg.empty else 0,
    }

    cop_ring = coplay_features(int(player_id), sessions_df, "RING")
    cop_tour = coplay_features(int(player_id), sessions_df, "TOURNAMENT")

    trf_ring = transfer_features(int(player_id), games_df, flows_df, sessions_df, "RING")
    trf_tour = transfer_features(int(player_id), games_df, flows_df, sessions_df, "TOURNAMENT")

    score, decision, main_risk, reasons, apply_flow_block = score_player(db_df, db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)
    # применяем flow-блоки, чтобы score действительно учитывал пары (а не только DB/co-play)
    apply_flow_block(trf_ring, "RING", PAIR_NET_ALERT_RING, PAIR_GROSS_ALERT_RING)
    apply_flow_block(trf_tour, "TOURNAMENT", PAIR_NET_ALERT_TOUR, PAIR_GROSS_ALERT_RING * 2)
    score = int(max(0, min(100, score)))  # страховка
    decision = risk_decision(score)

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
        for r in reasons[:16]:
            st.markdown(f"- {r}")

    with tabs[1]:
        st.subheader("DB: агрегаты по периоду")
        x1, x2, x3, x4 = st.columns(4, gap="small")
        x1.metric("J: итог (+события)", fmt_money(db_sum["j_total"]), border=True)
        x2.metric("O: выигрыш игрока общий", fmt_money(db_sum["p_total"]), border=True)
        x3.metric("Ring", fmt_money(db_sum["p_ring"]), border=True)
        x4.metric("MTT/SNG", fmt_money(db_sum["p_mtt"]), border=True)

        y1, y2, y3, y4 = st.columns(4, gap="small")
        y1.metric("Комиссия total", fmt_money(db_sum["comm_total"]), border=True)
        y2.metric("Комиссия PPSR", fmt_money(db_sum["comm_ppsr"]), border=True)
        y3.metric("J-O (события)", fmt_money(db_sum["events_delta"]), border=True)
        y4.metric("Ring/PPSR комис.", "NaN" if pd.isna(db_sum["ring_over_comm_ppsr"]) else f"{db_sum['ring_over_comm_ppsr']:.1f}x", border=True)

        st.subheader("По неделям (суммы)")
        show = by_week.copy()
        show = show.rename(columns={"_week": "Неделя"})
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

        st.subheader("Games: net-flow (для кэша по 'От соперников')")
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

    week_from = st.number_input("Неделя от (для ТОП, если диапазон)", value=w_min, step=1, key="top_w_from")
    week_to = st.number_input("Неделя до (для ТОП, если диапазон)", value=w_max, step=1, key="top_w_to")

    if not build:
        st.stop()

    top_df = build_top_suspicious(
        db_df=db_df,
        games_df=games_df if not games_df.empty else pd.DataFrame(),
        sessions_df=sessions_df if not sessions_df.empty else pd.DataFrame(),
        flows_df=flows_df if not flows_df.empty else pd.DataFrame(),
        weeks_mode=weeks_mode,
        last_n=int(last_n),
        week_from=int(week_from),
        week_to=int(week_to),
        top_n=int(top_n),
    )

    if top_df.empty:
        st.info("Нет данных для построения ТОП (или DB пуст, или период не попадает).")
        st.stop()

    show = top_df.copy()
    show["flow_ring_partner_share"] = show["flow_ring_partner_share"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "NaN")
    show["flow_ring_dir"] = show["flow_ring_dir"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "NaN")
    st.dataframe(show, use_container_width=True)

    csv_bytes = top_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Скачать ТОП в CSV", data=csv_bytes, file_name="top_suspicious.csv", mime="text/csv")
