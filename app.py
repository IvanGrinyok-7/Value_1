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
APP_TITLE = "PPPoker Anti-Fraud (DB.xlsx + Games export) — optimized"
CACHE_DIR = Path(".pppoker_app_cache")
CACHE_DIR.mkdir(exist_ok=True)

DB_KEY = "db_file"
GAMES_KEY = "games_file"

# решения
T_APPROVE = 25
T_FAST_CHECK = 55

# сигналы
MIN_SESSIONS_FOR_COPLAY = 6
PAIR_DOMINANCE_ALERT = 0.70       # доля пары в обороте
PAIR_ONE_SIDED_ALERT = 0.85       # односторонность потока

# пороги net-flow (условные единицы PPPoker)
PAIR_NET_ALERT_RING = 25.0
PAIR_NET_ALERT_TOUR = 60.0

# быстрый “дампинг” за мало раздач
DUMP_HANDS_MAX = 25
DUMP_LOSS_ALERT = 25.0

# большие выигрыши (как “триггер посмотреть игру”)
SINGLE_GAME_WIN_ALERT_RING = 60.0
SINGLE_GAME_WIN_ALERT_TOUR = 150.0


# =========================
# Cache helpers
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
# Utils
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


def risk_decision(score: int) -> str:
    if score < T_APPROVE:
        return "APPROVE"
    if score < T_FAST_CHECK:
        return "FAST_CHECK"
    return "MANUAL_REVIEW"


def _norm(s: str) -> str:
    return str(s).strip().lower()


def pick_col(df: pd.DataFrame, variants: list[str]) -> str | None:
    cols = {_norm(c): c for c in df.columns}
    for v in variants:
        if _norm(v) in cols:
            return cols[_norm(v)]
    return None


# =========================
# DB loader (XLSX/CSV)
# =========================
DB_COLS = {
    "week": ["Номер недели", "Week"],
    "player_id": ["ID игрока", "Player ID", "player_id"],
    "country": ["Страна/регион", "Country"],
    "nick": ["Ник", "Nick"],
    "game_name": ["Игровое имя", "Game name", "Игровое Имя"],
    "agent": ["Агент", "Agent"],
    "agent_id": ["ID агента", "Agent ID"],
    "super_agent": ["Супер-агент", "Super-agent", "Super agent"],
    "super_agent_id": ["ID cупер-агента", "ID супер-агента", "Super-agent ID", "Super agent ID"],
    "j_total": ["Общий выигрыш игроков + События"],
    "player_win_total": ["Выигрыш игрока Общий"],
    "player_win_ring": ["Выигрыш игрока Ring Game"],
    "player_win_mtt": ["Выигрыш игрока MTT, SNG"],
    "club_comm_total": ["Доход клуба Комиссия"],
    "club_comm_ppsr": ["Доход клуба Комиссия (только PPSR)"],
    "club_comm_ppst": ["Доход клуба Комиссия (только PPST)"],
    "leader_ppsr": ["PPSR Лидерборд"],
    "leaders_rg": ["Список лидеров Ring Game"],
}


def load_db(file_obj) -> pd.DataFrame:
    content = file_obj.getvalue()
    name = getattr(file_obj, "name", "db")

    if name.lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(io.BytesIO(content))
        # пытаемся найти лист "общий"
        sheet = None
        for s in xls.sheet_names:
            if _norm(s) in ("общий", "общее", "overall", "summary"):
                sheet = s
                break
        if sheet is None:
            sheet = xls.sheet_names[0]
        raw = pd.read_excel(xls, sheet_name=sheet)
    else:
        sep = detect_delimiter(content)
        raw = pd.read_csv(io.BytesIO(content), sep=sep, encoding="utf-8-sig")

    # маппинг колонок
    c_week = pick_col(raw, DB_COLS["week"])
    c_pid = pick_col(raw, DB_COLS["player_id"])
    if c_pid is None:
        raise ValueError("В DB не найдена колонка 'ID игрока'.")

    out = pd.DataFrame()
    out["_week"] = pd.to_numeric(raw[c_week], errors="coerce").fillna(-1).astype(int) if c_week else -1
    out["_player_id"] = pd.to_numeric(raw[c_pid], errors="coerce")
    out = out.dropna(subset=["_player_id"]).copy()
    out["_player_id"] = out["_player_id"].astype(int)

    def put_text(key, variants):
        c = pick_col(raw, variants)
        out[key] = raw.loc[out.index, c].astype(str).fillna("") if c else ""

    put_text("_country", DB_COLS["country"])
    put_text("_nick", DB_COLS["nick"])
    put_text("_game_name", DB_COLS["game_name"])
    put_text("_agent", DB_COLS["agent"])
    put_text("_super_agent", DB_COLS["super_agent"])

    def put_num(key, variants):
        c = pick_col(raw, variants)
        out[key] = to_float_series(raw.loc[out.index, c]) if c else np.nan

    put_num("_agent_id", DB_COLS["agent_id"])
    put_num("_super_agent_id", DB_COLS["super_agent_id"])

    put_num("_j_total", DB_COLS["j_total"])
    put_num("_player_win_total", DB_COLS["player_win_total"])
    put_num("_player_win_ring", DB_COLS["player_win_ring"])
    put_num("_player_win_mtt", DB_COLS["player_win_mtt"])
    put_num("_club_comm_total", DB_COLS["club_comm_total"])
    put_num("_club_comm_ppsr", DB_COLS["club_comm_ppsr"])
    put_num("_club_comm_ppst", DB_COLS["club_comm_ppst"])

    put_num("_leader_ppsr", DB_COLS["leader_ppsr"])
    put_num("_leaders_rg", DB_COLS["leaders_rg"])

    return out


def build_db_views(db: pd.DataFrame, player_id: int, weeks_mode: str, last_n: int, week_from: int, week_to: int):
    d = db[db["_player_id"] == int(player_id)].copy()
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

    by_week = (
        d.groupby("_week", as_index=False)[
            ["_j_total", "_player_win_total", "_player_win_ring", "_player_win_mtt",
             "_club_comm_total", "_club_comm_ppsr", "_club_comm_ppst", "_leader_ppsr", "_leaders_rg"]
        ].sum(min_count=1)
        .sort_values("_week")
    )

    agg = by_week.sum(numeric_only=True)

    total_j = float(agg.get("_j_total", 0.0) or 0.0)
    ring_win = float(agg.get("_player_win_ring", 0.0) or 0.0)
    mtt_win = float(agg.get("_player_win_mtt", 0.0) or 0.0)
    pure_win = float(agg.get("_player_win_total", 0.0) or 0.0)

    comm_total = float(agg.get("_club_comm_total", 0.0) or 0.0)
    comm_ppsr = float(agg.get("_club_comm_ppsr", 0.0) or 0.0)  # критично для кэша
    comm_ppst = float(agg.get("_club_comm_ppst", 0.0) or 0.0)

    ring_share = safe_div(ring_win, total_j) if total_j > 0 else np.nan
    profit_to_ppsr_rake = safe_div(ring_win, comm_ppsr) if comm_ppsr > 0 else np.nan

    events_delta = total_j - pure_win

    if by_week.empty:
        top_week = None
        top_week_share = np.nan
    else:
        top_row = by_week.sort_values("_j_total", ascending=False).iloc[0]
        top_week = int(top_row["_week"])
        top_week_share = safe_div(float(top_row["_j_total"] or 0.0), total_j) if total_j > 0 else np.nan

    # профиль игрока (текст)
    prof = d.iloc[-1]
    profile = {
        "nick": str(prof.get("_nick", "")),
        "game_name": str(prof.get("_game_name", "")),
        "country": str(prof.get("_country", "")),
        "agent": str(prof.get("_agent", "")),
        "agent_id": int(prof["_agent_id"]) if pd.notna(prof.get("_agent_id")) else None,
        "super_agent": str(prof.get("_super_agent", "")),
        "super_agent_id": int(prof["_super_agent_id"]) if pd.notna(prof.get("_super_agent_id")) else None,
    }

    summary = {
        "weeks_count": int(len(by_week)),
        "week_min": int(by_week["_week"].min()) if len(by_week) else None,
        "week_max": int(by_week["_week"].max()) if len(by_week) else None,
        "total_j": total_j,
        "pure_win": pure_win,
        "events_delta": float(events_delta),
        "ring_win": ring_win,
        "mtt_win": mtt_win,
        "comm_total": comm_total,
        "comm_ppsr": comm_ppsr,
        "comm_ppst": comm_ppst,
        "ring_share": ring_share,
        "profit_to_ppsr_rake": profit_to_ppsr_rake,
        "top_week": top_week,
        "top_week_share": top_week_share,
        "leader_ppsr_sum": float(agg.get("_leader_ppsr", 0.0) or 0.0),
        "leaders_rg_sum": float(agg.get("_leaders_rg", 0.0) or 0.0),
        "profile": profile,
    }
    return summary, by_week


# =========================
# Games parser (PPPoker text export)
# =========================
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)
START_RE = re.compile(r"Начало:\s*([0-9/:\s]+)", re.IGNORECASE)
END_RE = re.compile(r"Окончание:\s*([0-9/:\s]+)", re.IGNORECASE)
TABLE_RE = re.compile(r"Название стола:\s*(.+)$", re.IGNORECASE)

TYPE_RING_RE = re.compile(r"\bPPSR\b", re.IGNORECASE)
TYPE_TOUR_RE = re.compile(r"\bPPST\b", re.IGNORECASE)


def _split(line: str) -> list[str]:
    # В этих выгрузках чаще ';'
    if ";" in line:
        return [c.strip().strip('"') for c in line.split(";")]
    return [c.strip().strip('"') for c in line.split(",")]


def _forward_fill_group(a: list[str]) -> list[str]:
    out = []
    last = ""
    for x in a:
        if x != "":
            last = x
        out.append(last)
    return out


def parse_games_text(file_obj) -> pd.DataFrame:
    """
    Возвращает строки уровня player-in-game:
    game_id, game_type(RING/TOURNAMENT/UNKNOWN), players_n, hands, win_total, win_vs_opponents,
    club_commission, buyin_ppchips, bounty, table_name, start_ts, end_ts
    """
    text = file_obj.getvalue().decode("utf-8", errors="ignore")
    lines = text.splitlines()

    rows = []
    cur_game_id = None
    cur_table = ""
    cur_start = ""
    cur_end = ""
    cur_type = "UNKNOWN"

    header1 = None
    header2 = None
    idx_map = None

    def flush_headers():
        nonlocal header1, header2, idx_map
        header1 = None
        header2 = None
        idx_map = None

    for i, line in enumerate(lines):
        # начало/окончание
        if "Начало:" in line:
            m1 = START_RE.search(line)
            m2 = END_RE.search(line)
            if m1:
                cur_start = m1.group(1).strip()
            if m2:
                cur_end = m2.group(1).strip()

        m = GAME_ID_RE.search(line)
        if m:
            cur_game_id = m.group(1).strip()
            cur_table = ""
            cur_type = "UNKNOWN"
            flush_headers()
            continue

        if cur_game_id is None:
            continue

        if "Название стола:" in line:
            mt = TABLE_RE.search(line)
            if mt:
                cur_table = mt.group(1).strip()
            continue

        # строка типа игры: PPSR/... или PPST/...
        if "PPSR" in line or "PPST" in line:
            if TYPE_RING_RE.search(line):
                cur_type = "RING"
            elif TYPE_TOUR_RE.search(line):
                cur_type = "TOURNAMENT"
            continue

        # header line
        if ";ID игрока" in line or "ID игрока" in line and "Ник" in line:
            header1 = _split(line)
            header1 = [h for h in header1 if h != ""]  # иногда слева пусто
            # но из-за лидирующего ';' в строках данные обычно имеют пустую первую ячейку
            # поэтому сохраняем “сырой” header по split без удаления:
            header1 = _split(line)
            # попробуем прочитать следующий line как subheader (для Ring)
            header2 = None
            idx_map = None
            continue

        # subheader line for Ring
        if header1 is not None and header2 is None and ("От соперников" in line or "Доход клуба" in line) and ("Общий" in line):
            header2 = _split(line)
            # сделаем комбинированные имена колонок
            h1 = _forward_fill_group(header1)
            h2 = header2
            cols = []
            for a, b in zip(h1, h2):
                a = a.strip()
                b = b.strip()
                if a == "":
                    a = "COL"
                if b != "":
                    cols.append(f"{a}|{b}")
                else:
                    cols.append(a)
            idx_map = {c: j for j, c in enumerate(cols)}
            continue

        # data lines
        if header1 is None:
            continue

        parts = _split(line)
        if len(parts) < 3:
            continue
        if "Итог" in line:
            continue

        # если idx_map не собран — значит таблица простая (турнирная)
        if idx_map is None:
            cols = header1
            idx_map = {c: j for j, c in enumerate(cols)}

        def get(variants: list[str]):
            for v in variants:
                if v in idx_map and idx_map[v] < len(parts):
                    return parts[idx_map[v]]
            return None

        # player_id
        pid_raw = get(["ID игрока"])
        try:
            pid = int(float(str(pid_raw).replace(",", ".")))
        except Exception:
            continue

        # hands
        hands = to_float(get(["Раздачи"]))  # для ring
        # buyin
        buyin = to_float(get(["Бай-ин с PP-фишками", "Бай-ин с PP-фишками "]))
        # tournament bounty
        bounty = to_float(get(["От баунти"]))

        # win_total: разные варианты
        win_total = to_float(get(["Выигрыш", "Выигрыш игрока", "Выигрыш игрока|Общий", "Выигрыш игрока|Общий "]))
        # win_vs_opponents (самый важный для переливов в кэше)
        win_vs_opp = to_float(get(["Выигрыш игрока|От соперников", "Выигрыш игрока|От соперников "]))

        # club commission (иногда отдельно в таблице ring)
        club_comm = to_float(get(["Доход клуба|Комиссия", "Комиссия"]))

        rows.append({
            "game_id": cur_game_id,
            "game_type": cur_type,
            "player_id": pid,
            "table_name": cur_table,
            "start_ts": cur_start,
            "end_ts": cur_end,
            "hands": hands,
            "buyin_ppchips": buyin,
            "bounty": bounty,
            "win_total": win_total,
            "win_vs_opponents": win_vs_opp,
            "club_commission": club_comm,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "game_id", "game_type", "player_id", "table_name", "start_ts", "end_ts",
            "hands", "buyin_ppchips", "bounty", "win_total", "win_vs_opponents", "club_commission"
        ])

    # players_n на игру
    pn = df.groupby(["game_id", "game_type"])["player_id"].nunique().reset_index().rename(columns={"player_id": "players_n"})
    df = df.merge(pn, on=["game_id", "game_type"], how="left")

    # нормализация win_base: для ring берём vs_opponents если есть, иначе win_total
    df["win_base"] = np.where(
        (df["game_type"] == "RING") & pd.notna(df["win_vs_opponents"]),
        df["win_vs_opponents"],
        df["win_total"],
    )
    df["win_base"] = pd.to_numeric(df["win_base"], errors="coerce")

    return df


# =========================
# Pair flows (net-flow)
# =========================
def build_pair_flows(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Строим потоки from_player -> to_player по win_base:
    проигрыш распределяется по победителям пропорционально их win_base.
    """
    if games_df.empty:
        return pd.DataFrame(columns=["from_player", "to_player", "game_type", "amount", "games_cnt"])

    gdf = games_df.copy()
    gdf = gdf[pd.notna(gdf["win_base"])].copy()
    gdf["win_base"] = pd.to_numeric(gdf["win_base"], errors="coerce")
    gdf = gdf.dropna(subset=["win_base"]).copy()

    flows = {}
    games_cnt = {}

    for (gid, gtype), part in gdf.groupby(["game_id", "game_type"]):
        part = part[["player_id", "win_base"]].dropna().copy()
        if part["player_id"].nunique() < 2:
            continue
        winners = part[part["win_base"] > 0].copy()
        losers = part[part["win_base"] < 0].copy()
        if winners.empty or losers.empty:
            continue
        total_pos = float(winners["win_base"].sum())
        if total_pos <= 0:
            continue

        for _, lr in losers.iterrows():
            lpid = int(lr["player_id"])
            loss = float(-lr["win_base"])
            if loss <= 0:
                continue
            for _, wr in winners.iterrows():
                wpid = int(wr["player_id"])
                wwin = float(wr["win_base"])
                if wwin <= 0:
                    continue
                amt = loss * (wwin / total_pos)
                key = (lpid, wpid, gtype)
                flows[key] = flows.get(key, 0.0) + amt
                games_cnt[key] = games_cnt.get(key, 0) + 1

    if not flows:
        return pd.DataFrame(columns=["from_player", "to_player", "game_type", "amount", "games_cnt"])

    out = pd.DataFrame(
        [{"from_player": k[0], "to_player": k[1], "game_type": k[2], "amount": float(v), "games_cnt": int(games_cnt[k])}
         for k, v in flows.items()]
    )
    out = out.groupby(["from_player", "to_player", "game_type"], as_index=False).agg({"amount": "sum", "games_cnt": "sum"})
    return out


def pair_features(target_id: int, games_df: pd.DataFrame, flows_df: pd.DataFrame, game_type: str | None):
    df = games_df.copy()
    if game_type:
        df = df[df["game_type"] == game_type]

    # coverage
    tg = df[df["player_id"] == target_id].copy()
    games_cnt = int(len(tg))

    # dump sessions (много проиграл за мало рук)
    dump_hits = []
    if game_type == "RING" and not tg.empty:
        for _, r in tg.iterrows():
            hands = r.get("hands", np.nan)
            wb = r.get("win_base", np.nan)
            if pd.isna(wb):
                continue
            if pd.notna(hands) and float(hands) <= DUMP_HANDS_MAX and float(wb) <= -DUMP_LOSS_ALERT:
                dump_hits.append({
                    "game_id": r["game_id"],
                    "hands": float(hands) if pd.notna(hands) else None,
                    "loss": float(wb),
                    "table": r.get("table_name", ""),
                })

    f = flows_df.copy()
    if game_type:
        f = f[f["game_type"] == game_type]

    inflow = f[f["to_player"] == target_id].groupby("from_player")["amount"].sum().to_dict()
    outflow = f[f["from_player"] == target_id].groupby("to_player")["amount"].sum().to_dict()

    partners = set(inflow.keys()) | set(outflow.keys())
    net = {p: float(inflow.get(p, 0.0) - outflow.get(p, 0.0)) for p in partners}

    top_net_partner = None
    top_net = 0.0
    if net:
        top_net_partner, top_net = sorted(net.items(), key=lambda x: abs(x[1]), reverse=True)[0]

    gross_in = float(sum(inflow.values()))
    gross_out = float(sum(outflow.values()))
    gross = gross_in + gross_out
    one_sidedness = float(abs(gross_in - gross_out) / gross) if gross > 0 else 0.0

    if top_net_partner is not None and gross > 0:
        top_pair_gross = float(inflow.get(top_net_partner, 0.0) + outflow.get(top_net_partner, 0.0))
        top_pair_share = float(top_pair_gross / gross)
    else:
        top_pair_share = 0.0

    return {
        "games_cnt": games_cnt,
        "gross_in": gross_in,
        "gross_out": gross_out,
        "one_sidedness": one_sidedness,
        "top_net_partner": top_net_partner,
        "top_net": float(top_net),
        "top_pair_share": float(top_pair_share),
        "top_inflows": sorted(inflow.items(), key=lambda x: x[1], reverse=True)[:12],
        "top_outflows": sorted(outflow.items(), key=lambda x: x[1], reverse=True)[:12],
        "dump_hits": dump_hits[:12],
    }


# =========================
# Scoring
# =========================
def score_player(db_sum: dict, ring_pf: dict, tour_pf: dict):
    score = 0
    reasons = []

    total_j = db_sum["total_j"]
    ring_win = db_sum["ring_win"]
    comm_ppsr = db_sum["comm_ppsr"]
    profit_to_ppsr_rake = db_sum["profit_to_ppsr_rake"]
    top_week_share = db_sum["top_week_share"]

    # 1) DB: плюсовой игрок — проверять
    if total_j > 0:
        score += 18
        reasons.append("DB: игрок в плюсе по итоговому результату — базовая проверка обязательна.")
    else:
        score += 5
        reasons.append("DB: игрок в минусе по итогу — как 'получатель' менее вероятен (но не исключено).")

    # 2) DB: кэш + высокая эффективность к рейку
    if ring_win > 0 and comm_ppsr > 0 and pd.notna(profit_to_ppsr_rake):
        if profit_to_ppsr_rake >= 8 and ring_win >= 80:
            score += 12
            reasons.append("DB: очень высокий профит в PPSR относительно PPSR-комиссии.")
        elif profit_to_ppsr_rake >= 5 and ring_win >= 50:
            score += 7
            reasons.append("DB: высокий профит в PPSR относительно PPSR-комиссии.")

    # 3) DB: концентрация по неделе
    if total_j > 0 and pd.notna(top_week_share) and top_week_share >= 0.60 and db_sum["weeks_count"] >= 2:
        score += 8
        reasons.append("DB: результат сильно сконцентрирован в одной неделе.")

    # 4) DB: лидерборды (как доп. триггер)
    if (db_sum.get("leader_ppsr_sum", 0.0) or 0.0) > 0:
        score += 4
        reasons.append("DB: есть активность в PPSR лидерборде (нужна внимательнее проверка источника профита).")

    # 5) Games: net-flow по RING
    if ring_pf["games_cnt"] == 0:
        score += 10
        reasons.append("GAMES/RING: нет покрывающих игр — перелив по кэшу не проверить.")
    else:
        net_abs = abs(ring_pf["top_net"])
        if ring_pf["top_net_partner"] is not None and net_abs >= PAIR_NET_ALERT_RING:
            score += 28
            direction = "в пользу игрока" if ring_pf["top_net"] > 0 else "в пользу партнёра"
            reasons.append(f"GAMES/RING: крупный net-flow с одним ID ({ring_pf['top_net_partner']}), {direction}, net≈{fmt_money(ring_pf['top_net'])}.")
        if ring_pf["top_pair_share"] >= PAIR_DOMINANCE_ALERT and net_abs >= (PAIR_NET_ALERT_RING / 2):
            score += 12
            reasons.append(f"GAMES/RING: доминирование одной пары по обороту (доля≈{ring_pf['top_pair_share']:.0%}).")
        if ring_pf["one_sidedness"] >= PAIR_ONE_SIDED_ALERT and net_abs >= (PAIR_NET_ALERT_RING / 2):
            score += 8
            reasons.append(f"GAMES/RING: поток односторонний (one_sided≈{ring_pf['one_sidedness']:.0%}).")
        if ring_pf["dump_hits"]:
            score += 12
            reasons.append("GAMES/RING: найдено 'слил много за мало раздач' (дампинг-паттерн).")

    # 6) Games: турниры — слабее как сигнал, но учитываем
    if tour_pf["games_cnt"] > 0:
        net_abs = abs(tour_pf["top_net"])
        if tour_pf["top_net_partner"] is not None and net_abs >= PAIR_NET_ALERT_TOUR:
            score += 8
            reasons.append(f"GAMES/TOUR: заметный net-flow по турнирам с ID {tour_pf['top_net_partner']}, net≈{fmt_money(tour_pf['top_net'])}.")

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)

    if decision == "APPROVE":
        main_risk = "Явных признаков перелива по доступным агрегатам не выявлено."
    elif decision == "FAST_CHECK":
        main_risk = "Есть признаки риска: рекомендуется быстрая проверка перед выводом."
    else:
        main_risk = "Высокий риск перелива/сговора: нужна ручная проверка СБ."

    return score, decision, main_risk, reasons


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Загрузка")

    st.subheader("1) DB (XLSX/CSV)")
    st.caption("Рекомендуется DB.xlsx с листом 'общий'.")
    db_up = st.file_uploader("DB.xlsx / DB.csv", type=["xlsx", "xls", "csv"], key="db_up")

    st.subheader("2) Games (текстовый экспорт PPPoker)")
    st.caption("Подходит файл вида 'Игры данные.csv' (по факту это текст с блоками).")
    games_up = st.file_uploader("Games export", type=["csv", "txt"], key="games_up")

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
    st.subheader("Период (по неделям)")
    weeks_mode = st.selectbox("Фильтр недель", ["Все недели", "Последние N недель", "Диапазон недель"])
    last_n = st.number_input("N (если 'Последние N недель')", min_value=1, value=4, step=1)

if db_file is None:
    st.info("Загрузи DB.xlsx/DB.csv.")
    st.stop()

try:
    db_df = load_db(db_file)
except Exception as e:
    st.error("Не удалось прочитать DB. Проверь файл/лист/колонки.")
    st.exception(e)
    st.stop()

games_df = pd.DataFrame()
flows_df = pd.DataFrame()
if games_file is not None:
    try:
        games_df = parse_games_text(games_file)
        flows_df = build_pair_flows(games_df)
    except Exception as e:
        st.warning("Games загружен, но не удалось корректно разобрать. Аналитика по играм будет ограничена.")
        st.exception(e)
        games_df = pd.DataFrame()
        flows_df = pd.DataFrame()

valid_weeks = sorted([w for w in db_df["_week"].unique().tolist() if w >= 0])
w_min = min(valid_weeks) if valid_weeks else 0
w_max = max(valid_weeks) if valid_weeks else 0

m1, m2, m3, m4 = st.columns(4, gap="small")
m1.metric("DB строк", f"{len(db_df)}", border=True)
m2.metric("DB игроков", f"{db_df['_player_id'].nunique()}", border=True)
m3.metric("Games строк (player-game)", f"{len(games_df) if not games_df.empty else 0}", border=True)
m4.metric("Pair flows", f"{len(flows_df) if not flows_df.empty else 0}", border=True)

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
        st.subheader("Логика решения")
        st.markdown(
            "- DB: общий плюс, PPSR (кэш) vs комиссия PPSR.\n"
            "- Games/RING: net-flow по 'От соперников' (если доступно), односторонность, доминирование пары.\n"
            "- Games/RING: 'слил много за мало раздач' (дампинг)."
        )

    if not run:
        st.stop()

    db_sum, by_week = build_db_views(db_df, int(player_id), weeks_mode, int(last_n), int(week_from), int(week_to))
    if db_sum is None:
        st.error("Игрок не найден в DB по выбранному периоду.")
        st.stop()

    ring_pf = pair_features(int(player_id), games_df, flows_df, "RING") if not games_df.empty else pair_features(int(player_id), pd.DataFrame(), pd.DataFrame(), "RING")
    tour_pf = pair_features(int(player_id), games_df, flows_df, "TOURNAMENT") if not games_df.empty else pair_features(int(player_id), pd.DataFrame(), pd.DataFrame(), "TOURNAMENT")

    score, decision, main_risk, reasons = score_player(db_sum, ring_pf, tour_pf)

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

    tabs = st.tabs(["Кратко", "DB", "Games/RING", "Пары/дампинг"])
    with tabs[0]:
        st.subheader("Главный риск")
        st.info(main_risk)
        st.subheader("Причины")
        for r in reasons[:16]:
            st.markdown(f"- {r}")

        prof = db_sum["profile"]
        st.subheader("Профиль")
        st.markdown(
            f"- Ник: {prof.get('nick','')}\n"
            f"- Игровое имя: {prof.get('game_name','')}\n"
            f"- Страна: {prof.get('country','')}\n"
            f"- Агент: {prof.get('agent','')} (ID: {prof.get('agent_id')})\n"
            f"- Супер-агент: {prof.get('super_agent','')} (ID: {prof.get('super_agent_id')})"
        )

    with tabs[1]:
        st.subheader("DB агрегаты")
        x1, x2, x3, x4 = st.columns(4, gap="small")
        x1.metric("J итог (+события)", fmt_money(db_sum["total_j"]), border=True)
        x2.metric("Win общий", fmt_money(db_sum["pure_win"]), border=True)
        x3.metric("Ring win", fmt_money(db_sum["ring_win"]), border=True)
        x4.metric("MTT win", fmt_money(db_sum["mtt_win"]), border=True)

        y1, y2, y3, y4 = st.columns(4, gap="small")
        y1.metric("Комиссия PPSR", fmt_money(db_sum["comm_ppsr"]), border=True)
        y2.metric("Комиссия PPST", fmt_money(db_sum["comm_ppst"]), border=True)
        y3.metric("Ring доля", "NaN" if pd.isna(db_sum["ring_share"]) else f"{db_sum['ring_share']:.0%}", border=True)
        y4.metric("Ring/Комиссия PPSR", "NaN" if pd.isna(db_sum["profit_to_ppsr_rake"]) else f"{db_sum['profit_to_ppsr_rake']:.1f}x", border=True)

        st.subheader("По неделям")
        show = by_week.rename(columns={
            "_week": "Неделя",
            "_j_total": "J итог (+события)",
            "_player_win_total": "Win общий",
            "_player_win_ring": "Ring win",
            "_player_win_mtt": "MTT win",
            "_club_comm_ppsr": "Комиссия PPSR",
            "_club_comm_ppst": "Комиссия PPST",
            "_leader_ppsr": "PPSR лидерборд",
            "_leaders_rg": "Лидеры RG",
        }).copy()
        show["J-общийWin (дельта событий)"] = show["J итог (+события)"] - show["Win общий"]
        st.dataframe(show.sort_values("Неделя", ascending=False), use_container_width=True)

    with tabs[2]:
        st.subheader("Games/RING: потоки")
        st.markdown(
            f"- Игр: {ring_pf['games_cnt']}\n"
            f"- Inflow: {fmt_money(ring_pf['gross_in'])}\n"
            f"- Outflow: {fmt_money(ring_pf['gross_out'])}\n"
            f"- One-sided: {ring_pf['one_sidedness']:.0%}\n"
            f"- Топ партнёр: {ring_pf['top_net_partner']}\n"
            f"- Top net: {fmt_money(ring_pf['top_net'])}\n"
            f"- Доля пары: {ring_pf['top_pair_share']:.0%}"
        )

    with tabs[3]:
        st.subheader("Топ inflow (кто 'кормит')")
        if ring_pf["top_inflows"]:
            st.dataframe(pd.DataFrame(ring_pf["top_inflows"], columns=["from_player", "amount_to_target"]), use_container_width=True)
        else:
            st.info("Нет выраженных источников inflow.")

        st.subheader("Топ outflow (кого 'кормит' игрок)")
        if ring_pf["top_outflows"]:
            st.dataframe(pd.DataFrame(ring_pf["top_outflows"], columns=["to_player", "amount_from_target"]), use_container_width=True)
        else:
            st.info("Нет выраженных направлений outflow.")

        st.subheader("Dumping hits (мало раздач, большой минус)")
        if ring_pf["dump_hits"]:
            st.dataframe(pd.DataFrame(ring_pf["dump_hits"]), use_container_width=True)
        else:
            st.info("Не найдено дампинг-сессий по заданным порогам.")

with tab2:
    st.subheader("Топ подозрительных (быстрая приоритизация)")
    st.caption("Сортировка по risk_score. Это сигнал для СБ, не 'приговор'.")

    colA, colB = st.columns([1, 1])
    with colA:
        top_n = st.number_input("Сколько показать", min_value=5, max_value=200, value=30, step=5)
    with colB:
        build = st.button("Посчитать ТОП", type="primary", use_container_width=True)

    week_from2 = st.number_input("Неделя от (для ТОП)", value=w_min, step=1, key="top_w_from")
    week_to2 = st.number_input("Неделя до (для ТОП)", value=w_max, step=1, key="top_w_to")

    if not build:
        st.stop()

    # пересчёт по игрокам
    res = []
    for pid in sorted(db_df["_player_id"].unique().tolist()):
        db_sum, _ = build_db_views(db_df, int(pid), weeks_mode, int(last_n), int(week_from2), int(week_to2))
        if db_sum is None:
            continue

        ring_pf = pair_features(int(pid), games_df, flows_df, "RING") if not games_df.empty else pair_features(int(pid), pd.DataFrame(), pd.DataFrame(), "RING")
        tour_pf = pair_features(int(pid), games_df, flows_df, "TOURNAMENT") if not games_df.empty else pair_features(int(pid), pd.DataFrame(), pd.DataFrame(), "TOURNAMENT")

        score, decision, _, _ = score_player(db_sum, ring_pf, tour_pf)

        res.append({
            "player_id": int(pid),
            "risk_score": int(score),
            "decision": decision,
            "db_total_j": float(db_sum["total_j"]),
            "db_ring_win": float(db_sum["ring_win"]),
            "db_comm_ppsr": float(db_sum["comm_ppsr"]),
            "ring_games": int(ring_pf["games_cnt"]),
            "ring_top_partner": ring_pf["top_net_partner"] if ring_pf["top_net_partner"] is not None else "",
            "ring_top_net": float(ring_pf["top_net"]),
            "ring_pair_share": float(ring_pf["top_pair_share"]),
            "ring_one_sided": float(ring_pf["one_sidedness"]),
            "dump_hits": len(ring_pf["dump_hits"]),
        })

    top_df = pd.DataFrame(res)
    if top_df.empty:
        st.info("Нет данных для ТОП.")
        st.stop()

    top_df = top_df.sort_values(["risk_score", "db_total_j"], ascending=[False, False]).head(int(top_n)).copy()
    show = top_df.copy()
    show["ring_pair_share"] = show["ring_pair_share"].apply(lambda x: f"{x:.0%}")
    show["ring_one_sided"] = show["ring_one_sided"].apply(lambda x: f"{x:.0%}")
    st.dataframe(show, use_container_width=True)

    st.download_button(
        "Скачать ТОП в CSV",
        data=top_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="top_suspicious.csv",
        mime="text/csv",
    )
