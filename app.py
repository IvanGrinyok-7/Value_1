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
APP_TITLE = "PPPoker Anti-Fraud: общий CSV + игры CSV (fixed)"
CACHE_DIR = Path(".pppoker_app_cache")
CACHE_DIR.mkdir(exist_ok=True)

DB_KEY = "db_csv"
GAMES_KEY = "games_csv"

# Скоринг (пороги оставлены близко к твоим, но логика стала точнее)
T_APPROVE = 25
T_FAST_CHECK = 55

MIN_SESSIONS_FOR_COPLAY = 6
COPLAY_TOP1_SHARE_SUSP = 0.60
COPLAY_TOP2_SHARE_SUSP = 0.80

# Для net-flow (в условных единицах выгрузки)
PAIR_NET_ALERT_RING = 25.0
PAIR_NET_ALERT_TOUR = 60.0
PAIR_DOMINANCE_ALERT = 0.70
PAIR_ONE_SIDED_ALERT = 0.85

SINGLE_GAME_WIN_ALERT_RING = 60.0
SINGLE_GAME_WIN_ALERT_TOUR = 150.0

# PPPoker games parsing (fallback text export)
TOUR_HINT_RE = re.compile(r"\bPPST/|бай-ин:\s*|satellite|pko|mko\b", re.IGNORECASE)
RING_HINT_RE = re.compile(r"\bPPSR/|Ring|NLH|PLO|Bomb Pot|Ante\b", re.IGNORECASE)
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)

# Ожидаемые колонки общего CSV (как у твоего кода)
COL_WEEK = "Номер недели"
COL_PLAYER_ID = "ID игрока"
COL_J_TOTAL = "Общий выигрыш игроков + События"
COL_PLAYER_WIN_TOTAL = "Выигрыш игрока Общий"
COL_PLAYER_WIN_RING = "Выигрыш игрока Ring Game"
COL_PLAYER_WIN_MTT = "Выигрыш игрока MTT, SNG"
COL_CLUB_INCOME_TOTAL = "Доход клуба Общий"
COL_CLUB_COMMISSION = "Доход клуба Комиссия"


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


# =========================
# Load DB (общий CSV)
# =========================
def load_db_csv(file_obj) -> pd.DataFrame:
    content = file_obj.getvalue()
    sep = detect_delimiter(content)

    df = pd.read_csv(io.BytesIO(content), sep=sep, encoding="utf-8-sig")

    required = [
        COL_WEEK, COL_PLAYER_ID, COL_J_TOTAL,
        COL_PLAYER_WIN_TOTAL, COL_PLAYER_WIN_RING, COL_PLAYER_WIN_MTT,
        COL_CLUB_INCOME_TOTAL, COL_CLUB_COMMISSION
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"В общем CSV не найдены колонки: {missing}. Проверь экспорт/разделитель.")

    out = pd.DataFrame()
    out["_week"] = pd.to_numeric(df[COL_WEEK], errors="coerce").fillna(-1).astype(int)
    out["_player_id"] = pd.to_numeric(df[COL_PLAYER_ID], errors="coerce")
    out = out.dropna(subset=["_player_id"]).copy()
    out["_player_id"] = out["_player_id"].astype(int)

    out["_total_win_j"] = to_float_series(df.loc[out.index, COL_J_TOTAL])
    out["_player_win_total"] = to_float_series(df.loc[out.index, COL_PLAYER_WIN_TOTAL])
    out["_player_win_ring"] = to_float_series(df.loc[out.index, COL_PLAYER_WIN_RING])
    out["_player_win_mtt"] = to_float_series(df.loc[out.index, COL_PLAYER_WIN_MTT])
    out["_club_income_total"] = to_float_series(df.loc[out.index, COL_CLUB_INCOME_TOTAL])
    out["_club_commission"] = to_float_series(df.loc[out.index, COL_CLUB_COMMISSION])

    return out[
        [
            "_week",
            "_player_id",
            "_total_win_j",
            "_player_win_total",
            "_player_win_ring",
            "_player_win_mtt",
            "_club_income_total",
            "_club_commission",
        ]
    ].copy()


# =========================
# Games parsing
# =========================
def classify_game_type_text(block_lines: list[str]) -> str:
    text = " ".join(block_lines)
    if TOUR_HINT_RE.search(text):
        return "TOURNAMENT"
    if RING_HINT_RE.search(text):
        return "RING"
    return "UNKNOWN"


def _normalize_col(df: pd.DataFrame, variants: list[str]) -> str | None:
    cols = {c.strip().lower(): c for c in df.columns}
    for v in variants:
        key = v.strip().lower()
        if key in cols:
            return cols[key]
    return None


def parse_games_csv(file_obj) -> pd.DataFrame:
    """
    FIX: сначала пытаемся разобрать как обычный табличный CSV (твой текущий парсер
    рассчитан на текстовый экспорт с блоками и часто возвращает пустоту).
    Если таблично не выходит — fallback на старый text-парсер по "ID игры:".
    """
    content = file_obj.getvalue()
    sep = detect_delimiter(content)

    # 1) TABULAR попытка
    try:
        df = pd.read_csv(io.BytesIO(content), sep=sep, encoding="utf-8-sig")
        # минимальные требования
        c_pid = _normalize_col(df, ["ID игрока", "player id", "player_id"])
        c_win = _normalize_col(df, ["Выигрыш", "win", "net win", "result"])
        c_fee = _normalize_col(df, ["Комиссия", "fee", "rake", "commission"])
        c_gid = _normalize_col(df, ["ID игры", "game id", "game_id", "ID Game"])
        c_gname = _normalize_col(df, ["Название игры", "game", "тип игры", "game type", "room"])

        if c_pid and c_win and c_gid:
            out = pd.DataFrame()
            out["player_id"] = pd.to_numeric(df[c_pid], errors="coerce")
            out = out.dropna(subset=["player_id"]).copy()
            out["player_id"] = out["player_id"].astype(int)

            out["game_id"] = df.loc[out.index, c_gid].astype(str).str.strip()
            out["win"] = to_float_series(df.loc[out.index, c_win])
            out["fee"] = to_float_series(df.loc[out.index, c_fee]) if c_fee else np.nan

            # game_type классификация
            if c_gname:
                gtxt = df.loc[out.index, c_gname].astype(str)
                out["game_type"] = np.where(
                    gtxt.str.contains(TOUR_HINT_RE),
                    "TOURNAMENT",
                    np.where(gtxt.str.contains(RING_HINT_RE), "RING", "UNKNOWN"),
                )
            else:
                out["game_type"] = "UNKNOWN"

            # очищаем мусор
            out = out[["game_id", "game_type", "player_id", "win", "fee"]].copy()
            out = out[pd.notna(out["win"])].copy()
            if not out.empty:
                return out
    except Exception:
        pass

    # 2) FALLBACK: TEXT parser (твой старый подход, но чуть подчистим)
    text = content.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    rows = []
    current_game_id = None
    block_lines = []
    header = None
    win_idx = fee_idx = pid_idx = None

    def split_line(line: str):
        if ";" in line:
            return [c.strip().strip('"') for c in line.split(";")]
        return [c.strip().strip('"') for c in line.split(",")]

    for line in lines:
        m = GAME_ID_RE.search(line)
        if m:
            current_game_id = m.group(1).strip()
            block_lines = [line]
            header = None
            win_idx = fee_idx = pid_idx = None
            continue

        if not current_game_id:
            continue

        block_lines.append(line)

        if "ID игрока" in line:
            header = split_line(line)
            pid_idx = header.index("ID игрока") if "ID игрока" in header else 0
            win_idx = header.index("Выигрыш") if "Выигрыш" in header else None
            fee_idx = header.index("Комиссия") if "Комиссия" in header else None
            continue

        if header is None:
            continue

        parts = split_line(line)
        if len(parts) < 2:
            continue

        try:
            pid_raw = parts[pid_idx] if pid_idx is not None and pid_idx < len(parts) else parts[0]
            pid = int(float(str(pid_raw).replace(",", ".")))
        except Exception:
            continue

        win = np.nan
        fee = np.nan
        if win_idx is not None and win_idx < len(parts):
            win = to_float(parts[win_idx])
        if fee_idx is not None and fee_idx < len(parts):
            fee = to_float(parts[fee_idx])

        gtype = classify_game_type_text(block_lines)
        rows.append({"game_id": current_game_id, "game_type": gtype, "player_id": pid, "win": win, "fee": fee})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["game_id", "game_type", "player_id", "win", "fee"])
    df = df[pd.notna(df["win"])].copy()
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
# Net-flow mechanics (FIX вместо min(win, loss) по всем лузерам)
# =========================
def build_pair_flows(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает таблицу потоков:
    from_player -> to_player -> amount (сумма по всем game_id)
    """
    if games_df.empty:
        return pd.DataFrame(columns=["from_player", "to_player", "amount", "game_type", "games_cnt"])

    # чистим
    gdf = games_df.copy()
    gdf = gdf[pd.notna(gdf["win"])].copy()
    gdf["win"] = pd.to_numeric(gdf["win"], errors="coerce")
    gdf = gdf.dropna(subset=["win"]).copy()

    flows = {}
    games_cnt = {}

    for (gid, gtype), part in gdf.groupby(["game_id", "game_type"]):
        part = part[["player_id", "win"]].dropna().copy()
        if part["player_id"].nunique() < 2:
            continue

        winners = part[part["win"] > 0].copy()
        losers = part[part["win"] < 0].copy()
        if winners.empty or losers.empty:
            continue

        total_pos = float(winners["win"].sum())
        if total_pos <= 0:
            continue

        # Распределяем каждый проигрыш лузера по победителям пропорционально их выигрышу.
        for _, lr in losers.iterrows():
            lpid = int(lr["player_id"])
            loss = float(-lr["win"])
            if loss <= 0:
                continue

            for _, wr in winners.iterrows():
                wpid = int(wr["player_id"])
                wwin = float(wr["win"])
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

    # Аггрегируем (на случай повторов ключей)
    out = (
        out.groupby(["from_player", "to_player", "game_type"], as_index=False)
        .agg({"amount": "sum", "games_cnt": "sum"})
    )
    return out


def transfer_features_v2(target_id: int, games_df: pd.DataFrame, flows_df: pd.DataFrame, game_type: str | None = None) -> dict:
    if games_df.empty:
        return {
            "target_games": 0, "target_total_win": 0.0,
            "top_inflows": [], "top_outflows": [],
            "top_net_partner": None, "top_net": 0.0,
            "top_net_share": 0.0, "one_sidedness": 0.0,
            "extremes": []
        }

    df = games_df.copy()
    if game_type:
        df = df[df["game_type"] == game_type]

    df = df[pd.notna(df["win"])].copy()
    t = df[df["player_id"] == target_id][["game_id", "win"]].copy()

    # extremes: большие выигрыши в одной игре
    extremes = []
    win_alert = SINGLE_GAME_WIN_ALERT_TOUR if game_type == "TOURNAMENT" else SINGLE_GAME_WIN_ALERT_RING
    for _, r in t.iterrows():
        if pd.notna(r["win"]) and float(r["win"]) >= win_alert:
            extremes.append({"game_id": r["game_id"], "target_win": float(r["win"])})

    if t.empty:
        return {
            "target_games": 0, "target_total_win": 0.0,
            "top_inflows": [], "top_outflows": [],
            "top_net_partner": None, "top_net": 0.0,
            "top_net_share": 0.0, "one_sidedness": 0.0,
            "extremes": extremes[:12]
        }

    f = flows_df
    if game_type:
        f = f[f["game_type"] == game_type]

    inflow = f[f["to_player"] == target_id].groupby("from_player")["amount"].sum().to_dict()
    outflow = f[f["from_player"] == target_id].groupby("to_player")["amount"].sum().to_dict()

    # net по каждому партнёру
    partners = set(inflow.keys()) | set(outflow.keys())
    net = {}
    for p in partners:
        net[p] = float(inflow.get(p, 0.0) - outflow.get(p, 0.0))

    # сортировки
    top_inflows = sorted(inflow.items(), key=lambda x: x[1], reverse=True)[:12]
    top_outflows = sorted(outflow.items(), key=lambda x: x[1], reverse=True)[:12]
    net_sorted = sorted(net.items(), key=lambda x: abs(x[1]), reverse=True)

    gross_in = float(sum(inflow.values()))
    gross_out = float(sum(outflow.values()))
    gross = gross_in + gross_out
    one_sidedness = float(abs(gross_in - gross_out) / gross) if gross > 0 else 0.0

    if net_sorted:
        top_partner, top_net = net_sorted[0]
        gross_with_top = float(inflow.get(top_partner, 0.0) + outflow.get(top_partner, 0.0))
        top_net_share = float(gross_with_top / gross) if gross > 0 else 0.0
    else:
        top_partner, top_net, top_net_share = None, 0.0, 0.0

    return {
        "target_games": int(len(t)),
        "target_total_win": float(t["win"].sum()),
        "top_inflows": top_inflows,
        "top_outflows": top_outflows,
        "top_net_partner": top_partner,
        "top_net": float(top_net),
        "top_net_share": float(top_net_share),
        "one_sidedness": float(one_sidedness),
        "extremes": extremes[:12],
    }


# =========================
# DB summaries + scoring
# =========================
def build_db_views(df: pd.DataFrame, player_id: int, weeks_mode: str, last_n: int, week_from: int, week_to: int):
    d = df[df["_player_id"] == int(player_id)].copy()
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
            [
                "_total_win_j",
                "_player_win_total",
                "_player_win_ring",
                "_player_win_mtt",
                "_club_income_total",
                "_club_commission",
            ]
        ]
        .sum(min_count=1)
        .sort_values("_week")
    )

    agg = by_week[
        [
            "_total_win_j",
            "_player_win_total",
            "_player_win_ring",
            "_player_win_mtt",
            "_club_income_total",
            "_club_commission",
        ]
    ].sum(numeric_only=True)

    total_win = float(agg.get("_total_win_j", 0.0) or 0.0)
    ring_win = float(agg.get("_player_win_ring", 0.0) or 0.0)
    mtt_win = float(agg.get("_player_win_mtt", 0.0) or 0.0)
    comm = float(agg.get("_club_commission", 0.0) or 0.0)
    pure_player_win = float(agg.get("_player_win_total", 0.0) or 0.0)

    ring_share = safe_div(ring_win, total_win) if total_win > 0 else np.nan
    mtt_share = safe_div(mtt_win, total_win) if total_win > 0 else np.nan
    profit_to_rake = safe_div(total_win, comm) if comm > 0 else np.nan
    events_delta = total_win - pure_player_win

    if by_week.empty:
        top_week = None
        top_week_win = 0.0
        top_week_share = np.nan
    else:
        top_row = by_week.sort_values("_total_win_j", ascending=False).iloc[0]
        top_week = int(top_row["_week"])
        top_week_win = float(top_row["_total_win_j"] or 0.0)
        top_week_share = safe_div(top_week_win, total_win) if total_win > 0 else np.nan

    summary = {
        "weeks_count": int(len(by_week)),
        "week_min": int(by_week["_week"].min()) if len(by_week) else None,
        "week_max": int(by_week["_week"].max()) if len(by_week) else None,
        "total_win": total_win,
        "pure_player_win": pure_player_win,
        "events_delta": float(events_delta),
        "ring_win": ring_win,
        "mtt_win": mtt_win,
        "commission": comm,
        "ring_share": ring_share,
        "mtt_share": mtt_share,
        "profit_to_rake": profit_to_rake,
        "top_week": top_week,
        "top_week_win": top_week_win,
        "top_week_share": top_week_share,
    }
    return summary, by_week


def score_player(db_sum: dict, cop_ring: dict, cop_tour: dict, trf_ring: dict, trf_tour: dict, coverage: dict):
    score = 0
    reasons = []

    total_win = db_sum["total_win"]
    ring_share = db_sum["ring_share"]
    profit_to_rake = db_sum["profit_to_rake"]
    top_week_share = db_sum["top_week_share"]

    # DB базово
    if total_win <= 0:
        score += 5
        reasons.append("DB: игрок в минусе по J — как 'получатель перелива' менее вероятен.")
    else:
        score += 20
        reasons.append("DB: игрок в плюсе по J — базовая проверка обязательна.")

    if total_win > 0 and pd.notna(ring_share):
        if ring_share >= 0.70 and abs(db_sum["ring_win"]) >= 50:
            score += 15
            reasons.append("DB: профит в основном из Ring (кэш).")
        elif ring_share >= 0.50 and abs(db_sum["ring_win"]) >= 50:
            score += 8
            reasons.append("DB: значимая доля профита из Ring.")

    if total_win > 0:
        if db_sum["commission"] <= 0:
            score += 10
            reasons.append("DB: комиссия не видна/нулевая — интерпретация менее надёжна.")
        else:
            if pd.notna(profit_to_rake) and profit_to_rake >= 8 and total_win >= 100:
                score += 10
                reasons.append("DB: очень высокий профит относительно комиссии.")

    if total_win > 0 and db_sum["weeks_count"] >= 2 and pd.notna(top_week_share) and top_week_share >= 0.60:
        score += 8
        reasons.append("DB: профит концентрирован в одной неделе.")

    # Coverage
    if coverage["ring_games"] + coverage["tour_games"] == 0:
        score += 15
        reasons.append("GAMES: игрок не найден в файле игр — перелив по играм не проверить.")
    else:
        reasons.append(f"GAMES: покрытие — RING={coverage['ring_games']}, TOURNAMENT={coverage['tour_games']}.")

    # Co-play: HU учитывать отдельно
    if cop_ring["sessions_count"] >= MIN_SESSIONS_FOR_COPLAY:
        if cop_ring["unique_opponents"] <= 5:
            score += 10
            reasons.append("GAMES/RING: мало уникальных оппонентов (узкий круг).")
        if cop_ring["top2_coplay_share"] >= COPLAY_TOP2_SHARE_SUSP:
            score += 10
            reasons.append("GAMES/RING: топ‑2 оппонента покрывают большую часть игр.")
        if cop_ring["hu_sessions"] >= 6 and (cop_ring["hu_sessions"] / cop_ring["sessions_count"]) >= 0.6:
            score += 8
            reasons.append("GAMES/RING: много HU — co-play сам по себе не доказателен, но усиливает риск при наличии net-flow.")
    elif cop_ring["sessions_count"] > 0:
        score += 3
        reasons.append("GAMES/RING: сессий мало — сигнал co-play слабый.")

    # Net-flow (основа)
    def apply_transfer_block(trf: dict, label: str, net_alert: float):
        nonlocal score, reasons
        if trf["top_net_partner"] is None:
            return
        partner = trf["top_net_partner"]
        top_net = float(trf["top_net"])
        share = float(trf["top_net_share"])
        one_sided = float(trf["one_sidedness"])

        # если net положительный — партнёр "кормит" target; отрицательный — target "кормит" партнёра
        if abs(top_net) >= net_alert:
            score += 25 if label == "RING" else 10
            direction = "в пользу игрока" if top_net > 0 else "в пользу партнёра"
            reasons.append(f"GAMES/{label}: крупный net-flow с одним ID ({partner}), {direction}, net≈{fmt_money(top_net)}.")

        if share >= PAIR_DOMINANCE_ALERT and abs(top_net) >= (net_alert / 2):
            score += 15 if label == "RING" else 6
            reasons.append(f"GAMES/{label}: доминирование одной пары по обороту (доля≈{share:.0%}).")

        if one_sided >= PAIR_ONE_SIDED_ALERT and abs(top_net) >= (net_alert / 2):
            score += 10 if label == "RING" else 4
            reasons.append(f"GAMES/{label}: поток односторонний (one_sided≈{one_sided:.0%}).")

        if trf["extremes"]:
            score += 8 if label == "RING" else 4
            reasons.append(f"GAMES/{label}: есть игры с аномально крупным выигрышем.")

    apply_transfer_block(trf_ring, "RING", PAIR_NET_ALERT_RING)
    apply_transfer_block(trf_tour, "TOURNAMENT", PAIR_NET_ALERT_TOUR)

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)

    if decision == "APPROVE":
        main_risk = "Явных признаков перелива/узкого круга по текущим данным не выявлено."
    elif decision == "FAST_CHECK":
        main_risk = "Есть признаки риска: нужна быстрая проверка перед выводом."
    else:
        main_risk = "Высокий риск перелива/сговора: нужна ручная проверка СБ."

    return score, decision, main_risk, reasons


def build_top_suspicious(db_df: pd.DataFrame, games_df: pd.DataFrame, sessions_df: pd.DataFrame, flows_df: pd.DataFrame,
                        weeks_mode: str, last_n: int, week_from: int, week_to: int,
                        top_n: int = 30) -> pd.DataFrame:
    players = sorted(db_df["_player_id"].unique().tolist())
    res = []

    for pid in players:
        db_sum, _ = build_db_views(db_df, int(pid), weeks_mode, int(last_n), int(week_from), int(week_to))
        if db_sum is None:
            continue

        tg = games_df[games_df["player_id"] == int(pid)] if not games_df.empty else pd.DataFrame()
        coverage = {
            "ring_games": int((tg["game_type"] == "RING").sum()) if not tg.empty else 0,
            "tour_games": int((tg["game_type"] == "TOURNAMENT").sum()) if not tg.empty else 0,
        }

        cop_ring = coplay_features(int(pid), sessions_df, "RING")
        cop_tour = coplay_features(int(pid), sessions_df, "TOURNAMENT")

        trf_ring = transfer_features_v2(int(pid), games_df, flows_df, "RING")
        trf_tour = transfer_features_v2(int(pid), games_df, flows_df, "TOURNAMENT")

        score, decision, _, _ = score_player(db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)

        res.append({
            "player_id": int(pid),
            "risk_score": int(score),
            "decision": decision,
            "db_total_win_j": float(db_sum["total_win"]),
            "db_ring_share": (np.nan if pd.isna(db_sum["ring_share"]) else float(db_sum["ring_share"])),
            "games_ring": coverage["ring_games"],
            "games_tour": coverage["tour_games"],
            "coplay_ring_sessions": cop_ring["sessions_count"],
            "coplay_ring_top1": float(cop_ring["top1_coplay_share"]),
            "flow_ring_top_net": float(trf_ring["top_net"]),
            "flow_ring_top_partner": trf_ring["top_net_partner"] if trf_ring["top_net_partner"] is not None else "",
        })

    out = pd.DataFrame(res)
    if out.empty:
        return out

    out = out.sort_values(["risk_score", "db_total_win_j"], ascending=[False, False]).head(int(top_n)).copy()
    return out


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Загрузка (2 файла)")

    st.subheader("1) Общий CSV (по неделям)")
    db_up = st.file_uploader("DB.csv", type=["csv"], key="db_up")

    st.subheader("2) Games CSV")
    st.caption("Теперь поддерживается и обычный табличный CSV, и текстовый экспорт с 'ID игры:'.")
    games_up = st.file_uploader("Games.csv", type=["csv", "txt"], key="games_up")

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

# validate
if db_file is None:
    st.info("Загрузи общий CSV (по неделям).")
    st.stop()

try:
    db_df = load_db_csv(db_file)
except Exception as e:
    st.error("Не удалось прочитать общий CSV. Проверь разделитель/кодировку/колонки.")
    st.exception(e)
    st.stop()

games_df = pd.DataFrame(columns=["game_id", "game_type", "player_id", "win", "fee"])
sessions_df = pd.DataFrame(columns=["session_id", "game_type", "players", "players_n"])
flows_df = pd.DataFrame(columns=["from_player", "to_player", "game_type", "amount", "games_cnt"])

if games_file is not None:
    try:
        games_df = parse_games_csv(games_file)
        sessions_df = build_sessions_from_games(games_df)
        flows_df = build_pair_flows(games_df)
    except Exception as e:
        st.warning("Games.csv загружен, но парсер не смог корректно разобрать файл. Анализ по играм будет ограничен.")
        st.exception(e)

# UI metrics
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
            "- Пары/источники: co-play + net-flow (кто кому и сколько по играм)."
        )

    if not run:
        st.stop()

    db_sum, by_week = build_db_views(db_df, int(player_id), weeks_mode, int(last_n), int(week_from), int(week_to))
    if db_sum is None:
        st.error("Игрок не найден в DB по выбранному периоду.")
        st.stop()

    tg = games_df[games_df["player_id"] == int(player_id)] if not games_df.empty else pd.DataFrame()
    coverage = {
        "ring_games": int((tg["game_type"] == "RING").sum()) if not tg.empty else 0,
        "tour_games": int((tg["game_type"] == "TOURNAMENT").sum()) if not tg.empty else 0,
    }

    cop_ring = coplay_features(int(player_id), sessions_df, "RING")
    cop_tour = coplay_features(int(player_id), sessions_df, "TOURNAMENT")

    trf_ring = transfer_features_v2(int(player_id), games_df, flows_df, "RING")
    trf_tour = transfer_features_v2(int(player_id), games_df, flows_df, "TOURNAMENT")

    score, decision, main_risk, reasons = score_player(db_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)

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
        for r in reasons[:14]:
            st.markdown(f"- {r}")

    with tabs[1]:
        st.subheader("DB: агрегаты по периоду")
        x1, x2, x3, x4 = st.columns(4, gap="small")
        x1.metric("J: итог (+события)", fmt_money(db_sum["total_win"]), border=True)
        x2.metric("O: выигрыш игрока общий", fmt_money(db_sum["pure_player_win"]), border=True)
        x3.metric("P: Ring win", fmt_money(db_sum["ring_win"]), border=True)
        x4.metric("Q: MTT win", fmt_money(db_sum["mtt_win"]), border=True)

        y1, y2, y3, y4 = st.columns(4, gap="small")
        y1.metric("Комиссия", fmt_money(db_sum["commission"]), border=True)
        y2.metric("Ring доля", "NaN" if pd.isna(db_sum["ring_share"]) else f"{db_sum['ring_share']:.0%}", border=True)
        y3.metric("MTT доля", "NaN" if pd.isna(db_sum["mtt_share"]) else f"{db_sum['mtt_share']:.0%}", border=True)
        y4.metric("Профит/комиссия", "NaN" if pd.isna(db_sum["profit_to_rake"]) else f"{db_sum['profit_to_rake']:.1f}x", border=True)

        st.subheader("По неделям")
        out = by_week.rename(
            columns={
                "_week": "Неделя",
                "_total_win_j": "J: итог (+события)",
                "_player_win_total": "O: выигрыш игрока общий",
                "_player_win_ring": "P: Ring win",
                "_player_win_mtt": "Q: MTT win",
                "_club_income_total": "Доход клуба общий",
                "_club_commission": "Комиссия",
            }
        ).copy()
        out["J-O (дельта событий)"] = out["J: итог (+события)"] - out["O: выигрыш игрока общий"]
        st.dataframe(out.sort_values("Неделя", ascending=False), use_container_width=True)

    with tabs[2]:
        st.subheader("Games: co-play")
        st.markdown(
            f"- RING: сессий={cop_ring['sessions_count']}, HU={cop_ring['hu_sessions']}, ≤3 игроков={cop_ring['sh_sessions']}, "
            f"уникальных оппонентов={cop_ring['unique_opponents']}, топ-1 доля={cop_ring['top1_coplay_share']:.0%}."
        )
        st.markdown(
            f"- TOURNAMENT: сессий={cop_tour['sessions_count']}, уникальных оппонентов={cop_tour['unique_opponents']}, "
            f"топ-1 доля={cop_tour['top1_coplay_share']:.0%}."
        )

        st.subheader("Games: net-flow")
        st.markdown(
            f"- RING: игр={trf_ring['target_games']}, сумма win={fmt_money(trf_ring['target_total_win'])}, "
            f"топ-партнёр={trf_ring['top_net_partner']}, net≈{fmt_money(trf_ring['top_net'])}, "
            f"доля пары≈{trf_ring['top_net_share']:.0%}, one_sided≈{trf_ring['one_sidedness']:.0%}."
        )
        st.markdown(
            f"- TOURNAMENT: игр={trf_tour['target_games']}, сумма win={fmt_money(trf_tour['target_total_win'])}, "
            f"топ-партнёр={trf_tour['top_net_partner']}, net≈{fmt_money(trf_tour['top_net'])}, "
            f"доля пары≈{trf_tour['top_net_share']:.0%}, one_sided≈{trf_tour['one_sidedness']:.0%}."
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
        games_df=games_df,
        sessions_df=sessions_df,
        flows_df=flows_df,
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
    show["db_ring_share"] = show["db_ring_share"].apply(lambda x: "NaN" if pd.isna(x) else f"{x:.0%}")
    show["coplay_ring_top1"] = show["coplay_ring_top1"].apply(lambda x: f"{x:.0%}")
    st.dataframe(show, use_container_width=True)

    csv_bytes = top_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Скачать ТОП в CSV", data=csv_bytes, file_name="top_suspicious.csv", mime="text/csv")
