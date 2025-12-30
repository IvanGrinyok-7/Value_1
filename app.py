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
APP_TITLE = "PPPoker: Anti-Fraud (Общий CSV + Игры CSV)"
CACHE_DIR = Path(".pppoker_app_cache")
CACHE_DIR.mkdir(exist_ok=True)

SUMMARY_KEY = "summary_csv"
GAMES_DIR = CACHE_DIR / "games_files"
GAMES_DIR.mkdir(exist_ok=True)
GAMES_MANIFEST = CACHE_DIR / "games_manifest.json"

# ---- Required column names in summary CSV (как в Primer.csv) ----
COL_WEEK = "Номер недели"
COL_PID = "ID игрока"
COL_TOTAL = "Общий выигрыш игроков + События"          # аналог твоей J
COL_WIN_TOTAL = "Выигрыш игрока Общий"
COL_WIN_RING = "Выигрыш игрока Ring Game"
COL_WIN_MTT = "Выигрыш игрока MTT, SNG"
COL_CLUB_INCOME = "Доход клуба Общий"
COL_CLUB_COMM = "Доход клуба Комиссия"

# ---- Scoring thresholds ----
T_APPROVE = 25
T_FAST_CHECK = 55

MIN_SESSIONS_FOR_COPLAY = 6
COPLAY_TOP1_SHARE_SUSP = 0.60
COPLAY_TOP2_SHARE_SUSP = 0.80

PAIR_NET_TRANSFER_ALERT_RING = 25.0
PAIR_NET_TRANSFER_ALERT_TOUR = 60.0
PAIR_DOMINANCE_ALERT = 0.70

SINGLE_GAME_WIN_ALERT_RING = 60.0
SINGLE_GAME_WIN_ALERT_TOUR = 150.0

# ---- Games parsing ----
TOUR_HINT_RE = re.compile(r"\bPPST/|бай-ин:\s*|satellite|pko|mko\b", re.IGNORECASE)
RING_HINT_RE = re.compile(r"\bPPSR/|Ring|NLH\s+\d|PLO|Bomb Pot|Ante\b", re.IGNORECASE)
GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)


# =========================
# PERSISTENT FILE CACHE (single file)
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
            name = key
    return BytesFile(content, name)


def cache_meta(key: str) -> dict | None:
    mp = _meta_path(key)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return None


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
# PERSISTENT FILE CACHE (games multi)
# =========================
def _load_games_manifest() -> list[dict]:
    if not GAMES_MANIFEST.exists():
        return []
    try:
        return json.loads(GAMES_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_games_manifest(items: list[dict]) -> None:
    GAMES_MANIFEST.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _file_id_from_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def games_add_files(uploaded_files) -> None:
    if not uploaded_files:
        return
    manifest = _load_games_manifest()
    seen = {m["id"] for m in manifest}

    for uf in uploaded_files:
        content = uf.getvalue()
        fid = _file_id_from_bytes(content)
        if fid in seen:
            continue

        fname = getattr(uf, "name", f"{fid}.csv")
        path = GAMES_DIR / f"{fid}.csv"
        path.write_bytes(content)

        manifest.append(
            {
                "id": fid,
                "name": fname,
                "bytes": len(content),
                "saved_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "path": str(path),
            }
        )
        seen.add(fid)

    _save_games_manifest(manifest)


def games_remove_file(file_id: str) -> None:
    manifest = _load_games_manifest()
    keep = []
    for m in manifest:
        if m.get("id") == file_id:
            p = Path(m.get("path", ""))
            if p.exists():
                p.unlink()
        else:
            keep.append(m)
    _save_games_manifest(keep)


def games_clear_all() -> None:
    manifest = _load_games_manifest()
    for m in manifest:
        p = Path(m.get("path", ""))
        if p.exists():
            p.unlink()
    _save_games_manifest([])


def games_load_all_bytes() -> list[BytesFile]:
    res = []
    for m in _load_games_manifest():
        p = Path(m.get("path", ""))
        if p.exists():
            res.append(BytesFile(p.read_bytes(), m.get("name", p.name)))
    return res


# =========================
# HELPERS
# =========================
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


def read_csv_smart_bytes(content: bytes) -> pd.DataFrame:
    # пытаемся ; потом , (Primer.csv читался так)
    bio = io.BytesIO(content)
    try:
        df = pd.read_csv(bio, sep=";", engine="python")
        if df.shape[1] == 1:
            bio = io.BytesIO(content)
            df = pd.read_csv(bio, sep=",", engine="python")
        return df
    except Exception:
        bio = io.BytesIO(content)
        return pd.read_csv(bio, engine="python")


# =========================
# LOAD SUMMARY CSV
# =========================
def load_summary_csv(file_obj) -> pd.DataFrame:
    df = read_csv_smart_bytes(file_obj.getvalue())

    required = [COL_WEEK, COL_PID, COL_TOTAL, COL_WIN_TOTAL, COL_WIN_RING, COL_WIN_MTT, COL_CLUB_INCOME, COL_CLUB_COMM]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"В summary CSV не найдены колонки: {miss}. Проверь шапку файла.")

    out = pd.DataFrame()
    out["_week"] = pd.to_numeric(df[COL_WEEK], errors="coerce").fillna(-1).astype(int)
    out["_player_id"] = pd.to_numeric(df[COL_PID], errors="coerce")
    out = out.dropna(subset=["_player_id"]).copy()
    out["_player_id"] = out["_player_id"].astype(int)

    out["_total_win"] = to_float_series(df.loc[out.index, COL_TOTAL])
    out["_player_win_total"] = to_float_series(df.loc[out.index, COL_WIN_TOTAL])
    out["_player_win_ring"] = to_float_series(df.loc[out.index, COL_WIN_RING])
    out["_player_win_mtt"] = to_float_series(df.loc[out.index, COL_WIN_MTT])
    out["_club_income_total"] = to_float_series(df.loc[out.index, COL_CLUB_INCOME])
    out["_club_commission"] = to_float_series(df.loc[out.index, COL_CLUB_COMM])

    return out


# =========================
# GAMES PARSING
# =========================
def classify_game_type(block_lines: list[str]) -> str:
    text = " ".join(block_lines)
    if TOUR_HINT_RE.search(text):
        return "TOURNAMENT"
    if RING_HINT_RE.search(text):
        return "RING"
    return "UNKNOWN"


def parse_games_csv_bytes(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    rows = []
    current_game_id = None
    block_lines = []
    header = None
    pid_idx = win_idx = fee_idx = None

    def split_semicolon(line: str):
        return [c.strip().strip('"') for c in line.split(";")]

    for line in lines:
        m = GAME_ID_RE.search(line)
        if m:
            current_game_id = m.group(1).strip()
            block_lines = [line]
            header = None
            pid_idx = win_idx = fee_idx = None
            continue

        if not current_game_id:
            continue

        block_lines.append(line)

        if "ID игрока" in line:
            header = split_semicolon(line)
            pid_idx = header.index("ID игрока") if "ID игрока" in header else 0
            win_idx = header.index("Выигрыш") if "Выигрыш" in header else None
            fee_idx = header.index("Комиссия") if "Комиссия" in header else None
            continue

        if header is None:
            continue

        parts = split_semicolon(line)
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

        gtype = classify_game_type(block_lines)
        rows.append({"game_id": current_game_id, "game_type": gtype, "player_id": pid, "win": win, "fee": fee})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["game_id", "game_type", "player_id", "win", "fee"])
    return df


def build_sessions_from_games(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["session_id", "game_type", "players"])

    g = (
        games_df.groupby(["game_id", "game_type"])["player_id"]
        .apply(lambda x: sorted(set(int(v) for v in x.dropna().tolist())))
        .reset_index()
        .rename(columns={"game_id": "session_id", "player_id": "players"})
    )
    g = g[g["players"].apply(lambda ps: len(ps) >= 2)].copy()
    return g


def coplay_features(target_id: int, sessions_df: pd.DataFrame, game_type: str | None = None) -> dict:
    if sessions_df.empty:
        return {"sessions_count": 0, "unique_opponents": 0, "top1_coplay_share": 0.0, "top2_coplay_share": 0.0, "top_partners": []}

    df = sessions_df
    if game_type:
        df = df[df["game_type"] == game_type]

    rows = df[df["players"].apply(lambda ps: target_id in ps)]
    sessions_count = int(len(rows))
    if sessions_count == 0:
        return {"sessions_count": 0, "unique_opponents": 0, "top1_coplay_share": 0.0, "top2_coplay_share": 0.0, "top_partners": []}

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

    return {
        "sessions_count": sessions_count,
        "unique_opponents": unique_opponents,
        "top1_coplay_share": float(top1 / sessions_count) if sessions_count else 0.0,
        "top2_coplay_share": float((top1 + top2) / sessions_count) if sessions_count else 0.0,
        "top_partners": partners[:12],
    }


def transfer_features(target_id: int, games_df: pd.DataFrame, game_type: str | None = None) -> dict:
    if games_df.empty:
        return {"target_games": 0, "target_total_win": 0.0, "top_sources": [], "top_source_net": 0.0, "top_source_share": 0.0, "extremes": []}

    df = games_df.copy()
    if game_type:
        df = df[df["game_type"] == game_type]

    df = df[pd.notna(df["win"])].copy()
    t = df[df["player_id"] == target_id][["game_id", "win"]].copy()
    if t.empty:
        return {"target_games": 0, "target_total_win": 0.0, "top_sources": [], "top_source_net": 0.0, "top_source_share": 0.0, "extremes": []}

    transfer = {}
    extremes = []
    win_alert = SINGLE_GAME_WIN_ALERT_TOUR if game_type == "TOURNAMENT" else SINGLE_GAME_WIN_ALERT_RING

    for _, tr in t.iterrows():
        gid = tr["game_id"]
        t_win = float(tr["win"])

        if t_win >= win_alert:
            extremes.append({"game_id": gid, "target_win": t_win})

        if t_win <= 0:
            continue

        game = df[df["game_id"] == gid][["player_id", "win"]]
        losers = game[game["win"] < 0]
        if losers.empty:
            continue

        for _, lr in losers.iterrows():
            pid = int(lr["player_id"])
            loss = float(-lr["win"])
            amt = min(t_win, loss)
            transfer[pid] = transfer.get(pid, 0.0) + amt

    total_from_sources = float(sum(transfer.values()))
    sources_sorted = sorted(transfer.items(), key=lambda x: x[1], reverse=True)
    top_source_net = float(sources_sorted[0][1]) if sources_sorted else 0.0
    top_source_share = (top_source_net / total_from_sources) if total_from_sources > 0 else 0.0

    return {
        "target_games": int(len(t)),
        "target_total_win": float(t["win"].sum()),
        "top_sources": sources_sorted[:12],
        "top_source_net": top_source_net,
        "top_source_share": float(top_source_share),
        "extremes": extremes[:12],
    }


# =========================
# SUMMARY VIEW + SCORE
# =========================
def build_summary_views(df: pd.DataFrame, player_id: int, weeks_mode: str, last_n: int, week_from: int, week_to: int):
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
                "_total_win",
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

    agg = by_week[[
        "_total_win",
        "_player_win_total",
        "_player_win_ring",
        "_player_win_mtt",
        "_club_income_total",
        "_club_commission",
    ]].sum(numeric_only=True)

    total_win = float(agg.get("_total_win", 0.0) or 0.0)
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
        top_row = by_week.sort_values("_total_win", ascending=False).iloc[0]
        top_week = int(top_row["_week"])
        top_week_win = float(top_row["_total_win"] or 0.0)
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


def score_player(sum_sum: dict, cop_ring: dict, cop_tour: dict, trf_ring: dict, trf_tour: dict, coverage: dict):
    score = 0
    reasons = []

    total_win = sum_sum["total_win"]
    ring_share = sum_sum["ring_share"]
    profit_to_rake = sum_sum["profit_to_rake"]
    top_week_share = sum_sum["top_week_share"]

    # --- Summary baseline ---
    if total_win <= 0:
        score += 5
        reasons.append("ОБЩИЙ: игрок в минусе по итогу (+события) — как 'получатель' менее вероятен.")
    else:
        score += 20
        reasons.append("ОБЩИЙ: игрок в плюсе по итогу (+события) — базовый триггер на проверку источника.")

    if total_win > 0 and pd.notna(ring_share):
        if ring_share >= 0.70 and abs(sum_sum["ring_win"]) >= 50:
            score += 15
            reasons.append("ОБЩИЙ: большая доля профита из Ring (кэш) — более рискованный формат для переливов.")
        elif ring_share >= 0.50 and abs(sum_sum["ring_win"]) >= 50:
            score += 8
            reasons.append("ОБЩИЙ: заметная доля профита из Ring (кэш).")

    if total_win > 0:
        if sum_sum["commission"] <= 0:
            score += 10
            reasons.append("ОБЩИЙ: комиссия не видна/нулевая — интерпретация профита менее надёжна.")
        else:
            if pd.notna(profit_to_rake) and profit_to_rake >= 8 and total_win >= 100:
                score += 10
                reasons.append("ОБЩИЙ: профит очень высокий относительно комиссии — выглядит как 'дешёвый' профит.")

    if total_win > 0 and sum_sum["weeks_count"] >= 2 and pd.notna(top_week_share) and top_week_share >= 0.60:
        score += 8
        reasons.append("ОБЩИЙ: концентрация профита в одной неделе — похоже на 'разовый занос под вывод'.")

    # --- Games coverage ---
    if coverage["ring_games"] + coverage["tour_games"] == 0:
        score += 15
        reasons.append("ИГРЫ: игрок не найден в файлах игр — нет проверки на перелив по играм.")
    else:
        reasons.append(f"ИГРЫ: покрытие — RING игр={coverage['ring_games']}, TOURNAMENT игр={coverage['tour_games']}.")

    # --- co-play (ring)
    if cop_ring["sessions_count"] >= MIN_SESSIONS_FOR_COPLAY:
        if cop_ring["top1_coplay_share"] >= COPLAY_TOP1_SHARE_SUSP:
            score += 25
            reasons.append("ИГРЫ/RING: один и тот же оппонент слишком часто (узкий круг).")
        if cop_ring["top2_coplay_share"] >= COPLAY_TOP2_SHARE_SUSP:
            score += 15
            reasons.append("ИГРЫ/RING: топ‑2 оппонента покрывают большую часть сессий (узкий круг).")
        if cop_ring["unique_opponents"] <= 5:
            score += 10
            reasons.append("ИГРЫ/RING: мало уникальных оппонентов.")
    elif cop_ring["sessions_count"] > 0:
        score += 3
        reasons.append("ИГРЫ/RING: сессий мало — co-play сигнал слабый.")

    # --- transfer-proxy
    if trf_ring["top_sources"]:
        top_pid, top_amt = trf_ring["top_sources"][0]
        if top_amt >= PAIR_NET_TRANSFER_ALERT_RING:
            score += 25
            reasons.append(f"ИГРЫ/RING: крупный поток к игроку от одного ID (источник {top_pid} ≈ {fmt_money(top_amt)}).")
        if trf_ring["top_source_share"] >= PAIR_DOMINANCE_ALERT and top_amt >= (PAIR_NET_TRANSFER_ALERT_RING / 2):
            score += 15
            reasons.append("ИГРЫ/RING: доминирование одного источника результата.")
        if trf_ring["extremes"]:
            score += 8
            reasons.append("ИГРЫ/RING: есть игры с аномально крупным выигрышем.")

    if trf_tour["top_sources"]:
        top_pid, top_amt = trf_tour["top_sources"][0]
        if top_amt >= PAIR_NET_TRANSFER_ALERT_TOUR:
            score += 8
            reasons.append(f"ИГРЫ/TOURNAMENT: крупный поток от одного ID (источник {top_pid} ≈ {fmt_money(top_amt)}).")
        if trf_tour["extremes"]:
            score += 5
            reasons.append("ИГРЫ/TOURNAMENT: есть аномально крупные выигрыши (турниры более шумные).")

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)

    if decision == "APPROVE":
        main_risk = "Явных признаков перелива/узкого круга по текущим данным не выявлено."
    elif decision == "FAST_CHECK":
        main_risk = "Есть признаки риска: нужна быстрая проверка (перед выводом)."
    else:
        main_risk = "Высокий риск перелива/сговора: требуется ручная проверка СБ."

    return score, decision, main_risk, reasons


# =========================
# TOP SUSPICIOUS (скоринг по всем)
# =========================
def filter_summary_period(summary_df: pd.DataFrame, weeks_mode: str, last_n: int, week_from: int, week_to: int) -> pd.DataFrame:
    df = summary_df.copy()
    valid_weeks = sorted([w for w in df["_week"].unique().tolist() if w >= 0])
    if not valid_weeks:
        return df

    if weeks_mode == "Все недели":
        return df

    if weeks_mode == "Последние N недель":
        max_w = max(valid_weeks)
        min_w = max_w - max(0, int(last_n) - 1)
        return df[(df["_week"] >= min_w) & (df["_week"] <= max_w)].copy()

    return df[(df["_week"] >= int(week_from)) & (df["_week"] <= int(week_to))].copy()


def compute_top_suspicious(summary_df_period: pd.DataFrame, games_df: pd.DataFrame, sessions_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    # агрегаты по summary
    agg = (
        summary_df_period.groupby("_player_id", as_index=False)[
            ["_total_win", "_player_win_total", "_player_win_ring", "_player_win_mtt", "_club_income_total", "_club_commission"]
        ]
        .sum(min_count=1)
    )

    # подготовим быстрые счётчики coverage по games
    if games_df.empty:
        ring_counts = {}
        tour_counts = {}
    else:
        g = games_df.groupby(["player_id", "game_type"]).size().reset_index(name="cnt")
        ring_counts = dict(zip(g[g["game_type"] == "RING"]["player_id"], g[g["game_type"] == "RING"]["cnt"]))
        tour_counts = dict(zip(g[g["game_type"] == "TOURNAMENT"]["player_id"], g[g["game_type"] == "TOURNAMENT"]["cnt"]))

    # (не самый быстрый вариант, но стабильный): считаем score на основе функций для выбранного игрока
    # чтобы топ работал приемлемо, ограничим анализом игроков, которые вообще в плюсе или есть в games.
    candidate_ids = set(agg["_player_id"].tolist())
    if not games_df.empty:
        candidate_ids |= set(games_df["player_id"].unique().tolist())

    rows = []
    for pid in candidate_ids:
        sum_sum, _ = build_summary_views(summary_df_period, int(pid), "Все недели", 1, 0, 0)
        if sum_sum is None:
            continue

        # games features для топа: считаем в “облегчённом” режиме (через функции по sessions_df / games_df)
        cop_ring = coplay_features(int(pid), sessions_df, "RING")
        cop_tour = coplay_features(int(pid), sessions_df, "TOURNAMENT")
        trf_ring = transfer_features(int(pid), games_df, "RING")
        trf_tour = transfer_features(int(pid), games_df, "TOURNAMENT")

        coverage = {
            "ring_games": int(ring_counts.get(int(pid), 0)),
            "tour_games": int(tour_counts.get(int(pid), 0)),
        }

        score, decision, main_risk, _reasons = score_player(sum_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)

        rows.append(
            {
                "player_id": int(pid),
                "risk_score": int(score),
                "decision": decision,
                "total_win": float(sum_sum["total_win"]),
                "ring_share": np.nan if pd.isna(sum_sum["ring_share"]) else float(sum_sum["ring_share"]),
                "games_ring": int(coverage["ring_games"]),
                "games_tour": int(coverage["tour_games"]),
                "main_risk": main_risk,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["risk_score", "total_win"], ascending=[False, False]).head(int(top_n)).copy()
    out["ring_share"] = out["ring_share"].apply(lambda x: "NaN" if pd.isna(x) else f"{x:.0%}")
    out["total_win"] = out["total_win"].apply(lambda x: float(x))
    return out


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Загрузка данных")

    st.subheader("1) Общий файл (CSV)")
    summary_up = st.file_uploader("Загрузи общий CSV (как Primer.csv)", type=["csv"], key="summary_up")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Очистить общий", use_container_width=True):
            cache_clear(SUMMARY_KEY)
            st.rerun()
    with col_b:
        if st.button("Очистить ВСЁ", use_container_width=True):
            cache_clear(SUMMARY_KEY)
            games_clear_all()
            st.rerun()

    summary_file = resolve_file(SUMMARY_KEY, summary_up)
    meta = cache_meta(SUMMARY_KEY)
    if meta:
        st.caption(f"Сохранён общий: {meta.get('name','file')} ({meta.get('saved_at','')})")
    else:
        st.caption("Общий файл не загружен.")

    st.divider()
    st.subheader("2) Игры (CSV, пачкой)")
    games_up = st.file_uploader(
        "Загрузи файлы 'Игры' (CSV), можно сразу пачкой (35+)",
        type=["csv"],
        accept_multiple_files=True,
        key="games_up",
    )
    if games_up:
        games_add_files(games_up)
        st.rerun()

    manifest = _load_games_manifest()
    st.caption(f"Сохранено файлов игр: {len(manifest)}")

    if manifest:
        with st.expander("Управление games-файлами", expanded=False):
            names = [f"{m['name']}  |  {m['id']}  |  {m.get('saved_at','')}" for m in manifest]
            sel = st.selectbox("Выбери файл для удаления", options=["—"] + names)
            if st.button("Удалить выбранный", use_container_width=True):
                if sel != "—":
                    file_id = sel.split("|")[1].strip()
                    games_remove_file(file_id)
                    st.rerun()

            if st.button("Удалить ВСЕ games-файлы", type="secondary", use_container_width=True):
                games_clear_all()
                st.rerun()

    st.divider()
    st.subheader("Период (по неделям)")
    weeks_mode = st.selectbox("Фильтр недель", ["Все недели", "Последние N недель", "Диапазон недель"])
    last_n = st.number_input("N (если выбран 'Последние N недель')", min_value=1, value=4, step=1)

# Require summary
if summary_file is None:
    st.info("Загрузи общий CSV (как Primer.csv).")
    st.stop()

try:
    summary_df = load_summary_csv(summary_file)
except Exception as e:
    st.error("Не удалось прочитать общий CSV. Проверь, что файл содержит шапку и нужные колонки.")
    st.exception(e)
    st.stop()

# Games: optional
games_files = games_load_all_bytes()
games_rows = []
for gf in games_files:
    try:
        gdf = parse_games_csv_bytes(gf.getvalue())
        games_rows.append(gdf)
    except Exception:
        continue

games_df = pd.concat(games_rows, ignore_index=True) if games_rows else pd.DataFrame(columns=["game_id", "game_type", "player_id", "win", "fee"])
sessions_df = build_sessions_from_games(games_df)

# Determine min/max week for UI
valid_weeks = sorted([w for w in summary_df["_week"].unique().tolist() if w >= 0])
w_min = min(valid_weeks) if valid_weeks else 0
w_max = max(valid_weeks) if valid_weeks else 0

# Header metrics
m1, m2, m3, m4 = st.columns(4, gap="small")
m1.metric("Общий: строк", f"{len(summary_df)}", border=True)
m2.metric("Общий: игроков", f"{summary_df['_player_id'].nunique()}", border=True)
m3.metric("Игры: строк", f"{len(games_df)}", border=True)
m4.metric("Игры: сессий", f"{len(sessions_df)}", border=True)

st.divider()

tab_check, tab_top = st.tabs(["Проверка ID", "Топ подозрительных"])

with tab_check:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Проверка игрока")
        default_id = int(summary_df["_player_id"].iloc[0]) if len(summary_df) else 0
        player_id = st.number_input("ID игрока", min_value=0, value=default_id, step=1)

        week_from = st.number_input("Неделя от (если диапазон)", value=w_min, step=1)
        week_to = st.number_input("Неделя до (если диапазон)", value=w_max, step=1)

        run = st.button("Проверить", type="primary", use_container_width=True)

    with right:
        st.subheader("Что считается")
        st.markdown(
            "- **Общий CSV**: итог (+события), доли Ring/MTT, комиссия, концентрация результата по неделям.\n"
            "- **Игры CSV**: co-play (узкий круг), transfer-proxy (доминирующий источник), экстремальные выигрыши.\n"
            "- Итог: Risk score + Decision + причины + таблицы партнёров/источников."
        )

    if not run:
        st.stop()

    sum_sum, by_week = build_summary_views(summary_df, int(player_id), weeks_mode, int(last_n), int(week_from), int(week_to))
    if sum_sum is None:
        st.error("Игрок не найден в общем CSV по выбранному периоду.")
        st.stop()

    tg = games_df[games_df["player_id"] == int(player_id)] if not games_df.empty else pd.DataFrame()
    coverage = {
        "ring_games": int((tg["game_type"] == "RING").sum()) if not tg.empty else 0,
        "tour_games": int((tg["game_type"] == "TOURNAMENT").sum()) if not tg.empty else 0,
    }

    cop_ring = coplay_features(int(player_id), sessions_df, "RING")
    cop_tour = coplay_features(int(player_id), sessions_df, "TOURNAMENT")
    trf_ring = transfer_features(int(player_id), games_df, "RING")
    trf_tour = transfer_features(int(player_id), games_df, "TOURNAMENT")

    score, decision, main_risk, reasons = score_player(sum_sum, cop_ring, cop_tour, trf_ring, trf_tour, coverage)

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

    tabs = st.tabs(["Кратко", "Детали (общий)", "Детали (игры)", "Пары/источники"])

    with tabs[0]:
        st.subheader("Главный риск")
        st.info(main_risk)
        st.subheader("Причины")
        for r in reasons[:12]:
            st.markdown(f"- {r}")

    with tabs[1]:
        st.subheader("Общий: агрегаты по периоду")
        x1, x2, x3, x4 = st.columns(4, gap="small")
        x1.metric("Итог (+события)", fmt_money(sum_sum["total_win"]), border=True)
        x2.metric("Выигрыш игрока (общ.)", fmt_money(sum_sum["pure_player_win"]), border=True)
        x3.metric("Ring win", fmt_money(sum_sum["ring_win"]), border=True)
        x4.metric("MTT win", fmt_money(sum_sum["mtt_win"]), border=True)

        y1, y2, y3, y4 = st.columns(4, gap="small")
        y1.metric("Комиссия", fmt_money(sum_sum["commission"]), border=True)
        y2.metric("Ring доля", "NaN" if pd.isna(sum_sum["ring_share"]) else f"{sum_sum['ring_share']:.0%}", border=True)
        y3.metric("MTT доля", "NaN" if pd.isna(sum_sum["mtt_share"]) else f"{sum_sum['mtt_share']:.0%}", border=True)
        y4.metric("Профит/комиссия", "NaN" if pd.isna(sum_sum["profit_to_rake"]) else f"{sum_sum['profit_to_rake']:.1f}x", border=True)

        st.subheader("По неделям")
        out = by_week.rename(
            columns={
                "_week": "Неделя",
                "_total_win": "Итог (+события)",
                "_player_win_total": "Выигрыш игрока (общ.)",
                "_player_win_ring": "Ring win",
                "_player_win_mtt": "MTT win",
                "_club_income_total": "Доход клуба общий",
                "_club_commission": "Комиссия",
            }
        ).copy()
        out["Итог - выигрыш игрока (дельта событий)"] = out["Итог (+события)"] - out["Выигрыш игрока (общ.)"]
        st.dataframe(out.sort_values("Неделя", ascending=False), use_container_width=True)

    with tabs[2]:
        st.subheader("Игры: co-play (узкий круг)")
        st.markdown(
            f"- RING: сессий={cop_ring['sessions_count']}, уникальных оппонентов={cop_ring['unique_opponents']}, "
            f"топ-1 доля={cop_ring['top1_coplay_share']:.0%}, топ-2 доля={cop_ring['top2_coplay_share']:.0%}."
        )
        st.markdown(
            f"- TOURNAMENT: сессий={cop_tour['sessions_count']}, уникальных оппонентов={cop_tour['unique_opponents']}, "
            f"топ-1 доля={cop_tour['top1_coplay_share']:.0%}."
        )

        st.subheader("Игры: transfer-proxy (источники результата)")
        st.markdown(
            f"- RING: игр={trf_ring['target_games']}, сумма win={fmt_money(trf_ring['target_total_win'])}, "
            f"топ-источник={fmt_money(trf_ring['top_source_net'])}, доля={trf_ring['top_source_share']:.0%}."
        )
        st.markdown(
            f"- TOURNAMENT: игр={trf_tour['target_games']}, сумма win={fmt_money(trf_tour['target_total_win'])}, "
            f"топ-источник={fmt_money(trf_tour['top_source_net'])}, доля={trf_tour['top_source_share']:.0%}."
        )

        if trf_ring["extremes"]:
            st.subheader("Аномально крупные выигрыши (RING)")
            st.dataframe(pd.DataFrame(trf_ring["extremes"]), use_container_width=True)

        if trf_tour["extremes"]:
            st.subheader("Аномально крупные выигрыши (TOURNAMENT)")
            st.dataframe(pd.DataFrame(trf_tour["extremes"]), use_container_width=True)

    with tabs[3]:
        st.subheader("Co-play партнёры (RING)")
        if cop_ring["top_partners"]:
            st.dataframe(pd.DataFrame(cop_ring["top_partners"], columns=["partner_id", "coplay_sessions"]), use_container_width=True)
        else:
            st.info("Нет данных по партнёрам в RING.")

        st.subheader("Co-play партнёры (TOURNAMENT)")
        if cop_tour["top_partners"]:
            st.dataframe(pd.DataFrame(cop_tour["top_partners"], columns=["partner_id", "coplay_sessions"]), use_container_width=True)
        else:
            st.info("Нет данных по партнёрам в TOURNAMENT.")

        st.subheader("Источники transfer-proxy (RING)")
        if trf_ring["top_sources"]:
            st.dataframe(pd.DataFrame(trf_ring["top_sources"], columns=["source_player_id", "estimated_transfer_to_target"]), use_container_width=True)
        else:
            st.info("Нет выраженных источников в RING (по текущим файлам).")

        st.subheader("Источники transfer-proxy (TOURNAMENT)")
        if trf_tour["top_sources"]:
            st.dataframe(pd.DataFrame(trf_tour["top_sources"], columns=["source_player_id", "estimated_transfer_to_target"]), use_container_width=True)
        else:
            st.info("Нет выраженных источников в TOURNAMENT (по текущим файлам).")

with tab_top:
    st.subheader("Топ подозрительных (по выбранному периоду)")

    top_n = st.slider("Сколько показать", min_value=10, max_value=200, value=50, step=10)
    week_from_top = st.number_input("Неделя от (если диапазон)", value=w_min, step=1, key="top_w_from")
    week_to_top = st.number_input("Неделя до (если диапазон)", value=w_max, step=1, key="top_w_to")

    summary_period = filter_summary_period(summary_df, weeks_mode, int(last_n), int(week_from_top), int(week_to_top))

    if st.button("Посчитать ТОП", type="primary"):
        with st.spinner("Считаю ТОП... (может занять время на больших объёмах)"):
            top_df = compute_top_suspicious(summary_period, games_df, sessions_df, int(top_n))

        if top_df is None or top_df.empty:
            st.info("Не удалось сформировать ТОП (проверь, что есть общий CSV и/или games-файлы).")
        else:
            st.dataframe(top_df, use_container_width=True)
            st.caption("Подсказка: кликни по player_id и проверь его во вкладке 'Проверка ID' для полной расшифровки.")
