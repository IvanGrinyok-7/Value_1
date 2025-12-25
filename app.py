import io
import re
import json
import hashlib
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# Optional dependency for persistent storage
try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:
    boto3 = None
    ClientError = Exception


# =========================
# APP CONFIG
# =========================
APP_TITLE = "PPPoker: проверка риска вывода (2 недели)"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)


# =========================
# STORAGE (R2 preferred)
# =========================
DATA_PREFIX = "pppoker_shared/"
META_KEY = DATA_PREFIX + "_meta.json"


@dataclass
class StoredUpload:
    name: str
    data: bytes
    def getvalue(self):
        return self.data


def _now() -> dt.datetime:
    return dt.datetime.now()


def _now_str() -> str:
    return _now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_ts(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


class StorageBackend:
    def enabled(self) -> bool:
        return False

    def load_bytes(self, key: str) -> bytes | None:
        return None

    def save_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def load_meta(self) -> dict:
        raw = self.load_bytes(META_KEY)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def save_meta(self, meta: dict) -> None:
        self.save_bytes(META_KEY, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"), content_type="application/json")


class R2Backend(StorageBackend):
    def __init__(self):
        self.ok = False
        self.bucket = None
        self.client = None

        if boto3 is None:
            return

        # Streamlit Secrets expected:
        # R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
        try:
            endpoint = st.secrets["R2_ENDPOINT"]
            access_key = st.secrets["R2_ACCESS_KEY_ID"]
            secret_key = st.secrets["R2_SECRET_ACCESS_KEY"]
            bucket = st.secrets["R2_BUCKET"]
        except Exception:
            return

        try:
            self.client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="auto",
            )
            self.bucket = bucket
            # lightweight check
            self.client.head_bucket(Bucket=self.bucket)
            self.ok = True
        except Exception:
            self.ok = False

    def enabled(self) -> bool:
        return bool(self.ok)

    def load_bytes(self, key: str) -> bytes | None:
        if not self.ok:
            return None
        try:
            r = self.client.get_object(Bucket=self.bucket, Key=key)
            return r["Body"].read()
        except ClientError:
            return None
        except Exception:
            return None

    def save_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        if not self.ok:
            raise RuntimeError("R2 backend not configured")
        extra = {}
        if content_type:
            extra["ContentType"] = content_type
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)

    def delete(self, key: str) -> None:
        if not self.ok:
            return
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except Exception:
            pass


class MemoryBackend(StorageBackend):
    """
    Фоллбек (без внешнего хранилища) — хранит только в st.session_state текущего пользователя.
    Это НЕ общий режим и НЕ переживает reboot Community Cloud.
    """
    def __init__(self):
        if "_mem_store" not in st.session_state:
            st.session_state["_mem_store"] = {}

    def enabled(self) -> bool:
        return True

    def load_bytes(self, key: str) -> bytes | None:
        return st.session_state["_mem_store"].get(key)

    def save_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        st.session_state["_mem_store"][key] = data

    def delete(self, key: str) -> None:
        if key in st.session_state["_mem_store"]:
            del st.session_state["_mem_store"][key]


def get_storage() -> StorageBackend:
    r2 = R2Backend()
    if r2.enabled():
        return r2
    return MemoryBackend()


storage = get_storage()


def is_stale_saved_at(saved_at_str: str | None) -> bool:
    ts = _parse_ts(saved_at_str)
    if ts is None:
        return True
    return _now().date() > ts.date()


def load_shared_file(key_file: str) -> StoredUpload | None:
    meta = storage.load_meta().get(key_file, {})
    raw = storage.load_bytes(DATA_PREFIX + key_file)
    if raw is None:
        return None
    return StoredUpload(name=meta.get("filename", key_file), data=raw)


def save_shared_file(key_file: str, uploaded_file, label: str, uploaded_by: str):
    data = uploaded_file.getvalue()
    key = DATA_PREFIX + key_file

    # content type (optional)
    ct = None
    low = key_file.lower()
    if low.endswith(".json"):
        ct = "application/json"
    elif low.endswith(".csv"):
        ct = "text/csv"
    elif low.endswith(".xlsx"):
        ct = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    storage.save_bytes(key, data, content_type=ct)

    meta = storage.load_meta()
    meta[key_file] = {
        "label": label,
        "filename": getattr(uploaded_file, "name", key_file),
        "size_bytes": len(data),
        "md5": _md5(data),
        "saved_at": _now_str(),
        "uploaded_by": (uploaded_by or "—").strip(),
        "backend": storage.__class__.__name__,
    }
    storage.save_meta(meta)


def delete_shared_file(key_file: str):
    storage.delete(DATA_PREFIX + key_file)
    meta = storage.load_meta()
    if key_file in meta:
        del meta[key_file]
        storage.save_meta(meta)


def delete_all_shared_files(keys: list[str]):
    for k in keys:
        delete_shared_file(k)


def show_file_status(key_file: str):
    meta = storage.load_meta().get(key_file)
    if not meta:
        st.caption("Сохранённого файла нет.")
        st.error("Нет данных: нужно загрузить файл.")
        return

    saved_at = meta.get("saved_at")
    who = meta.get("uploaded_by", "—")
    fn = meta.get("filename", "—")
    size_b = meta.get("size_bytes", "—")

    st.caption(f"Сохранённый файл: {fn} | загружен: {saved_at} | кто: {who} | размер: {size_b} байт")

    if is_stale_saved_at(saved_at):
        st.error("Данные устарели: наступил новый день — нужно обновить выгрузку.")
    else:
        st.success("Данные актуальны на сегодня.")


def shared_uploader(label: str, key_file: str, type_list: list[str], uploaded_by: str):
    show_file_status(key_file)
    uploaded = st.file_uploader(label, type=type_list, key=f"uploader_{key_file}")

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Удалить файл", key=f"del_{key_file}"):
            delete_shared_file(key_file)
            st.rerun()
    with c2:
        st.caption("Если новый файл не загружать — будет использован сохранённый (общий для приглашённых пользователей).")

    if uploaded is not None:
        save_shared_file(key_file, uploaded, label, uploaded_by=uploaded_by)
        st.success("Файл сохранён.")
        st.rerun()

    return load_shared_file(key_file)


# =========================
# ANALYTICS CORE
# =========================
def col_idx(col_letter: str) -> int:
    col_letter = col_letter.strip().upper()
    n = 0
    for ch in col_letter:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


IDX_ID = col_idx("A")
IDX_NET_TOTAL = col_idx("I")
IDX_NET_RING = col_idx("O")
IDX_NET_MTT = col_idx("P")
IDX_COMM = col_idx("AB")

T_APPROVE = 25
T_FAST_CHECK = 55

MIN_SESSIONS_FOR_COPLAY = 6
COPLAY_TOP1_SHARE_SUSP = 0.60
COPLAY_TOP2_SHARE_SUSP = 0.80

PAIR_NET_TRANSFER_ALERT_RING = 25.0
PAIR_NET_TRANSFER_ALERT_TOUR = 60.0
PAIR_DOMINANCE_ALERT = 0.70

SINGLE_GAME_WIN_ALERT_RING = 60.0
SINGLE_GAME_LOSS_ALERT_RING = 60.0
SINGLE_GAME_WIN_ALERT_TOUR = 150.0
SINGLE_GAME_LOSS_ALERT_TOUR = 150.0


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
    x = s.fillna("").astype(str).str.replace("\u00a0", "", regex=False).str.strip()
    x = x.str.replace(",", ".", regex=False)
    x = x.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    return pd.to_numeric(x, errors="coerce")


def fmt_money(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NaN"
    return f"{v:.2f}"


def risk_decision(score: int) -> str:
    if score < T_APPROVE:
        return "APPROVE"
    if score < T_FAST_CHECK:
        return "FAST_CHECK"
    return "MANUAL_REVIEW"


def read_summary_excel(uploaded_xlsx) -> pd.DataFrame:
    if uploaded_xlsx is None:
        return pd.DataFrame()
    return pd.read_excel(io.BytesIO(uploaded_xlsx.getvalue()), engine="openpyxl", header=None)


GAME_ID_RE = re.compile(r"ID игры:\s*([0-9\-]+)", re.IGNORECASE)
PLAYER_ROW_ID_RE = re.compile(r"(?:^|;)\s*(\d{6,10})\s*;", re.IGNORECASE)

TOUR_HINT_RE = re.compile(r"\bPPST/|бай-ин:\s*|satellite|pko|mko\b", re.IGNORECASE)
RING_HINT_RE = re.compile(r"\bPPSR/|NLH\s+\d|\bPLO\b|Bomb Pot|Ante\b", re.IGNORECASE)


def classify_game_type(lines_in_block: list[str]) -> str:
    text = " ".join(lines_in_block)
    if TOUR_HINT_RE.search(text):
        return "TOURNAMENT"
    if RING_HINT_RE.search(text):
        return "RING"
    return "UNKNOWN"


def parse_games_csv(uploaded_file, known_player_ids: set[int]) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["game_id", "game_type", "player_id", "win", "fee"])

    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
    lines = text.splitlines()

    rows = []
    current_game_id = None
    current_block_lines = []
    header_cols = None

    def split_semicolon(line: str):
        return [c.strip().strip('"') for c in line.split(";")]

    for line in lines:
        m_game = GAME_ID_RE.search(line)
        if m_game:
            current_game_id = m_game.group(1).strip()
            current_block_lines = [line]
            header_cols = None
            continue

        if not current_game_id:
            continue

        current_block_lines.append(line)

        if "ID игрока" in line:
            header_cols = split_semicolon(line)
            continue

        if "Итог" in line:
            continue

        m_pid = PLAYER_ROW_ID_RE.search(line)
        if not m_pid:
            continue

        pid = int(m_pid.group(1))
        if pid not in known_player_ids:
            continue

        parts = split_semicolon(line)
        win = np.nan
        fee = np.nan

        if header_cols is not None and len(parts) == len(header_cols):
            if "Выигрыш" in header_cols:
                win = to_float(parts[header_cols.index("Выигрыш")])
            if "Комиссия" in header_cols:
                fee = to_float(parts[header_cols.index("Комиссия")])
        else:
            if len(parts) >= 7:
                win = to_float(parts[6])

            tail = parts[-6:] if len(parts) >= 6 else parts
            candidates = [to_float(t) for t in tail]
            candidates = [c for c in candidates if not np.isnan(c)]
            if candidates:
                pos = [c for c in candidates if c >= 0]
                fee = min(pos) if pos else np.nan

        gtype = classify_game_type(current_block_lines)
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
    if g.empty:
        return pd.DataFrame(columns=["session_id", "game_type", "players"])
    return g


def coplay_features(target_id: int, sessions_df: pd.DataFrame, game_type: str | None = None) -> dict:
    if sessions_df.empty or "players" not in sessions_df.columns:
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

    top1_share = top1 / sessions_count if sessions_count else 0.0
    top2_share = (top1 + top2) / sessions_count if sessions_count else 0.0

    return {
        "sessions_count": sessions_count,
        "unique_opponents": unique_opponents,
        "top1_coplay_share": float(top1_share),
        "top2_coplay_share": float(top2_share),
        "top_partners": partners[:8],
    }


def transfer_features(target_id: int, games_df: pd.DataFrame, game_type: str | None = None) -> dict:
    if games_df.empty:
        return {"target_games": 0, "target_total_win_from_games": 0.0, "top_sources": [], "top_source_net": 0.0, "top_source_share": 0.0, "single_game_extremes": []}

    df = games_df.copy()
    if game_type:
        df = df[df["game_type"] == game_type]

    df = df[pd.notna(df["win"])].copy()
    if df.empty:
        return {"target_games": 0, "target_total_win_from_games": 0.0, "top_sources": [], "top_source_net": 0.0, "top_source_share": 0.0, "single_game_extremes": []}

    t = df[df["player_id"] == target_id][["game_id", "win"]].copy()
    target_games = int(len(t))
    if target_games == 0:
        return {"target_games": 0, "target_total_win_from_games": 0.0, "top_sources": [], "top_source_net": 0.0, "top_source_share": 0.0, "single_game_extremes": []}

    transfer = {}
    extremes = []

    if game_type == "TOURNAMENT":
        win_alert, loss_alert = SINGLE_GAME_WIN_ALERT_TOUR, SINGLE_GAME_LOSS_ALERT_TOUR
    else:
        win_alert, loss_alert = SINGLE_GAME_WIN_ALERT_RING, SINGLE_GAME_LOSS_ALERT_RING

    for _, tr in t.iterrows():
        gid = tr["game_id"]
        t_win = float(tr["win"])
        game = df[df["game_id"] == gid][["player_id", "win"]]

        if t_win >= win_alert or t_win <= -loss_alert:
            extremes.append({"game_id": gid, "target_win": t_win})

        if t_win <= 0:
            continue

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
        "target_games": target_games,
        "target_total_win_from_games": float(t["win"].sum()),
        "top_sources": sources_sorted[:8],
        "top_source_net": top_source_net,
        "top_source_share": float(top_source_share),
        "single_game_extremes": extremes[:10],
    }


def pick_main_risk(decision: str, sb_reasons: list[str], coverage: dict) -> str:
    if sb_reasons:
        return sb_reasons[0].replace("RING:", "RING —").replace("TOURNAMENT:", "TOURNAMENT —")

    if coverage["ring_games_with_target"] == 0 and coverage["tour_games_with_target"] == 0 and coverage["unknown_games_with_target"] == 0:
        return "Недостаточно данных по играм: по файлу «Игры» игрок не найден."

    if decision == "APPROVE":
        return "Явных сигналов перелива не обнаружено."
    if decision == "FAST_CHECK":
        return "Есть слабые сигналы риска: рекомендуется повышенное внимание."
    return "Совокупность факторов даёт высокий риск: нужна ручная проверка СБ."


def structured_assessment(row, cop_ring, cop_tour, trf_ring, trf_tour, coverage):
    net_total = row["_net_total"]
    net_ring = row["_net_ring"]
    net_mtt = row["_net_mtt"]
    comm = row["_comm"]

    score = 0
    blocks = {
        "Ключевой вывод": [],
        "Покрытие данных": [],
        "Профиль игрока (Excel)": [],
        "Сеть/поведение (co-play)": [],
        "Перелив (transfer-proxy)": [],
        "Причины для ручной проверки СБ": [],
    }

    blocks["Покрытие данных"].append(
        f"RING игр с результатом: {coverage['ring_games_with_target']}; "
        f"TOURNAMENT игр с результатом: {coverage['tour_games_with_target']}; "
        f"UNKNOWN: {coverage['unknown_games_with_target']}."
    )

    if pd.notna(net_total):
        blocks["Профиль игрока (Excel)"].append(f"Общий выигрыш (I): {fmt_money(net_total)}.")
        if net_total > 0:
            score += 30
    else:
        blocks["Профиль игрока (Excel)"].append("Общий выигрыш (I): не распознан.")
        score += 10

    if pd.notna(comm):
        blocks["Профиль игрока (Excel)"].append(f"Комиссия (AB): {fmt_money(comm)}.")
    else:
        blocks["Профиль игрока (Excel)"].append("Комиссия (AB): не распознана.")
        score += 5

    if pd.notna(net_total) and net_total > 0 and pd.notna(net_mtt) and net_total != 0:
        mtt_share = float(min(1.0, max(0.0, net_mtt / net_total)))
        blocks["Профиль игрока (Excel)"].append(f"Доля MTT/SNG в профите (P/I): {mtt_share:.0%}.")
        if mtt_share >= 0.7:
            score -= 15
            blocks["Профиль игрока (Excel)"].append("Профит в основном MTT/SNG — риск ниже.")

    if pd.notna(net_total) and net_total > 0 and pd.notna(net_ring) and net_total != 0:
        ring_share = float(min(1.0, max(0.0, net_ring / net_total)))
        blocks["Профиль игрока (Excel)"].append(f"Доля Ring в профите (O/I): {ring_share:.0%}.")
        if ring_share >= 0.7:
            score += 15
            blocks["Профиль игрока (Excel)"].append("Профит в основном Ring — риск выше.")

    blocks["Сеть/поведение (co-play)"].append(
        f"RING: сессий={cop_ring['sessions_count']}, уникальных оппонентов={cop_ring['unique_opponents']}, "
        f"топ-1 доля={cop_ring['top1_coplay_share']:.0%}, топ-2 доля={cop_ring['top2_coplay_share']:.0%}."
    )
    blocks["Сеть/поведение (co-play)"].append(
        f"TOURNAMENT: сессий={cop_tour['sessions_count']}, уникальных оппонентов={cop_tour['unique_opponents']}, "
        f"топ-1 доля={cop_tour['top1_coplay_share']:.0%}."
    )

    if cop_ring["sessions_count"] >= MIN_SESSIONS_FOR_COPLAY:
        if cop_ring["top1_coplay_share"] >= COPLAY_TOP1_SHARE_SUSP:
            score += 30
            blocks["Причины для ручной проверки СБ"].append("RING: один партнёр встречается слишком часто.")
        if cop_ring["top2_coplay_share"] >= COPLAY_TOP2_SHARE_SUSP:
            score += 15
            blocks["Причины для ручной проверки СБ"].append("RING: узкий круг (топ-2 покрывает большую часть игр).")
        if cop_ring["unique_opponents"] <= 5:
            score += 10
            blocks["Причины для ручной проверки СБ"].append("RING: мало уникальных оппонентов.")
    elif cop_ring["sessions_count"] > 0:
        score += 5
        blocks["Сеть/поведение (co-play)"].append("RING: сессий мало — сигнал слабый, вывод осторожный.")
    else:
        blocks["Сеть/поведение (co-play)"].append("RING: данных по совместной игре мало/нет.")

    blocks["Перелив (transfer-proxy)"].append(
        f"RING: игр={trf_ring['target_games']}, суммарный результат={fmt_money(trf_ring['target_total_win_from_games'])}, "
        f"топ-источник={fmt_money(trf_ring['top_source_net'])}, доля={trf_ring['top_source_share']:.0%}."
    )
    blocks["Перелив (transfer-proxy)"].append(
        f"TOURNAMENT: игр={trf_tour['target_games']}, суммарный результат={fmt_money(trf_tour['target_total_win_from_games'])}, "
        f"топ-источник={fmt_money(trf_tour['top_source_net'])}, доля={trf_tour['top_source_share']:.0%}."
    )

    if trf_ring["top_sources"]:
        top_pid, top_amt = trf_ring["top_sources"][0]
        if top_amt >= PAIR_NET_TRANSFER_ALERT_RING:
            score += 30
            blocks["Причины для ручной проверки СБ"].append(
                f"RING: крупный поток фишек от одного ID (источник {top_pid} ≈ {fmt_money(top_amt)})."
            )
        if trf_ring["top_source_share"] >= PAIR_DOMINANCE_ALERT and top_amt >= (PAIR_NET_TRANSFER_ALERT_RING / 2):
            score += 20
            blocks["Причины для ручной проверки СБ"].append("RING: доминирование одного источника результата.")
        if trf_ring["single_game_extremes"]:
            score += 10
            blocks["Причины для ручной проверки СБ"].append("RING: аномально крупный результат в отдельной игре.")

    if trf_tour["top_sources"]:
        top_pid, top_amt = trf_tour["top_sources"][0]
        if top_amt >= PAIR_NET_TRANSFER_ALERT_TOUR:
            score += 10
            blocks["Причины для ручной проверки СБ"].append(
                f"TOURNAMENT: крупный поток от одного ID (источник {top_pid} ≈ {fmt_money(top_amt)})."
            )
        if trf_tour["single_game_extremes"]:
            score += 5
            blocks["Сеть/поведение (co-play)"].append("TOURNAMENT: есть аномальные результаты в отдельных турнирах.")

    if coverage["ring_games_with_target"] == 0 and coverage["tour_games_with_target"] == 0 and coverage["unknown_games_with_target"] == 0:
        score += 15
        blocks["Причины для ручной проверки СБ"].append("Нет данных по игроку в файле 'Игры' (нельзя подтвердить чистоту).")

    score = int(max(0, min(100, score)))
    decision = risk_decision(score)

    if decision == "APPROVE":
        blocks["Ключевой вывод"].append("Вывод можно разрешить: явных признаков перелива по текущим данным не выявлено.")
    elif decision == "FAST_CHECK":
        blocks["Ключевой вывод"].append("Нужна быстрая проверка: есть слабые/неустойчивые признаки риска.")
    else:
        blocks["Ключевой вывод"].append("Нужна ручная проверка СБ: повышенная вероятность перелива/сговора по текущим данным.")

    if decision == "MANUAL_REVIEW" and not blocks["Причины для ручной проверки СБ"]:
        blocks["Причины для ручной проверки СБ"].append("Общий риск высокий по совокупности факторов (см. детали).")

    main_risk = pick_main_risk(decision, blocks["Причины для ручной проверки СБ"], coverage)
    return score, decision, main_risk, blocks


def pill(text: str, kind: str):
    if kind == "ok":
        st.success(text)
    elif kind == "warn":
        st.warning(text)
    else:
        st.error(text)


def load_excel_period(uploaded_excel) -> pd.DataFrame:
    if uploaded_excel is None:
        return pd.DataFrame(columns=["_player_id", "_net_total", "_net_ring", "_net_mtt", "_comm"])

    x = read_summary_excel(uploaded_excel)
    if x is None or x.empty:
        return pd.DataFrame(columns=["_player_id", "_net_total", "_net_ring", "_net_mtt", "_comm"])

    x["_player_id"] = pd.to_numeric(x.iloc[:, IDX_ID], errors="coerce")
    x = x.dropna(subset=["_player_id"]).copy()
    x["_player_id"] = x["_player_id"].astype(int)

    x["_net_total"] = to_float_series(x.iloc[:, IDX_NET_TOTAL])
    x["_net_ring"] = to_float_series(x.iloc[:, IDX_NET_RING])
    x["_net_mtt"] = to_float_series(x.iloc[:, IDX_NET_MTT])
    x["_comm"] = to_float_series(x.iloc[:, IDX_COMM])

    return x[["_player_id", "_net_total", "_net_ring", "_net_mtt", "_comm"]].copy()


def merge_excels(ex1: pd.DataFrame, ex2: pd.DataFrame) -> pd.DataFrame:
    if ex1.empty and ex2.empty:
        return pd.DataFrame(columns=["_player_id", "_net_total", "_net_ring", "_net_mtt", "_comm"])

    both = pd.concat([ex1, ex2], ignore_index=True)
    merged = both.groupby("_player_id", as_index=False)[["_net_total", "_net_ring", "_net_mtt", "_comm"]].sum(min_count=1)
    return merged


# =========================
# SIDEBAR: GLOBAL LOAD UI
# =========================
KEYS_ALL = ["excel_w1.xlsx", "games_w1.csv", "excel_w2.xlsx", "games_w2.csv"]

with st.sidebar:
    st.header("Данные (общие для приглашённых пользователей)")

    # storage backend indicator
    if isinstance(storage, R2Backend) and storage.enabled():
        st.success("Хранилище: Cloudflare R2 (данные переживают reboot).")
    else:
        st.warning("Хранилище: Memory fallback (общий режим и устойчивость НЕ гарантированы).")

    uploaded_by = st.text_input("Оператор (кто загружает файлы)", value="", placeholder="например: SB / Admin / Ivan")

    st.subheader("Окно 1: прошлая неделя (обязательно)")
    excel_w1 = shared_uploader("Общая таблица (.xlsx) — прошлая неделя", "excel_w1.xlsx", ["xlsx"], uploaded_by)
    games_w1 = shared_uploader("Игры (.csv) — прошлая неделя", "games_w1.csv", ["csv", "txt"], uploaded_by)

    st.subheader("Окно 2: позапрошлая неделя (дополнительно)")
    excel_w2 = shared_uploader("Общая таблица (.xlsx) — позапрошлая неделя", "excel_w2.xlsx", ["xlsx"], uploaded_by)
    games_w2 = shared_uploader("Игры (.csv) — позапрошлая неделя", "games_w2.csv", ["csv", "txt"], uploaded_by)

    st.divider()
    if st.button("Удалить ВСЕ данные (2 окна)", type="secondary"):
        delete_all_shared_files(KEYS_ALL)
        st.rerun()

    show_debug = st.checkbox("Показать технические детали", value=False)


st.divider()

if excel_w1 is None or games_w1 is None:
    st.info("Нужно загрузить файлы окна 1: Excel + Игры.")
    st.stop()

if (excel_w2 is None) ^ (games_w2 is None):
    st.warning("Окно 2 загружено неполностью (есть только один файл). Для корректной суммы лучше загрузить оба.")

# =========================
# LOAD + MERGE 2 WINDOWS
# =========================
ex1 = load_excel_period(excel_w1)
ex2 = load_excel_period(excel_w2) if excel_w2 is not None else pd.DataFrame(columns=ex1.columns)
df = merge_excels(ex1, ex2)

known_ids = set(df["_player_id"].tolist())

g1 = parse_games_csv(games_w1, known_ids)
g2 = parse_games_csv(games_w2, known_ids) if games_w2 is not None else pd.DataFrame(columns=g1.columns)
games_df = pd.concat([g1, g2], ignore_index=True)

sessions_df = build_sessions_from_games(games_df)

# =========================
# TOP STATUS
# =========================
c1, c2, c3, c4 = st.columns(4, gap="small")
c1.metric("Игроков (сумма 2 недель)", f"{len(df)}", border=True)
c2.metric("Строк games (сумма 2 недель)", f"{len(games_df)}", border=True)
c3.metric("Сессий (ID игры ≥2 игрока)", f"{len(sessions_df)}", border=True)
if not games_df.empty and "game_type" in games_df.columns:
    dist = dict(games_df["game_type"].value_counts())
    c4.metric("Типы игр (R/T/U)", f"{dist.get('RING',0)}/{dist.get('TOURNAMENT',0)}/{dist.get('UNKNOWN',0)}", border=True)
else:
    c4.metric("Типы игр (R/T/U)", "0/0/0", border=True)

st.divider()

# =========================
# PLAYER CHECK UI
# =========================
left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("Проверка игрока")
    default_id = int(df["_player_id"].iloc[0]) if len(df) else 0
    target_id = st.number_input("ID игрока на вывод", min_value=0, value=default_id, step=1)
    run = st.button("Проверить", type="primary")
    st.caption("Аналитика по сумме окна 1 + окна 2 (если загружено).")

with right:
    st.subheader("Как читать решение")
    st.markdown(
        "- APPROVE — можно разрешить.\n"
        "- FAST_CHECK — быстрая проверка.\n"
        "- MANUAL_REVIEW — ручная проверка СБ."
    )

if not run:
    st.stop()

row_df = df[df["_player_id"] == int(target_id)]
if row_df.empty:
    st.error("ID игрока не найден в суммарной Excel (2 недели).")
    st.stop()

row = row_df.iloc[0]

target_games = games_df[games_df["player_id"] == int(target_id)] if not games_df.empty else pd.DataFrame()
coverage = {
    "ring_games_with_target": int((target_games["game_type"] == "RING").sum()) if not target_games.empty else 0,
    "tour_games_with_target": int((target_games["game_type"] == "TOURNAMENT").sum()) if not target_games.empty else 0,
    "unknown_games_with_target": int((target_games["game_type"] == "UNKNOWN").sum()) if not target_games.empty else 0,
}

cop_ring = coplay_features(int(target_id), sessions_df, "RING")
cop_tour = coplay_features(int(target_id), sessions_df, "TOURNAMENT")
trf_ring = transfer_features(int(target_id), games_df, "RING")
trf_tour = transfer_features(int(target_id), games_df, "TOURNAMENT")

score, decision, main_risk, blocks = structured_assessment(row, cop_ring, cop_tour, trf_ring, trf_tour, coverage)

st.divider()

# =========================
# RESULT
# =========================
h1, h2, h3 = st.columns([1.2, 1, 1], gap="small")
with h1:
    st.subheader("Итог")
h2.metric("Risk score", f"{score}/100", border=True)
h3.metric("Decision", decision, border=True)

if decision == "APPROVE":
    pill("ВЫВОД: РАЗРЕШИТЬ", "ok")
elif decision == "FAST_CHECK":
    pill("ВЫВОД: БЫСТРАЯ ПРОВЕРКА", "warn")
else:
    pill("ВЫВОД: РУЧНАЯ ПРОВЕРКА СБ", "bad")

tab1, tab2, tab3 = st.tabs(["Кратко", "Детали", "Пары / источники"])

with tab1:
    st.subheader("Главный риск")
    st.info(f"Главный риск: {main_risk}")

    st.subheader("Причины")
    if blocks["Причины для ручной проверки СБ"]:
        for x in blocks["Причины для ручной проверки СБ"][:12]:
            st.markdown(f"- {x}")
    else:
        st.markdown("- Явных причин для ручной проверки СБ не выявлено.")

    st.subheader("Покрытие данных")
    for x in blocks["Покрытие данных"]:
        st.markdown(f"- {x}")

with tab2:
    st.subheader("Excel")
    for x in blocks["Профиль игрока (Excel)"]:
        st.markdown(f"- {x}")

    st.subheader("Co-play")
    for x in blocks["Сеть/поведение (co-play)"]:
        st.markdown(f"- {x}")

    st.subheader("Transfer-proxy")
    for x in blocks["Перелив (transfer-proxy)"]:
        st.markdown(f"- {x}")

with tab3:
    st.subheader("Co-play партнёры (RING)")
    if cop_ring["top_partners"]:
        st.dataframe(pd.DataFrame(cop_ring["top_partners"], columns=["partner_id", "coplay_sessions"]), use_container_width=True)
    else:
        st.info("Нет данных по партнёрам в RING.")

    st.subheader("Источники transfer-proxy (RING)")
    if trf_ring["top_sources"]:
        st.dataframe(pd.DataFrame(trf_ring["top_sources"], columns=["source_player_id", "estimated_transfer_to_target"]), use_container_width=True)
    else:
        st.info("Нет выраженных источников в RING.")

    st.subheader("Источники transfer-proxy (TOURNAMENT)")
    if trf_tour["top_sources"]:
        st.dataframe(pd.DataFrame(trf_tour["top_sources"], columns=["source_player_id", "estimated_transfer_to_target"]), use_container_width=True)
    else:
        st.info("Нет выраженных источников в TOURNAMENT.")

if show_debug:
    with st.expander("Debug: данные", expanded=False):
        st.write("Storage backend:", storage.__class__.__name__)
        st.write("Meta:", storage.load_meta())
        st.dataframe(df.head(30), use_container_width=True)
        st.dataframe(games_df.head(30), use_container_width=True)
        st.dataframe(sessions_df.head(30), use_container_width=True)
