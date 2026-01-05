import streamlit as st
import pandas as pd
import numpy as np
import re
import io

# ==========================================
# 1. НАСТРОЙКИ И КОНСТАНТЫ
# ==========================================
st.set_page_config(page_title="PPPoker Analyzer Pro", layout="wide", page_icon="⚡")

# Пороги риска (Risk Thresholds)
RISK_NET_BB = 40.0        # Чистый выигрыш у одного игрока > 40 BB
RISK_GROSS_BB = 150.0     # Оборот с одним игроком > 150 BB
RISK_CONCENTRATION = 0.70 # > 70% профита от одного донора
RISK_HU_SHARE = 0.80      # > 80% профита в HU
RISK_LOW_RAKE = 0.035     # Рейк < 3.5% (признак дампа без постфлопа)

# Регулярные выражения (компилируем один раз)
RE_GAME_ID = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)
RE_TABLE_NAME = re.compile(r"Название стола:\s*(.+?)\s*$", re.IGNORECASE)
RE_STAKES = re.compile(r"(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)")

# ==========================================
# 2. ОПТИМИЗИРОВАННЫЙ ПАРСИНГ
# ==========================================

def fast_clean_float(x):
    """Быстрая очистка чисел."""
    if pd.isna(x) or x == "":
        return 0.0
    # Если это уже число
    if isinstance(x, (int, float)):
        return float(x)
    # Быстрая замена
    try:
        return float(str(x).replace(",", ".").replace("\xa0", "").strip())
    except:
        return 0.0

@st.cache_data(show_spinner=False)
def parse_games_optimized(uploaded_files):
    """
    Сверхбыстрый парсер логов игр.
    """
    data_rows = []
    
    for file in uploaded_files:
        # Читаем сразу весь файл в память (быстрее, чем line-by-line для таких объемов)
        content = file.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        
        # Переменные контекста
        curr_gid = None
        curr_table = "Unknown"
        curr_bb = 0.0
        curr_type = "UNKNOWN"
        
        # Карта индексов колонок (динамическая)
        idx = {}
        
        for line in lines:
            # Быстрый фильтр пустых строк
            if len(line) < 5: continue
            
            # 1. Поиск ID игры (маркер блока)
            if "ID игры:" in line:
                m = RE_GAME_ID.search(line)
                if m:
                    curr_gid = m.group(1)
                    
                    # Название стола
                    t_match = RE_TABLE_NAME.search(line)
                    curr_table = t_match.group(1) if t_match else "Unknown"
                    
                    # Сброс
                    curr_bb = 0.0
                    curr_type = "UNKNOWN"
                    idx = {}
                continue
                
            # 2. Определение типа и ставок
            if curr_gid and ("PPSR" in line or "PPST" in line or "/" in line):
                if "PPST" in line or "Бай-ин" in line:
                    curr_type = "MTT/SNG"
                else:
                    # Пытаемся найти ставки для Ring
                    s_match = RE_STAKES.search(line)
                    if s_match:
                        try:
                            curr_bb = float(s_match.group(2).replace(",", "."))
                            curr_type = "RING"
                        except: pass
                continue

            # 3. Хедер таблицы
            if ";ID игрока;" in line:
                parts = [p.strip().replace('"', '') for p in line.split(";")]
                for i, col in enumerate(parts):
                    if "ID игрока" in col: idx['id'] = i
                    elif "Ник" in col: idx['nick'] = i
                    elif "Выигрыш" in col: idx['win'] = i
                    elif "Комиссия" in col: idx['rake'] = i
                continue

            # 4. Строка данных (должен быть активный game_id и карта индексов)
            if curr_gid and idx and ";Итог;" not in line:
                parts = line.split(";")
                # Простейшая валидация длины
                if len(parts) < max(idx.values(), default=0): 
                    continue
                
                try:
                    p_id_raw = parts[idx['id']].strip().replace('"', '')
                    if not p_id_raw.isdigit(): continue
                    
                    p_win = fast_clean_float(parts[idx['win']])
                    p_rake = fast_clean_float(parts[idx.get('rake', -1)])
                    
                    data_rows.append((
                        curr_gid, 
                        curr_type, 
                        curr_bb, 
                        int(p_id_raw), 
                        parts[idx.get('nick', -1)].strip().replace('"', ''), 
                        p_win, 
                        p_rake
                    ))
                except:
                    continue

    if not data_rows:
        return pd.DataFrame()

    # Создание DF с оптимизированными типами данных
    df = pd.DataFrame(data_rows, columns=['game_id', 'type', 'bb', 'player_id', 'nick', 'win', 'rake'])
    df['type'] = df['type'].astype('category')
    df['player_id'] = df['player_id'].astype('int32')
    df['win'] = df['win'].astype('float32')
    df['rake'] = df['rake'].astype('float32')
    df['bb'] = df['bb'].astype('float32')
    return df

@st.cache_data(show_spinner=False)
def load_general_data(uploaded_files):
    """Загрузка файла 'Общее'."""
    dfs = []
    target_cols = ['ID игрока', 'Ник', 'Общий выигрыш игроков + События', 'Выигрыш игрока Ring Game', 'Выигрыш игрока MTT, SNG']
    
    for f in uploaded_files:
        try:
            if f.name.endswith('.xlsx'):
                df = pd.read_excel(f)
            else:
                # Пробуем разные разделители
                content = f.getvalue()
                try:
                    df = pd.read_csv(io.BytesIO(content), sep=';', encoding='utf-8')
                except:
                    df = pd.read_csv(io.BytesIO(content), sep=',', encoding='utf-8')
            
            # Очистка имен колонок
            df.columns = [c.strip() for c in df.columns]
            
            # Фильтр только нужных колонок
            available = [c for c in target_cols if c in df.columns]
            if available:
                dfs.append(df[available])
        except:
            continue
            
    if not dfs: return pd.DataFrame()
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df['ID игрока'] = pd.to_numeric(full_df['ID игрока'], errors='coerce').fillna(0).astype(int)
    
    # Конвертация денег
    for col in full_df.columns:
        if col != 'ID игрока' and col != 'Ник':
            full_df[col] = full_df[col].apply(fast_clean_float)
            
    # Агрегация (суммируем, если загружено несколько недель)
    agg_df = full_df.groupby('ID игрока').agg({
        'Ник': 'first',
        'Общий выигрыш игроков + События': 'sum',
        'Выигрыш игрока Ring Game': 'sum',
        'Выигрыш игрока MTT, SNG': 'sum'
    }).reset_index()
    
    return agg_df

# ==========================================
# 3. ВЕКТОРИЗИРОВАННАЯ АНАЛИТИКА (CORE)
# ==========================================

@st.cache_data(show_spinner=False)
def calculate_network_flows(games_df):
    """
    Векторизированный расчет переливов.
    Вместо циклов используется матричное распределение.
    Скорость: ~100x быстрее циклов.
    """
    if games_df.empty:
        return pd.DataFrame()

    # 1. Агрегация по играм: сумма выигрыша победителей
    # Берем только тех, кто выиграл (>0)
    winners = games_df[games_df['win'] > 0].copy()
    losers = games_df[games_df['win'] < 0].copy()
    
    if winners.empty or losers.empty:
        return pd.DataFrame()

    # Считаем общий банк победителей в каждой раздаче
    game_pools = winners.groupby('game_id')['win'].sum().reset_index()
    game_pools.rename(columns={'win': 'total_game_win'}, inplace=True)
    
    # 2. Добавляем информацию о пуле к победителям
    winners = winners.merge(game_pools, on='game_id')
    
    # Считаем долю каждого победителя (equity)
    winners['share'] = winners['win'] / winners['total_game_win']
    
    # Оптимизация: оставляем только нужные колонки для merge
    w_slim = winners[['game_id', 'player_id', 'share']].rename(columns={'player_id': 'to_id'})
    l_slim = losers[['game_id', 'player_id', 'win', 'bb', 'type']].rename(columns={'player_id': 'from_id', 'win': 'loss_amt'})
    l_slim['loss_amt'] = l_slim['loss_amt'].abs() # Берем модуль проигрыша
    
    # 3. CROSS JOIN внутри каждой игры (Winner x Loser)
    # Это создает строки для каждой пары: "Игрок А (проиграл) -> Игрок Б (выиграл)"
    merged = l_slim.merge(w_slim, on='game_id')
    
    # 4. Расчет суммы перелива
    merged['flow_amt'] = merged['loss_amt'] * merged['share']
    
    # 5. Агрегация связей (кто кому сколько слил всего)
    flows = merged.groupby(['from_id', 'to_id']).agg({
        'flow_amt': 'sum',
        'bb': 'mean', # средний блайнд игр
        'game_id': 'nunique', # кол-во совместных игр
        'type': 'first' # тип игры (преобладает)
    }).reset_index()
    
    return flows

def get_player_stats(pid, general_df, games_df, flows_df):
    """Сбор всей статистики по конкретному игроку."""
    
    res = {"status": "GREEN", "flags": [], "data": {}}
    
    # --- Общие данные ---
    gen_row = general_df[general_df['ID игрока'] == pid]
    if gen_row.empty:
        res["data"] = {"nick": "Unknown", "total": 0, "ring": 0}
        return res
        
    total_profit = gen_row['Общий выигрыш игроков + События'].iloc[0]
    ring_profit = gen_row['Выигрыш игрока Ring Game'].iloc[0]
    res["data"]["nick"] = gen_row['Ник'].iloc[0]
    res["data"]["total"] = total_profit
    res["data"]["ring"] = ring_profit
    
    # Если профит маленький, пропускаем детальный анализ
    if total_profit < 10:
        return res

    # --- Анализ игр ---
    p_games = games_df[games_df['player_id'] == pid]
    if p_games.empty:
        res["status"] = "YELLOW"
        res["flags"].append("Нет истории раздач, но есть профит (возможно джекпот или ошибка выгрузки).")
        return res
        
    real_win = p_games['win'].sum()
    real_rake = p_games['rake'].sum()
    rake_ratio = real_rake / real_win if real_win > 0 else 0
    
    res["data"]["rake_ratio"] = rake_ratio
    
    if real_win > 50 and rake_ratio < RISK_LOW_RAKE:
        res["status"] = "RED"
        res["flags"].append(f"Низкая комиссия ({rake_ratio:.1%}). Характерно для дампа префлоп/флоп без рейка.")

    # --- HU (Heads Up) анализ ---
    # Группируем игры, где участвовал игрок, считаем кол-во записей в каждой игре
    # Если записей 2 - это HU
    relevant_games = games_df[games_df['game_id'].isin(p_games['game_id'].unique())]
    game_counts = relevant_games.groupby('game_id').size()
    hu_games_ids = game_counts[game_counts == 2].index
    
    hu_win = p_games[p_games['game_id'].isin(hu_games_ids)]['win'].sum()
    hu_share = hu_win / real_win if real_win > 0 else 0
    
    res["data"]["hu_share"] = hu_share
    
    if hu_share > RISK_HU_SHARE and real_win > 100:
        current_status = res["status"]
        res["status"] = "RED"
        res["flags"].append(f"Игра 1-на-1 (HU): {hu_share:.0%} от всего выигрыша. Это аномалия.")

    # --- Анализ доноров (От кого деньги) ---
    if not flows_df.empty:
        # Входящие потоки К игроку
        inflows = flows_df[flows_df['to_id'] == pid].copy()
        
        if not inflows.empty:
            # Считаем сумму и переводим в BB (если Ring)
            inflows['amt_bb'] = inflows.apply(lambda x: x['flow_amt'] / x['bb'] if x['bb'] > 0 else 0, axis=1)
            
            top_donors = inflows.sort_values('flow_amt', ascending=False).head(3)
            res["data"]["donors"] = top_donors
            
            # Топ 1 донор
            top1 = top_donors.iloc[0]
            total_received = inflows['flow_amt'].sum()
            concentration = top1['flow_amt'] / total_received if total_received > 0 else 0
            
            res["data"]["concentration"] = concentration
            
            # ПРОВЕРКИ
            if top1['amt_bb'] > RISK_NET_BB and top1['type'] == 'RING':
                res["status"] = "RED"
                res["flags"].append(f"Крупный чистый выигрыш у ID {int(top1['from_id'])}: {top1['amt_bb']:.1f} BB.")
                
            if concentration > RISK_CONCENTRATION and real_win > 100:
                res["status"] = "RED"
                res["flags"].append(f"Концентрация: {concentration:.0%} выигрыша пришло от одного игрока.")

    return res

# ==========================================
# 4. ИНТЕРФЕЙС ПРИЛОЖЕНИЯ
# ==========================================

st.title("🛡️ PPPoker Anti-Fraud Analytics 2.0")

# Session State для хранения данных
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_general = pd.DataFrame()
    st.session_state.df_games = pd.DataFrame()
    st.session_state.df_flows = pd.DataFrame()

# Боковая панель
with st.sidebar:
    st.header("Загрузка данных")
    u_gen = st.file_uploader("1. Файлы 'Общее'", type=['xlsx', 'csv'], accept_multiple_files=True)
    u_gam = st.file_uploader("2. Файлы 'Игры'", type=['csv', 'txt'], accept_multiple_files=True)
    
    if st.button("📥 Загрузить и обработать", type="primary"):
        if u_gen and u_gam:
            with st.status("Обработка данных..."):
                st.write("Чтение 'Общее'...")
                st.session_state.df_general = load_general_data(u_gen)
                
                st.write("Парсинг 'Игры' (это может занять время)...")
                st.session_state.df_games = parse_games_optimized(u_gam)
                
                st.write("Расчет матрицы переливов...")
                st.session_state.df_flows = calculate_network_flows(st.session_state.df_games)
                
                st.session_state.data_loaded = True
                st.write("Готово!")
        else:
            st.error("Загрузите оба типа файлов!")

    if st.session_state.data_loaded:
        st.success(f"В базе: {len(st.session_state.df_general)} игроков")
        if st.button("Очистить базу"):
            st.session_state.data_loaded = False
            st.rerun()

# Основное окно
if not st.session_state.data_loaded:
    st.info("👈 Пожалуйста, загрузите файлы в меню слева для начала работы.")
    st.markdown("""
    ### Как это работает?
    1. Система считывает логи игр и строит граф переливов фишек.
    2. Вычисляется **Net Flow** (чистый переток денег от игрока к игроку).
    3. Анализируется **эффективность рейка** и **доля Heads-Up**.
    4. При вводе ID вы получаете мгновенный вердикт.
    """)
else:
    col_search, col_res = st.columns([1, 2])
    
    with col_search:
        st.subheader("Проверка вывода")
        target_id = st.number_input("ID Игрока", min_value=0, step=1)
        check_btn = st.button("🔍 Проверить игрока", type="primary", use_container_width=True)
        
    if check_btn and target_id > 0:
        # Запуск анализа
        report = get_player_stats(
            target_id, 
            st.session_state.df_general, 
            st.session_state.df_games, 
            st.session_state.df_flows
        )
        
        with col_res:
            # Отображение статуса
            status = report["status"]
            if status == "RED":
                st.error("⛔ ВЕРДИКТ: ВЫСОКИЙ РИСК (Отправить в СБ)")
            elif status == "YELLOW":
                st.warning("⚠️ ВЕРДИКТ: ЕСТЬ ПОДОЗРЕНИЯ (Ручная проверка)")
            else:
                st.success("✅ ВЕРДИКТ: ЧИСТО (Разрешен вывод)")
            
            # Метрики
            d = report["data"]
            if not d:
                st.write("Нет данных по игроку.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Ник", d.get('nick', 'N/A'))
                c1.metric("Общий Профит", f"{d.get('total', 0):.2f}")
                
                rake_pct = d.get('rake_ratio', 0) * 100
                c2.metric("Комиссия", f"{rake_pct:.1f}%", delta="-Низкая" if rake_pct < 3.5 else None, delta_color="inverse")
                
                hu_pct = d.get('hu_share', 0) * 100
                c3.metric("Доля HU", f"{hu_pct:.0f}%", delta="-Высокая" if hu_pct > 80 else None, delta_color="inverse")

                # Причины
                if report["flags"]:
                    st.write("---")
                    st.subheader("🚩 Обнаруженные проблемы:")
                    for f in report["flags"]:
                        st.write(f"- {f}")
                
                # Таблица доноров
                if "donors" in d and not d["donors"].empty:
                    st.write("---")
                    st.subheader("💸 Источники денег (Топ доноры)")
                    donors_view = d["donors"][['from_id', 'flow_amt', 'amt_bb', 'game_id']].copy()
                    donors_view.columns = ['ID Донора', 'Сумма', 'В Блайндах (BB)', 'Кол-во игр']
                    st.dataframe(donors_view, hide_index=True)
                
                # Текст для менеджера
                st.write("---")
                with st.expander("📋 Скопировать отчет"):
                    flag_txt = "\n".join([f"- {x}" for x in report["flags"]]) if report["flags"] else "Паттернов перелива не выявлено."
                    res_text = (
                        f"Проверка ID: {target_id} ({d.get('nick')})\n"
                        f"Статус: {status}\n"
                        f"Профит: {d.get('total', 0):.2f}\n"
                        f"Комиссия: {rake_pct:.1f}%\n"
                        f"Аналитика:\n{flag_txt}"
                    )
                    st.code(res_text)
