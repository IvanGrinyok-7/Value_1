import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import datetime
from collections import defaultdict

# ==========================================
# 1. КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ==========================================
st.set_page_config(page_title="PPPoker Anti-Fraud Analytics", layout="wide", page_icon="🛡️")

# Пороги срабатывания (Risk Thresholds)
RISK_HIGH_NET_FLOW_BB = 40.0      # Если выиграл у одного игрока > 40 ББ (чистыми)
RISK_HIGH_GROSS_FLOW_BB = 150.0   # Если оборот с одним игроком > 150 ББ
RISK_CONCENTRATION = 0.75         # Если > 75% выигрыша получено от одного оппонента
RISK_HU_SHARE = 0.80              # Если > 80% профита получено в HU ситуациях
RISK_LOW_RAKE_RATIO = 0.03        # Если комиссия < 3% от выигрыша (для Ring)

# Регулярки для парсинга
RE_GAME_ID = re.compile(r"ID игры:\s*([0-9\.\-eE]+(?:-[0-9]+)?)", re.IGNORECASE)
RE_TABLE_NAME = re.compile(r"Название стола:\s*(.+?)\s*$", re.IGNORECASE)
RE_STAKES = re.compile(r"(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)") # Находит 0.1/0.2 и т.д.

# ==========================================
# 2. ПАРСИНГ И ЗАГРУЗКА ДАННЫХ
# ==========================================

def clean_float(x):
    """Превращает строки с запятыми и пробелами в float."""
    if pd.isna(x) or x == "":
        return 0.0
    s = str(x).replace(",", ".").replace("\xa0", "").replace(" ", "").strip()
    try:
        return float(s)
    except:
        return 0.0

@st.cache_data(show_spinner=False)
def parse_games_file(uploaded_files):
    """
    Сложный парсер для специфического формата CSV/TXT из PPPoker.
    Преобразует вложенную структуру в плоскую таблицу.
    """
    all_rows = []
    
    for file in uploaded_files:
        # Читаем как байты, декодируем, чтобы не зависеть от BOM и кодировок
        content = file.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        
        current_game_id = None
        current_table = ""
        current_bb = 0.0
        current_game_type = "UNKNOWN" # RING / MTT / SNG
        current_date = None
        
        # Индексы колонок (динамический поиск)
        idx_map = {}
        
        for line in lines:
            line = line.strip()
            if not line: 
                continue
                
            # 1. Поиск ID игры (Начало блока)
            m_id = RE_GAME_ID.search(line)
            if m_id:
                current_game_id = m_id.group(1)
                
                # Ищем название стола в той же строке
                m_table = RE_TABLE_NAME.search(line)
                current_table = m_table.group(1) if m_table else "Unknown Table"
                
                # Сброс параметров блока
                current_bb = 0.0
                current_game_type = "UNKNOWN"
                idx_map = {}
                continue
                
            # 2. Поиск даты (если есть в блоке)
            if "Начало:" in line and "Окончание:" in line:
                # Можно парсить дату, если нужно для таймлайна
                pass

            # 3. Определение типа игры и ставок
            # Если строка содержит ставки типа "0.5/1", это Ring
            m_stakes = RE_STAKES.search(line)
            if m_stakes and "Бай-ин:" not in line: # Исключаем турниры где бай-ин может быть похож
                try:
                    current_bb = float(m_stakes.group(2).replace(",", "."))
                    current_game_type = "RING"
                except:
                    pass
            
            if "PPST" in line or "Бай-ин:" in line or "Гарант." in line:
                current_game_type = "TOURNAMENT"

            # 4. Поиск заголовка таблицы игроков
            # Строка вида: ;ID игрока;Ник;...
            if "ID игрока" in line:
                parts = [p.strip().strip('"') for p in line.split(";")]
                # Создаем карту индексов, так как порядок может меняться
                for i, col in enumerate(parts):
                    if "ID игрока" in col: idx_map['id'] = i
                    elif "Ник" in col: idx_map['nick'] = i
                    elif "Выигрыш" in col: idx_map['win'] = i
                    elif "Комиссия" in col: idx_map['rake'] = i
                    elif "Бай-ин" in col and "PP" in col: idx_map['buyin'] = i
                continue

            # 5. Парсинг строки игрока
            # Строка данных начинается с ; (пустой первый элемент при split)
            if current_game_id and idx_map and "Итог" not in line:
                parts = [p.strip().strip('"') for p in line.split(";")]
                
                # Проверка: строка должна содержать данные (длина больше макс индекса)
                max_idx = max(idx_map.values()) if idx_map else 0
                if len(parts) <= max_idx:
                    continue
                
                try:
                    p_id_str = parts[idx_map['id']]
                    if not p_id_str.isdigit(): continue # Пропуск мусорных строк
                    
                    p_id = int(p_id_str)
                    p_nick = parts[idx_map.get('nick', -1)] if 'nick' in idx_map else ""
                    p_win = clean_float(parts[idx_map.get('win', -1)])
                    p_rake = clean_float(parts[idx_map.get('rake', -1)])
                    
                    all_rows.append({
                        'game_id': current_game_id,
                        'table_name': current_table,
                        'game_type': current_game_type,
                        'bb': current_bb,
                        'player_id': p_id,
                        'nick': p_nick,
                        'win': p_win,
                        'rake': p_rake
                    })
                except Exception:
                    continue

    return pd.DataFrame(all_rows)

@st.cache_data(show_spinner=False)
def load_general_files(uploaded_files):
    """Загружает файлы 'Общее.csv'."""
    dfs = []
    for f in uploaded_files:
        try:
            if f.name.endswith('.xlsx'):
                df = pd.read_excel(f)
            else:
                # Авто-детект разделителя
                content = f.getvalue()
                try:
                    df = pd.read_csv(io.BytesIO(content), sep=';', encoding='utf-8')
                except:
                    df = pd.read_csv(io.BytesIO(content), sep=',', encoding='utf-8')
            dfs.append(df)
        except Exception as e:
            st.error(f"Ошибка чтения файла {f.name}: {e}")
            
    if not dfs:
        return pd.DataFrame()
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Нормализация имен колонок (убираем лишние пробелы)
    full_df.columns = [c.strip() for c in full_df.columns]
    
    # Ключевые колонки, которые нам нужны
    target_cols = ['ID игрока', 'Ник', 'Общий выигрыш игроков + События', 'Выигрыш игрока Ring Game', 'Выигрыш игрока MTT, SNG']
    
    # Проверка наличия колонок
    available_cols = [c for c in target_cols if c in full_df.columns]
    
    df_clean = full_df[available_cols].copy()
    df_clean['ID игрока'] = pd.to_numeric(df_clean['ID игрока'], errors='coerce').fillna(0).astype(int)
    
    # Конвертация денег
    money_cols = [c for c in available_cols if c != 'ID игрока' and c != 'Ник']
    for c in money_cols:
        df_clean[c] = df_clean[c].apply(clean_float)
        
    # Агрегация по ID (если игрок встречался в нескольких неделях)
    df_grouped = df_clean.groupby('ID игрока').agg({
        'Ник': 'first',
        'Общий выигрыш игроков + События': 'sum',
        'Выигрыш игрока Ring Game': 'sum',
        'Выигрыш игрока MTT, SNG': 'sum'
    }).reset_index()
    
    return df_grouped

# ==========================================
# 3. АНАЛИТИЧЕСКИЙ ДВИЖОК
# ==========================================

def calculate_flows(games_df):
    """
    Вычисляет, кто кому проиграл деньги (Net Flow).
    Логика: За конкретным столом (session) сумма выигрышей > 0 распределяется
    между проигравшими пропорционально их проигрышу.
    """
    if games_df.empty:
        return pd.DataFrame()

    flows = []
    
    # Группируем по уникальной игре
    sessions = games_df.groupby('game_id')
    
    for g_id, group in sessions:
        # Разделяем на победителей и проигравших
        winners = group[group['win'] > 0]
        losers = group[group['win'] < 0]
        
        if winners.empty or losers.empty:
            continue
            
        total_win = winners['win'].sum()
        total_loss = abs(losers['win'].sum())
        
        # Если дисбаланс (из-за комиссии или ошибок логов), нормализуем по меньшему
        # Но для оценки перелива нам важно, кто сколько *отдал*
        
        # Матрица перелива в этой сессии
        for _, l_row in losers.iterrows():
            loss_amt = abs(l_row['win'])
            l_id = l_row['player_id']
            
            for _, w_row in winners.iterrows():
                w_amt = w_row['win']
                w_id = w_row['player_id']
                
                # Доля выигрыша этого победителя в общем пуле победителей
                share = w_amt / total_win
                
                # Предполагаемая сумма, перетекшая от Лузера к Победителю
                transfer = loss_amt * share
                
                flows.append({
                    'from': l_id,
                    'to': w_id,
                    'amount': transfer,
                    'game_type': l_row['game_type'],
                    'bb': l_row['bb'],
                    'game_id': g_id
                })
                
    return pd.DataFrame(flows)

def analyze_player(player_id, general_df, games_df, flows_df):
    """
    Главная функция анализа риска для конкретного игрока.
    """
    report = {
        "status": "GREEN",
        "reasons": [],
        "metrics": {},
        "top_donors": [],
        "games_stats": {}
    }
    
    # 1. Данные из "Общее"
    p_general = general_df[general_df['ID игрока'] == player_id]
    if p_general.empty:
        report["metrics"]["total_profit"] = 0.0
        report["metrics"]["nick"] = "Unknown"
    else:
        report["metrics"]["total_profit"] = p_general['Общий выигрыш игроков + События'].sum()
        report["metrics"]["nick"] = p_general['Ник'].iloc[0]
        report["metrics"]["ring_profit"] = p_general['Выигрыш игрока Ring Game'].sum()

    # Если игрок в минусе или около нуля, риск минимален (обычно проверяем вывод)
    if report["metrics"]["total_profit"] < 5:
        return report # Green

    # 2. Анализ Игр (Ring Games)
    p_games = games_df[games_df['player_id'] == player_id]
    
    if p_games.empty:
        report["status"] = "YELLOW"
        report["reasons"].append("Нет детальной истории игр, но есть профит в 'Общем'. Требуется ручная проверка источника.")
        return report

    total_win_games = p_games['win'].sum()
    total_rake = p_games['rake'].sum()
    
    # Эффективность перелива (Rake Check)
    # Если выиграл много, а комиссии заплатил мало -> подозрительно
    rake_ratio = total_rake / total_win_games if total_win_games > 0 else 0
    report["metrics"]["rake_ratio"] = rake_ratio
    
    if total_win_games > 100 and rake_ratio < RISK_LOW_RAKE_RATIO and p_general['Выигрыш игрока Ring Game'].sum() > 0:
        report["reasons"].append(f"🔴 Аномально низкая комиссия ({rake_ratio:.1%}). Возможно, играли мало рук с крупными банками (Dump).")
        report["status"] = "RED"

    # 3. Анализ Потоков (Flows) - КТО ДОНОР?
    if not flows_df.empty:
        # Деньги пришедшие Игроку
        inflow = flows_df[flows_df['to'] == player_id].copy()
        
        if not inflow.empty:
            # Агрегация по донорам
            donors = inflow.groupby('from').agg({
                'amount': 'sum',
                'bb': 'mean', # средний блайнд
                'game_id': 'nunique' # количество совместных игр
            }).reset_index().sort_values('amount', ascending=False)
            
            total_received = donors['amount'].sum()
            
            # Топ 1 донор
            top_donor = donors.iloc[0]
            top_donor_share = top_donor['amount'] / total_received if total_received > 0 else 0
            
            report["metrics"]["top_donor_id"] = int(top_donor['from'])
            report["metrics"]["top_donor_amt"] = top_donor['amount']
            report["metrics"]["concentration"] = top_donor_share
            
            # Перевод в ББ (приблизительно)
            avg_bb = top_donor['bb'] if top_donor['bb'] > 0 else 1
            amount_in_bb = top_donor['amount'] / avg_bb
            
            # Логика детекта
            if top_donor_share > RISK_CONCENTRATION and report["metrics"]["total_profit"] > 50:
                report["reasons"].append(f"🔴 Высокая концентрация: {top_donor_share:.0%} выигрыша получено от одного игрока (ID {int(top_donor['from'])}).")
                report["status"] = "RED"
                
            if amount_in_bb > RISK_HIGH_NET_FLOW_BB:
                report["reasons"].append(f"🔴 Крупный чистый выигрыш у одного игрока: {amount_in_bb:.0f} BB (>{RISK_HIGH_NET_FLOW_BB} BB).")
                if report["status"] != "RED": report["status"] = "RED" # Усиление до красного
                
            # Добавляем инфо в отчет
            for _, row in donors.head(3).iterrows():
                report["top_donors"].append({
                    "id": int(row['from']),
                    "amount": row['amount'],
                    "games": int(row['game_id'])
                })

    # 4. Анализ HU (Heads Up)
    # Считаем, сколько денег выиграно, когда за столом (в файле) было только 2 человека
    # Примечание: парсер группирует по game_id. Если там 2 записи - это HU.
    session_sizes = games_df.groupby('game_id').size()
    hu_game_ids = session_sizes[session_sizes == 2].index
    
    hu_wins = p_games[p_games['game_id'].isin(hu_game_ids)]['win'].sum()
    hu_share = hu_wins / total_win_games if total_win_games > 0 else 0
    
    report["metrics"]["hu_share"] = hu_share
    
    if hu_share > RISK_HU_SHARE and total_win_games > 50:
        report["reasons"].append(f"🟠 {hu_share:.0%} выигрыша получено в Heads-Up (игра 1 на 1).")
        if report["status"] == "GREEN": report["status"] = "YELLOW"

    return report

# ==========================================
# 4. ИНТЕРФЕЙС STREAMLIT
# ==========================================

st.title("🕵️‍♂️ PPPoker Security Check")
st.markdown("**Инструмент для выявления перелива фишек (Chip Dumping)**")

with st.expander("ℹ️ Инструкция (развернуть)"):
    st.markdown("""
    1. Загрузите файлы **Общее.csv/xlsx** (можно несколько за разные недели).
    2. Загрузите файлы **Игры.csv** (выгрузка истории рук).
    3. Введите **ID игрока**.
    4. Система проанализирует:
       - Источники денег (кто проиграл этому игроку).
       - Концентрацию выигрыша (все деньги от одного человека?).
       - Странности в комиссии и типах игры.
    """)

# --- Блок загрузки ---
col_u1, col_u2 = st.columns(2)
with col_u1:
    files_general = st.file_uploader("📂 1. Загрузить 'ОБЩЕЕ' (недели)", accept_multiple_files=True, type=['csv', 'xlsx'])
with col_u2:
    files_games = st.file_uploader("📂 2. Загрузить 'ИГРЫ' (детализация)", accept_multiple_files=True, type=['csv', 'txt'])

# --- Обработка данных ---
if files_general and files_games:
    with st.spinner("Анализ базы данных..."):
        df_general = load_general_files(files_general)
        df_games_raw = parse_games_file(files_games)
        
        # Кэшируем расчет потоков для всей базы (это самая тяжелая операция)
        if not df_games_raw.empty:
            df_flows = calculate_flows(df_games_raw)
        else:
            df_flows = pd.DataFrame()
            
    st.success(f"Загружено: {len(df_general)} записей профилей и {len(df_games_raw)} игровых сессий.")
    st.divider()

    # --- Блок проверки ---
    col_input, col_res = st.columns([1, 2])
    
    with col_input:
        st.subheader("Проверка игрока")
        target_id = st.number_input("Введите ID игрока", min_value=0, value=0, step=1)
        btn_check = st.button("🔍 Анализировать", type="primary")

    if btn_check and target_id > 0:
        report = analyze_player(target_id, df_general, df_games_raw, df_flows)
        
        with col_res:
            # Карточка вердикта
            if report["status"] == "RED":
                st.error(f"⛔ ВЕРДИКТ: ВЫСОКИЙ РИСК (ПЕРЕЛИВ)")
                st.markdown("**Действие:** Блокировка вывода. Передача в СБ для ручного разбора раздач.")
            elif report["status"] == "YELLOW":
                st.warning(f"⚠️ ВЕРДИКТ: ПОДОЗРИТЕЛЬНО")
                st.markdown("**Действие:** Запросить проверку раздач. Возможен 'мягкий' перелив или бамхант.")
            else:
                st.success(f"✅ ВЕРДИКТ: ЧИСТО")
                st.markdown("**Действие:** Можно проводить вывод.")

            # Причины
            if report["reasons"]:
                st.write("---")
                st.subheader("Обнаруженные паттерны:")
                for reason in report["reasons"]:
                    st.write(reason)
            
            # Детали
            st.write("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Никнейм", report["metrics"].get("nick", "N/A"))
            c1.metric("Общий профит", f"{report['metrics']['total_profit']:.2f}")
            
            c2.metric("Комиссия % (Rake)", f"{report['metrics'].get('rake_ratio', 0):.1%}", help="Норма > 5%. Меньше 3% - признак перелива.")
            c2.metric("HU доля (1 на 1)", f"{report['metrics'].get('hu_share', 0):.1%}", help="Если > 80% профита сделано 1 на 1, это подозрительно.")
            
            top_conc = report['metrics'].get('concentration', 0)
            c3.metric("Концентрация", f"{top_conc:.1%}", help="Какая часть денег пришла от ТОП-1 донора.")

            # Таблица доноров
            if report["top_donors"]:
                st.write("---")
                st.markdown("#### 💸 От кого получены деньги (Топ-3):")
                donors_df = pd.DataFrame(report["top_donors"])
                donors_df.columns = ["ID Донора", "Сумма (перелито)", "Кол-во игр"]
                st.dataframe(donors_df, hide_index=True)

            # Генерация текста для менеджера
            st.write("---")
            with st.expander("📋 Текст отчета для копирования"):
                res_text = f"Проверка ID: {target_id}\nСтатус: {report['status']}\nПрофит: {report['metrics']['total_profit']:.2f}\n"
                if report["reasons"]:
                    res_text += "Причины риска:\n" + "\n".join(report["reasons"])
                else:
                    res_text += "Подозрительных активностей по метрикам не найдено."
                st.code(res_text)

elif files_general or files_games:
    st.info("Пожалуйста, загрузите оба типа файлов ('Общее' и 'Игры') для корректного анализа.")
else:
    st.info("Ожидание загрузки файлов...")
