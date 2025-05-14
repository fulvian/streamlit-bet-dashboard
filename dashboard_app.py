import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import locale
import re # Importa il modulo per le espressioni regolari

# --- Impostazioni Iniziali ---
st.set_page_config(layout="wide")
st.title("📊 Dashboard Analisi Scommesse - Versione Avanzata")

# --- Gestione Localizzazione con Fallback ---
italian_locale_set = False
try:
    locale.setlocale(locale.LC_ALL, 'it_IT.UTF-8')
    italian_locale_set = True
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'Italian_Italy.1252')
        italian_locale_set = True
    except locale.Error:
        st.warning("Localizzazione italiana ('it_IT.UTF-8' o 'Italian_Italy.1252') non disponibile sul sistema. Verrà usata una formattazione valuta di fallback (€ X.XX).")

# --- Palette Colori Tenui ---
color_win = '#8FBC8F'
color_loss = '#CD5C5C'
color_line1 = '#B0C4DE'
color_line2 = '#FFEC8B'
color_tbd = '#D3D3D3'
color_unknown = '#A9A9A9'
color_map_esiti = {'Win': color_win, 'Loss': color_loss, 'TBD': color_tbd, 'Unknown': color_unknown}

# --- Funzioni di Pulizia Dati ---
def clean_numeric_value(value_str):
    """Converte un valore in float in modo robusto."""
    if pd.isna(value_str): return np.nan
    if isinstance(value_str, (int, float)): return float(value_str)
    try:
        cleaned_str = str(value_str).strip().replace('?', '')
        if cleaned_str == '': return np.nan
        try: return float(cleaned_str)
        except ValueError: return float(cleaned_str.replace(',', '.'))
    except (ValueError, TypeError): return np.nan

# --- Funzione Parsing Date più robusta ---
# Dizionario per mappare mesi italiani a inglesi
italian_to_english_month = {
    'Gen': 'Jan', 'Feb': 'Feb', 'Mar': 'Mar', 'Apr': 'Apr',
    'Mag': 'May', 'Giu': 'Jun', 'Lug': 'Jul', 'Ago': 'Aug',
    'Set': 'Sep', 'Ott': 'Oct', 'Nov': 'Nov', 'Dic': 'Dec'
}

def parse_italian_date(date_str):
    """
    Converte una stringa di data (potenzialmente in formato italiano 'gg Mmm AAAA')
    in un oggetto datetime in modo robusto, anche senza locale italiano.
    """
    if pd.isna(date_str):
        return pd.NaT

    cleaned_date_str = str(date_str).strip()

    # Tenta di sostituire le abbreviazioni dei mesi italiani con quelle inglesi
    # Itera sul dizionario
    for ita, eng in italian_to_english_month.items():
        # Cerca l'abbreviazione italiana (ignorando maiuscole/minuscole) nella stringa
        if ita.lower() in cleaned_date_str.lower():
            try:
                # Trova l'occorrenza esatta (es. "Mag") per sostituirla correttamente con "May"
                # re.IGNORECASE fa la ricerca case-insensitive
                match = re.search(ita, cleaned_date_str, re.IGNORECASE)
                if match:
                    # Sostituisce l'occorrenza trovata con l'abbreviazione inglese
                    cleaned_date_str = cleaned_date_str.replace(match.group(0), eng)
                    break # Esce dal ciclo dopo la prima sostituzione trovata
            except Exception:
                # Fallback se la regex fallisce (improbabile ma sicuro)
                # Prova a sostituire diverse varianti di case
                cleaned_date_str = cleaned_date_str.replace(ita, eng).replace(ita.lower(), eng.lower()).replace(ita.title(), eng.title())
                break # Esce comunque dal ciclo

    # Ora prova a fare il parsing con il formato atteso (usando abbreviazioni inglesi)
    try:
        # errors='raise' per forzare l'errore se il formato non corrisponde
        return pd.to_datetime(cleaned_date_str, format='%d %b %Y', errors='raise')
    except ValueError:
        # Se il formato specifico fallisce ancora (es. data in formato diverso),
        # prova a far indovinare a Pandas
        # errors='coerce' restituisce NaT se non riesce a indovinare
        return pd.to_datetime(cleaned_date_str, errors='coerce')

def calculate_pl(row, stake_col='Stake', quota_col='Quota', esito_col='Esito_Standard'):
    """Calcola il Profit/Loss (P/L) per una riga."""
    if stake_col not in row or quota_col not in row or esito_col not in row: return 0.0
    esito = row[esito_col]; quota = row[quota_col]; stake = row[stake_col]
    if esito == 'Win' and pd.notna(quota) and quota > 0 and pd.notna(stake) and stake > 0:
        return round((stake * quota) - stake, 2)
    elif esito == 'Loss' and pd.notna(stake) and stake > 0:
        return round(-stake, 2)
    else: return 0.0

def format_currency(value):
    """Formatta un valore come valuta con fallback."""
    if pd.isna(value): return "N/D"
    if italian_locale_set:
        try: return locale.currency(value, grouping=True)
        except ValueError: return f"{value:.2f} €" # Fallback ulteriore
    else: return f"{value:.2f} €" # Fallback principale

def calculate_expected_value(probability, odds):
    """Calcola l'Expected Value di una scommessa."""
    if pd.isna(probability) or pd.isna(odds): return np.nan
    return (probability * (odds - 1)) - (1 - probability)

def calculate_kelly_stake(probability, odds, fraction=1.0):
    """Calcola lo stake ottimale secondo il criterio di Kelly."""
    if pd.isna(probability) or pd.isna(odds): return np.nan
    if odds <= 1: return 0.0
    kelly = (probability - (1 - probability) / (odds - 1))
    return max(0, kelly * fraction)

# --- Caricamento e Pulizia Dati ---
file_path = "risultati_bet.csv"
colonna_sq_a = "Squadra A"; colonna_sq_b = "Squadra B"; colonna_data = "Data"
colonna_ora = "Ora"; colonna_quota = "Quota"; colonna_stake = "Stake"
colonna_prob = "Probabilita Stimata"; colonna_edge = "Edge Value"; colonna_esito = "Esito"
colonna_media_pt_stimati = "Media Punti Stimati"; colonna_confidenza = "Confidenza"; colonna_ris_finale = "Punteggio finale"

try:
    df = pd.read_csv(file_path, skipinitialspace=True)
    st.success(f"File '{file_path}' caricato con successo.")

    # --- Gestione Colonne ---
    colonne_essenziali = [colonna_sq_a, colonna_sq_b, colonna_data, colonna_quota, colonna_stake, colonna_esito]
    colonne_opzionali = [colonna_ora, colonna_prob, colonna_edge, colonna_media_pt_stimati, colonna_confidenza, colonna_ris_finale]
    colonne_essenziali_mancanti = [col for col in colonne_essenziali if col not in df.columns]
    if colonne_essenziali_mancanti:
        st.error(f"Errore CRITICO: Colonne essenziali non trovate: {colonne_essenziali_mancanti}.")
        st.info(f"Colonne trovate: {df.columns.tolist()}")
        st.stop()
    colonne_opzionali_mancanti = [col for col in colonne_opzionali if col not in df.columns]
    if colonne_opzionali_mancanti:
        st.warning(f"Attenzione: Colonne opzionali non trovate: {colonne_opzionali_mancanti}.")
        for col in colonne_opzionali_mancanti: df[col] = np.nan

    df_original = df.copy()
    original_rows = len(df)

    # --- Pulizia Iniziale ---
    df[colonna_esito] = df[colonna_esito].replace(r'^\s*$', np.nan, regex=True)
    rows_before_dropna = len(df)
    df.dropna(subset=[colonna_esito], inplace=True)
    rows_after_dropna = len(df)
    if rows_before_dropna > rows_after_dropna:
        st.info(f"Rimosse {rows_before_dropna - rows_after_dropna} righe causa Esito mancante.")

    df_cleaned = df.copy()

    # --- Pulizia Specifica per Colonna ---
    df_cleaned[colonna_data] = df_cleaned[colonna_data].apply(parse_italian_date) # Usa la nuova funzione
    # --- Verifica Date Non Valide (Opzionale, per Debug) ---
    invalid_dates_count = df_cleaned[colonna_data].isna().sum()
    if invalid_dates_count > 0:
        st.warning(f"Attenzione: {invalid_dates_count} righe hanno una data non valida dopo il parsing e potrebbero essere escluse dalle analisi temporali.")
        # Potresti voler vedere quali righe sono:
        # st.dataframe(df_cleaned[df_cleaned[colonna_data].isna()])

    cols_to_clean_numeric = [colonna_quota, colonna_stake, colonna_prob, colonna_edge, colonna_media_pt_stimati, colonna_confidenza, colonna_ris_finale]
    for col in cols_to_clean_numeric:
        if col in df_cleaned.columns: df_cleaned[col] = df_cleaned[col].apply(clean_numeric_value)
    for col in [colonna_prob, colonna_edge]:
         if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
              if df_cleaned[col].max(skipna=True) > 1: df_cleaned[col] = df_cleaned[col] / 100.0

    esito_map = {'W': 'Win', 'L': 'Loss', 'TBD': 'TBD'}
    df_cleaned['Esito_Standard'] = df_cleaned[colonna_esito].astype(str).str.strip().str.upper().map(esito_map).fillna('Unknown')

    # --- Calcoli Derivati ---
    df_cleaned['P/L'] = df_cleaned.apply(lambda row: calculate_pl(row, stake_col=colonna_stake, quota_col=colonna_quota, esito_col='Esito_Standard'), axis=1)

    # Sovrastima Punti
    df_cleaned['Errore_Sovrastima_PT'] = np.nan
    if colonna_media_pt_stimati in df_cleaned.columns and colonna_ris_finale in df_cleaned.columns:
        valid_error_calc = df_cleaned[colonna_media_pt_stimati].notna() & df_cleaned[colonna_ris_finale].notna()
        diff = df_cleaned.loc[valid_error_calc, colonna_media_pt_stimati] - df_cleaned.loc[valid_error_calc, colonna_ris_finale]
        df_cleaned.loc[valid_error_calc, 'Errore_Sovrastima_PT'] = diff.clip(lower=0)

    # Nuovi calcoli per analisi avanzata
    if colonna_prob in df_cleaned.columns and colonna_quota in df_cleaned.columns:
        # Expected Value
        df_cleaned['EV'] = df_cleaned.apply(lambda row: calculate_expected_value(row[colonna_prob], row[colonna_quota]), axis=1)

        # Break-even quota
        df_cleaned['Quota_BE'] = df_cleaned[colonna_prob].apply(lambda x: 1/x if pd.notna(x) and x > 0 else np.nan)

        # Kelly stake suggerito (percentuale del bankroll)
        df_cleaned['Kelly_Stake'] = df_cleaned.apply(lambda row: calculate_kelly_stake(row[colonna_prob], row[colonna_quota]), axis=1)

    # --- Preparazione Dati Globali (pre-filtro sidebar) ---
    # Filtra per date valide PRIMA di calcolare drawdown e cumulativo globale
    df_cleaned_valid_dates = df_cleaned[df_cleaned[colonna_data].notna()].copy()

    df_results_global = df_cleaned_valid_dates[df_cleaned_valid_dates['Esito_Standard'].isin(['Win', 'Loss'])].copy()
    df_results_global = df_results_global.sort_values(by=colonna_data)
    if not df_results_global.empty:
        df_results_global['Cumulative P/L'] = df_results_global['P/L'].fillna(0).cumsum()
        running_max = df_results_global['Cumulative P/L'].cummax()
        absolute_drawdown = running_max - df_results_global['Cumulative P/L']
        max_drawdown_global = absolute_drawdown.max()
    else:
        df_results_global['Cumulative P/L'] = np.nan; max_drawdown_global = np.nan

    # --- Inizio App Streamlit ---
    st.sidebar.header("Filtri")
    unique_outcomes = df_cleaned['Esito_Standard'].unique() # Usa df_cleaned per mostrare tutti gli esiti possibili
    options_outcomes = unique_outcomes
    selected_outcomes = st.sidebar.multiselect("Filtra per Esito", options=options_outcomes, default=list(options_outcomes))
    start_date, end_date = None, None
    # Usa df_cleaned_valid_dates per determinare il range del filtro data
    if colonna_data in df_cleaned_valid_dates.columns and not df_cleaned_valid_dates.empty:
        min_date_dt = df_cleaned_valid_dates[colonna_data].min()
        max_date_dt = df_cleaned_valid_dates[colonna_data].max()
        if pd.NaT not in [min_date_dt, max_date_dt] and min_date_dt.date() != max_date_dt.date():
            min_date = min_date_dt.date()
            max_date = max_date_dt.date()
            selected_date_range = st.sidebar.date_input(f"Filtra per {colonna_data}", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if len(selected_date_range) == 2: start_date, end_date = selected_date_range
            elif len(selected_date_range) == 1: start_date = end_date = selected_date_range[0]
        elif pd.NaT not in [min_date_dt, max_date_dt]: # Caso di una sola data o NaT
            single_date = min_date_dt.date()
            st.sidebar.info(f"Tutti i dati validi sono relativi al {single_date.strftime('%d/%m/%Y')}")
            start_date = end_date = single_date
        else: # Caso in cui tutte le date sono NaT o il dataframe è vuoto
             st.sidebar.info("Nessuna data valida disponibile per il filtro.")


    # --- Applicazione Filtri ---
    # Inizia dai dati con date valide
    df_filtered_final = df_cleaned_valid_dates.copy()
    df_results_filtered_final = df_results_global.copy() # Contiene già solo W/L con date valide

    if selected_outcomes:
        df_filtered_final = df_filtered_final[df_filtered_final['Esito_Standard'].isin(selected_outcomes)]
        df_results_filtered_final = df_results_filtered_final[df_results_filtered_final['Esito_Standard'].isin(selected_outcomes)]
    if start_date and end_date:
        start_date_dt = pd.to_datetime(start_date); end_date_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        df_filtered_final = df_filtered_final[(df_filtered_final[colonna_data] >= start_date_dt) & (df_filtered_final[colonna_data] < end_date_dt)]
        df_results_filtered_final = df_results_filtered_final[(df_results_filtered_final[colonna_data] >= start_date_dt) & (df_results_filtered_final[colonna_data] < end_date_dt)]

    if not df_results_filtered_final.empty:
         df_results_filtered_final = df_results_filtered_final.sort_values(by=colonna_data)
         df_results_filtered_final['Cumulative P/L Filtered'] = df_results_filtered_final['P/L'].fillna(0).cumsum()
    else: df_results_filtered_final['Cumulative P/L Filtered'] = np.nan

    # --- Visualizzazione Dashboard ---
    if df_filtered_final.empty:
        st.warning("Nessun dato corrisponde ai filtri selezionati.")
    else:
        # --- Metriche Chiave ---
        st.subheader("Metriche Chiave (Periodo Filtrato)")
        total_bets_wl = 0; total_stake_wl = 0; total_pl_wl = 0; roi_wl = 0; win_rate_wl = 0
        avg_stake_wl = np.nan; most_winning_quota_data = None;
        avg_overestimation_error = np.nan
        sharpe_ratio_daily = np.nan; var_95_daily = np.nan; avg_ev = np.nan; avg_be_quota = np.nan

        if not df_results_filtered_final.empty:
            total_bets_wl = len(df_results_filtered_final)
            total_stake_wl = df_results_filtered_final[colonna_stake].fillna(0).sum()
            total_pl_wl = df_results_filtered_final['P/L'].fillna(0).sum()
            if total_bets_wl > 0:
                win_count_wl = (df_results_filtered_final['Esito_Standard'] == 'Win').sum()
                win_rate_wl = win_count_wl / total_bets_wl
                avg_stake_wl = df_results_filtered_final[colonna_stake].mean(skipna=True)
                roi_wl = (total_pl_wl / total_stake_wl) * 100 if total_stake_wl != 0 else 0
                if win_count_wl > 0:
                    winning_odds_counts = df_results_filtered_final[(df_results_filtered_final['Esito_Standard'] == 'Win') & df_results_filtered_final[colonna_quota].notna()][colonna_quota].value_counts()
                    if not winning_odds_counts.empty:
                        quota_val = winning_odds_counts.idxmax(); wins_at_quota = winning_odds_counts.max()
                        total_bets_at_quota = len(df_results_filtered_final[df_results_filtered_final[colonna_quota] == quota_val])
                        win_perc_at_quota = (wins_at_quota / total_bets_at_quota) * 100 if total_bets_at_quota > 0 else 0
                        win_perc_of_total_wins = (wins_at_quota / win_count_wl) * 100
                        most_winning_quota_data = {"quota": quota_val, "wins": wins_at_quota, "total_bets": total_bets_at_quota, "win_perc": win_perc_at_quota, "perc_of_total": win_perc_of_total_wins}

                # Calcolo Expected Value medio e quota break-even media
                if 'EV' in df_results_filtered_final.columns:
                    avg_ev = df_results_filtered_final['EV'].mean(skipna=True)
                if 'Quota_BE' in df_results_filtered_final.columns:
                    avg_be_quota = df_results_filtered_final['Quota_BE'].mean(skipna=True)

                # ---- INIZIO MODIFICA: Calcolo Deviazione Standard Differenza Punteggi per Partite Perse ----
                std_dev_score_diff_loss = np.nan # Inizializza a NaN
                # Assicurati che le colonne necessarie esistano nel dataframe filtrato
                if not df_results_filtered_final.empty and \
                   colonna_ris_finale in df_results_filtered_final.columns and \
                   colonna_media_pt_stimati in df_results_filtered_final.columns:

                    # Filtra per partite perse (Loss) e dove i punteggi sono disponibili
                    # Usiamo .copy() per evitare SettingWithCopyWarning quando aggiungiamo la nuova colonna
                    df_loss_scores_for_std = df_results_filtered_final[
                        (df_results_filtered_final['Esito_Standard'] == 'Loss') &
                        (df_results_filtered_final[colonna_ris_finale].notna()) &
                        (df_results_filtered_final[colonna_media_pt_stimati].notna())
                    ].copy()

                    if not df_loss_scores_for_std.empty:
                        # Calcola la differenza: Risultato Finale - Media Punti Stimati
                        # Una differenza positiva significa che il risultato finale è stato SUPERIORE alla stima
                        # Una differenza negativa significa che il risultato finale è stato INFERIORE alla stima
                        df_loss_scores_for_std['Differenza_Punteggio_Loss'] = df_loss_scores_for_std[colonna_ris_finale] - df_loss_scores_for_std[colonna_media_pt_stimati]

                        # Calcola la deviazione standard della differenza
                        # ddof=1 per la deviazione standard campionaria
                        if len(df_loss_scores_for_std['Differenza_Punteggio_Loss']) >= 2: # La deviazione standard richiede almeno 2 punti dati
                            std_dev_score_diff_loss = df_loss_scores_for_std['Differenza_Punteggio_Loss'].std(ddof=1)
                        elif len(df_loss_scores_for_std['Differenza_Punteggio_Loss']) == 1:
                             std_dev_score_diff_loss = 0.0 # Se c'è solo un punto dati, la deviazione standard è 0
                        # Se 'Differenza_Punteggio_Loss' è vuota (ma df_loss_scores_for_std non lo era), std_dev_score_diff_loss rimane np.nan
                # ---- FINE MODIFICA: Calcolo Deviazione Standard ----

                # INIZIO MODIFICA: Calcolo Dev. Std. Diff. Punti (Win)
                std_dev_score_diff_win = np.nan
                if not df_results_filtered_final.empty and \
                   colonna_ris_finale in df_results_filtered_final.columns and \
                   colonna_media_pt_stimati in df_results_filtered_final.columns:

                    df_win_scores_for_std = df_results_filtered_final[
                        (df_results_filtered_final['Esito_Standard'] == 'Win') &
                        (df_results_filtered_final[colonna_ris_finale].notna()) &
                        (df_results_filtered_final[colonna_media_pt_stimati].notna())
                    ].copy()

                    if not df_win_scores_for_std.empty:
                        df_win_scores_for_std['Differenza_Punteggio_Win'] = df_win_scores_for_std[colonna_ris_finale] - df_win_scores_for_std[colonna_media_pt_stimati]

                        if len(df_win_scores_for_std['Differenza_Punteggio_Win']) >= 2:
                            std_dev_score_diff_win = df_win_scores_for_std['Differenza_Punteggio_Win'].std(ddof=1)
                        elif len(df_win_scores_for_std['Differenza_Punteggio_Win']) == 1:
                            std_dev_score_diff_win = 0.0
                # FINE MODIFICA

                # ---- Calcolo Sharpe Ratio Avanzato ----
                # Ottieni tutti i giorni nel periodo (inclusi quelli senza scommesse)
                if start_date and end_date:
                    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                elif not df_results_filtered_final.empty and colonna_data in df_results_filtered_final and df_results_filtered_final[colonna_data].notna().any():
                    min_date_val = df_results_filtered_final[colonna_data].min().date()
                    max_date_val = df_results_filtered_final[colonna_data].max().date()
                    full_date_range = pd.date_range(start=min_date_val, end=max_date_val, freq='D')
                else:
                    full_date_range = pd.Index([]) # Indice vuoto se non ci sono date


                # Crea un dataframe con tutti i giorni e riempi con 0 i giorni senza scommesse
                if not full_date_range.empty:
                    daily_pl = df_results_filtered_final.groupby(pd.Grouper(key=colonna_data, freq='D'))['P/L'].sum()
                    all_days_pl = pd.Series(0, index=full_date_range, dtype=float) # Specificare dtype
                    all_days_pl.update(daily_pl)
                else:
                    all_days_pl = pd.Series(dtype=float)


                # Calcola i rendimenti giornalieri come percentuale del bankroll corrente
                if not df_results_filtered_final.empty and not all_days_pl.empty:
                    # Assumi un bankroll iniziale pari a 20 volte lo stake medio
                    avg_stake_val = df_results_filtered_final[colonna_stake].mean() # Rinomina variabile locale
                    initial_bankroll = avg_stake_val * 20 if pd.notna(avg_stake_val) and avg_stake_val > 0 else 1000


                    # Calcola il rendimento percentuale invece dell'assoluto
                    daily_returns_pct = all_days_pl / initial_bankroll

                    if len(daily_returns_pct) > 1:
                        # Metriche base
                        mean_daily_return = daily_returns_pct.mean()
                        std_dev_daily_return = daily_returns_pct.std(ddof=1)  # Correzione gradi di libertà

                        # Risk-free rate (assunto 0 per le scommesse, ma può essere personalizzato)
                        risk_free_daily = 0.0

                        # Sharpe Ratio
                        if std_dev_daily_return > 0 and pd.notna(std_dev_daily_return):
                            sharpe_ratio_daily = (mean_daily_return - risk_free_daily) / std_dev_daily_return
                            # Annualizzazione (opzionale)
                            sharpe_ratio_annualized = sharpe_ratio_daily * np.sqrt(365)
                        else:
                            sharpe_ratio_daily = np.nan
                            sharpe_ratio_annualized = np.nan

                        # Value at Risk (VaR) al 95%
                        var_95_daily = daily_returns_pct.quantile(0.05) * initial_bankroll

                        # Sortino Ratio (solo downside risk)
                        downside_returns = daily_returns_pct[daily_returns_pct < 0]
                        if not downside_returns.empty and downside_returns.notna().any(): # Aggiunto controllo notna()
                            # Usa la radice della media dei quadrati negativi
                            downside_deviation = np.sqrt(np.mean(downside_returns[downside_returns.notna()]**2)) # Filtrato NaNs
                            sortino_ratio = (mean_daily_return - risk_free_daily) / downside_deviation if downside_deviation > 0 else np.nan
                            # Annualizzazione (opzionale)
                            sortino_ratio_annualized = sortino_ratio * np.sqrt(365) if pd.notna(sortino_ratio) else np.nan
                        else:
                            sortino_ratio = np.nan
                            sortino_ratio_annualized = np.nan
                    else:
                        sharpe_ratio_daily = np.nan
                        sharpe_ratio_annualized = np.nan
                        var_95_daily = np.nan
                        sortino_ratio = np.nan
                        sortino_ratio_annualized = np.nan
                else:
                    sharpe_ratio_daily = np.nan
                    sharpe_ratio_annualized = np.nan
                    var_95_daily = np.nan
                    sortino_ratio = np.nan
                    sortino_ratio_annualized = np.nan

        if 'Errore_Sovrastima_PT' in df_filtered_final.columns:
             overestimations = df_filtered_final[df_filtered_final['Errore_Sovrastima_PT'] > 0]['Errore_Sovrastima_PT']
             if not overestimations.empty:
                 avg_overestimation_error = overestimations.mean()

        # --- Visualizzazione Metriche (Usa format_currency) ---
        col_header = st.columns([3, 1])
        with col_header[0]:
            st.markdown("##### Metriche Base")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Scommesse Concluse (W/L)", f"{total_bets_wl}")
        col2.metric(f"{colonna_stake} Totale (W/L)", format_currency(total_stake_wl))
        col3.metric(f"P/L Totale (W/L)", format_currency(total_pl_wl))

        # MODIFICA: Sostituisco Sovrastima Media PT con Dev. Std. Diff. Punti (Win)
        col4.metric("Dev. Std. Diff. Punti (Win)",
                    f"{std_dev_score_diff_win:.2f}" if pd.notna(std_dev_score_diff_win) else "N/D",
                    help=f"Deviazione standard della differenza ({colonna_ris_finale} - {colonna_media_pt_stimati}) per le sole partite vinte (W) con dati di punteggio disponibili.")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("ROI (W/L)", f"{roi_wl:.2f}%" if pd.notna(roi_wl) else "N/D")
        col6.metric("Win Rate (W/L)", f"{win_rate_wl:.1%}" if pd.notna(win_rate_wl) else "N/D")
        col7.metric(f"{colonna_stake} Medio (W/L)", format_currency(avg_stake_wl))
        # --- INIZIO MODIFICA: Visualizzazione Nuova Metrica ---
        col8.metric("Dev. Std. Diff. Punti (Loss)",
                    f"{std_dev_score_diff_loss:.2f}" if pd.notna(std_dev_score_diff_loss) else "N/D",
                    help=f"Deviazione standard della differenza ({colonna_ris_finale} - {colonna_media_pt_stimati}) per le sole partite perse (L) con dati di punteggio disponibili.")
        # --- FINE MODIFICA: Visualizzazione Nuova Metrica ---

        # Nuove metriche strategiche
        with col_header[0]: # Questo with era già presente e corretto
            st.markdown("##### Metriche Strategiche Avanzate")

        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        adv_col1.metric("Expected Value Medio", f"{avg_ev:.2%}" if pd.notna(avg_ev) else "N/D",
                        help="Media dell'Expected Value (EV): (probabilità * (quota - 1)) - (1 - probabilità)")
        adv_col2.metric("Quota Break-Even Media", f"{avg_be_quota:.2f}" if pd.notna(avg_be_quota) else "N/D",
                        help="Quota media di break-even (1/probabilità). Se la quota reale è maggiore, c'è valore atteso positivo.")

        # Kelly Criterion
        if not df_results_filtered_final.empty and win_rate_wl is not None and pd.notna(win_rate_wl):
            avg_odds_wins = df_results_filtered_final[df_results_filtered_final['Esito_Standard'] == 'Win'][colonna_quota].mean()
            if pd.notna(avg_odds_wins) and avg_odds_wins > 1: # Aggiunto controllo avg_odds_wins > 1 per evitare divisione per zero
                kelly_fraction = ((avg_odds_wins - 1) * win_rate_wl - (1 - win_rate_wl)) / (avg_odds_wins - 1)
                kelly_percentage = max(0, kelly_fraction) * 100  # Clip negativo a 0
                adv_col3.metric("Kelly Criterion (%)", f"{kelly_percentage:.2f}%",
                             help="Percentuale ottimale del bankroll da puntare secondo il criterio di Kelly")
            else:
                adv_col3.metric("Kelly Criterion (%)", "N/D", help="Non calcolabile con i dati attuali (es. quota media vincite <= 1).")


        # Analisi delle streaks (serie consecutive)
        if not df_results_filtered_final.empty:
            df_results_filtered_final['Streak_Group'] = (df_results_filtered_final['Esito_Standard'] !=
                                                      df_results_filtered_final['Esito_Standard'].shift(1)).cumsum()

            # Trova la serie vincente e perdente più lunga
            win_streaks = df_results_filtered_final[df_results_filtered_final['Esito_Standard'] == 'Win'].groupby('Streak_Group').size()
            loss_streaks = df_results_filtered_final[df_results_filtered_final['Esito_Standard'] == 'Loss'].groupby('Streak_Group').size()

            max_win_streak = win_streaks.max() if not win_streaks.empty else 0
            max_loss_streak = loss_streaks.max() if not loss_streaks.empty else 0

            adv_col4.metric("Max Serie Vincente", f"{max_win_streak}", help="Massimo numero di scommesse vinte consecutivamente")

        if most_winning_quota_data:
            st.markdown("---")
            with st.container(border=True):
                 st.markdown(f"#### Quota Maggiormente Vincente: {most_winning_quota_data['quota']:.2f}")
                 q_col1, q_col2, q_col3 = st.columns(3)
                 q_col1.metric(label="Vittorie a q. Quota", value=f"{most_winning_quota_data['wins']}")
                 q_col2.metric(label="Su Tot. Giocate a Quota", value=f"{most_winning_quota_data['total_bets']} ({most_winning_quota_data['win_perc']:.1f}%)")
                 q_col3.metric(label="% sul Totale Vittorie", value=f"{most_winning_quota_data['perc_of_total']:.1f}%")

        st.markdown("---")
        st.subheader("Indicatori Rischio/Performance")

        # Definisci funzione helper per valutare e formattare le metriche
        def metric_evaluation(value, ranges, na_text="N/D"):
            """
            Valuta una metrica secondo i range specificati e restituisce testo HTML formattato.
            ranges: lista di tuple (valore_limite, descrizione, colore)
            """
            if pd.isna(value):
                return na_text

            # Trova la valutazione corretta
            evaluation = None
            # Assicurati che ranges sia ordinato per valore_limite ascendente
            sorted_ranges = sorted(ranges, key=lambda x: x[0])
            for limit, desc, color in sorted_ranges:
                if value <= limit:
                    evaluation = (desc, color)
                    break

            # Se nessun range corrisponde (valore > ultimo limite), usa l'ultimo se ha limite float('inf') o il precedente
            if evaluation is None and sorted_ranges:
                if sorted_ranges[-1][0] == float('inf'):
                     evaluation = (sorted_ranges[-1][1], sorted_ranges[-1][2])
                # elif len(sorted_ranges) > 1 and value > sorted_ranges[-1][0]: # Se valore è maggiore dell'ultimo limite finito
                #     evaluation = (sorted_ranges[-1][1], sorted_ranges[-1][2]) # Usa l'ultimo range finito


            # Formatta il valore e l'etichetta
            if evaluation:
                return f"{value:.2f} <span style='color:{evaluation[1]}; font-weight:bold;'>({evaluation[0]})</span>"
            else: # Fallback se nessuna valutazione trovata (dovrebbe essere raro con una buona definizione dei range)
                return f"{value:.2f}"


        # Definisci funzione helper per il tooltip delle metriche
        def create_metric_help(description, ranges, interpretation=""):
            """Crea un tooltip formattato per le metriche."""
            tooltip = f"{description}\n\nValori interpretativi:"
            sorted_ranges = sorted(ranges, key=lambda x: x[0]) # Ordina per coerenza
            for i, (limit, desc, _) in enumerate(sorted_ranges):
                if i == 0: # Primo range
                    prefix = f"≤ {limit:.2f}"
                else: # Range successivi
                    prev_limit = sorted_ranges[i-1][0]
                    if limit == float('inf'):
                        prefix = f"> {prev_limit:.2f}"
                    else:
                        prefix = f"{prev_limit:.2f} < X ≤ {limit:.2f}"

                tooltip += f"\n- {desc}: {prefix}"

            if interpretation:
                tooltip += f"\n\n{interpretation}"
            return tooltip


        with st.container(border=True):
            r_col1, r_col2 = st.columns(2)

            # Max Drawdown (come percentuale del bankroll)
            # Calcola bankroll simulato per drawdown (es. basato su 5x stake totale, o su storico P/L)
            # Usiamo il picco del P/L cumulativo globale come riferimento per il bankroll massimo raggiunto
            simulated_peak_bankroll_for_dd = df_results_global['Cumulative P/L'].max() if not df_results_global.empty and 'Cumulative P/L' in df_results_global and df_results_global['Cumulative P/L'].notna().any() else (total_stake_wl * 5 if total_stake_wl > 0 else 1)
            if pd.isna(simulated_peak_bankroll_for_dd) or simulated_peak_bankroll_for_dd <=0 : simulated_peak_bankroll_for_dd = 1 # Evita divisione per zero

            max_dd_perc = (max_drawdown_global / simulated_peak_bankroll_for_dd) * 100 if pd.notna(max_drawdown_global) and simulated_peak_bankroll_for_dd > 0 else np.nan

            drawdown_ranges = [ # Valori limite INCLUSIVI (<=)
                (5, "Eccellente", "#00CC00"),      # <= 5%
                (10, "Buono", "#88CC00"),         # > 5% e <= 10%
                (20, "Accettabile", "#CCCC00"),   # > 10% e <= 20%
                (30, "Problematico", "#CC8800"),  # > 20% e <= 30%
                (float('inf'), "Critico", "#CC0000") # > 30%
            ]
            dd_help = create_metric_help(
                "Massima perdita storica dal picco precedente (calcolata su tutti i dati). Percentuale rispetto al picco massimo del P/L cumulativo.",
                drawdown_ranges,
                "Un drawdown basso indica un sistema stabile con perdite controllate."
            )
            max_drawdown_display_perc = f" ({max_dd_perc:.1f}% del picco P/L)" if pd.notna(max_dd_perc) else ""
            r_col1.metric(
                "Max Drawdown Storico",
                f"{format_currency(max_drawdown_global)}{max_drawdown_display_perc}",
                help=dd_help
            )


            # Sharpe Ratio
            sharpe_ranges = [ # Valori limite per valutazione Sharpe (X <= limite)
                (0, "Negativo", "#CC0000"),       # <= 0
                (0.5, "Scarso", "#CC8800"),       # > 0 e <= 0.5
                (1.0, "Accettabile", "#CCCC00"),  # > 0.5 e <= 1.0
                (2.0, "Buono", "#88CC00"),        # > 1.0 e <= 2.0
                (float('inf'), "Eccellente", "#00CC00") # > 2.0
            ]
            sharpe_help = create_metric_help(
                "Misura il rendimento aggiustato per il rischio (basato su rendimenti giornalieri). Più alto è, meglio è.",
                sharpe_ranges,
                "Un Sharpe Ratio > 1 è generalmente considerato buono."
            )
            sharpe_text_eval = metric_evaluation(sharpe_ratio_daily, sharpe_ranges)
            r_col2.markdown(f"**Sharpe Ratio (Giornaliero)**<br>{sharpe_text_eval}", unsafe_allow_html=True)
            r_col2.caption(sharpe_help)


            r_col3, r_col4 = st.columns(2)

            # Sortino Ratio
            sortino_ranges = [ # Valori limite per valutazione Sortino (X <= limite)
                (0, "Negativo", "#CC0000"),       # <= 0
                (0.75, "Scarso", "#CC8800"),      # > 0 e <= 0.75
                (1.5, "Accettabile", "#CCCC00"),  # > 0.75 e <= 1.5
                (2.5, "Buono", "#88CC00"),        # > 1.5 e <= 2.5
                (float('inf'), "Eccellente", "#00CC00") # > 2.5
            ]
            sortino_help = create_metric_help(
                "Simile allo Sharpe, ma considera solo la volatilità negativa (rischio di perdite).",
                sortino_ranges,
                "Un Sortino Ratio elevato indica buon rendimento con basso rischio di perdite significative."
            )
            sortino_text_eval = metric_evaluation(sortino_ratio, sortino_ranges) # Usa sortino_ratio (non annualizzato)
            r_col3.markdown(f"**Sortino Ratio**<br>{sortino_text_eval}", unsafe_allow_html=True)
            r_col3.caption(sortino_help)


            # VaR (Value at Risk)
            avg_stake_for_var = avg_stake_wl if pd.notna(avg_stake_wl) and avg_stake_wl > 0 else 1 # Evita divisione per zero
            var_as_perc_of_avg_stake = abs(var_95_daily) / avg_stake_for_var * 100 if pd.notna(var_95_daily) and avg_stake_for_var > 0 else np.nan

            var_ranges = [ # Valori limite per VaR come % dello stake medio (X <= limite) - invertiti perché perdita
                (50, "Eccellente", "#00CC00"),     # Perdita <= 50% stake medio
                (100, "Buono", "#88CC00"),        # Perdita > 50% e <= 100% stake medio
                (200, "Accettabile", "#CCCC00"),  # Perdita > 100% e <= 200% stake medio
                (300, "Problematico", "#CC8800"), # Perdita > 200% e <= 300% stake medio
                (float('inf'), "Critico", "#CC0000")# Perdita > 300% stake medio
            ]
            var_help = create_metric_help(
                "Perdita giornaliera massima attesa nel 5% dei casi peggiori (basata sui dati filtrati).",
                var_ranges,
                "Un VaR basso (come % dello stake medio) indica un rischio limitato di perdite estreme in un singolo giorno."
            )
            var_display_perc = f" ({var_as_perc_of_avg_stake:.0f}% stake medio)" if pd.notna(var_as_perc_of_avg_stake) else ""
            r_col4.metric(
                "VaR 95% (Giornaliero)",
                f"{format_currency(var_95_daily)}{var_display_perc}",
                help=var_help
            )

            # --- INIZIO BLOCCO CODICE CORRETTO ---
            # Seconda riga di indicatori
            r_col5, r_col6, r_col7, r_col8 = st.columns(4)

            # Calcolo durata massima del drawdown
            if not df_results_filtered_final.empty:
                df_dd = df_results_filtered_final.copy()
                # Assicurati che 'P/L' sia numerico e gestisci NaN prima di cumsum
                df_dd['P/L'] = pd.to_numeric(df_dd['P/L'], errors='coerce').fillna(0)
                df_dd['Cumulative P/L'] = df_dd['P/L'].cumsum()
                running_max_dd = df_dd['Cumulative P/L'].cummax() # Rinomina per evitare conflitti
                df_dd['Drawdown'] = running_max_dd - df_dd['Cumulative P/L']

                # Identifica i periodi di drawdown
                df_dd['In_Drawdown'] = df_dd['Drawdown'] > 0
                # Marks group transitions
                df_dd['Drawdown_Group'] = (df_dd['In_Drawdown'] != df_dd['In_Drawdown'].shift(1)).cumsum()

                # Trova la durata di ogni drawdown
                drawdown_groups = df_dd[df_dd['In_Drawdown']].groupby('Drawdown_Group')
                if not drawdown_groups.groups: # Controlla se ci sono gruppi
                    max_drawdown_duration = 0
                else:
                    drawdown_durations = []
                    for _, group_df in drawdown_groups: # Itera sui dataframe dei gruppi
                        if not group_df.empty and colonna_data in group_df.columns and pd.api.types.is_datetime64_any_dtype(group_df[colonna_data]):
                            duration_td = group_df[colonna_data].max() - group_df[colonna_data].min()
                            if pd.notna(duration_td): # Controllo per NaT
                                duration = duration_td.days + 1
                                drawdown_durations.append(duration)
                    max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0

                r_col5.metric("Max Durata Drawdown", f"{max_drawdown_duration} giorni",
                             help="Durata massima (in giorni) di un periodo continuo di perdita dal precedente picco (basato sui dati filtrati).")

            # Serie perdente più lunga (max_loss_streak calcolato prima)
            r_col6.metric("Max Serie Perdente", f"{max_loss_streak}", help="Massimo numero di scommesse perse consecutivamente (basato sui dati filtrati).")

            # Altri indicatori di rischio
            if not df_results_filtered_final.empty:
                 # Calcoliamo la distribuzione dei rendimenti giornalieri (daily_pl calcolato prima per Sharpe)
                 # daily_returns = df_results_filtered_final.groupby(pd.Grouper(key=colonna_data, freq='D'))['P/L'].sum() # Già fatto

                 # Win/Loss Ratio (media vincite / media perdite)
                 avg_win = df_results_filtered_final[df_results_filtered_final['P/L'] > 0]['P/L'].mean()
                 avg_loss_abs = abs(df_results_filtered_final[df_results_filtered_final['P/L'] < 0]['P/L'].mean()) # Rinomina variabile locale
                 win_loss_ratio = avg_win / avg_loss_abs if pd.notna(avg_loss_abs) and avg_loss_abs != 0 and pd.notna(avg_win) else np.nan


                 r_col7.metric("Win/Loss Ratio", f"{win_loss_ratio:.2f}" if pd.notna(win_loss_ratio) else "N/D",
                              help="Rapporto tra media delle vincite e valore assoluto medio delle perdite.")

                 # Calvert Ratio = Win Rate / (1 - Win Rate)
                 if pd.notna(win_rate_wl) and win_rate_wl < 1 and win_rate_wl > 0: # Evita divisione per zero o infinito
                     calvert_ratio = win_rate_wl / (1 - win_rate_wl)
                     r_col8.metric("Calvert Ratio", f"{calvert_ratio:.2f}",
                                  help="Rapporto tra probabilità di vincita e probabilità di perdita. Valori > 1 indicano che si vince più spesso di quanto si perde.")
                 else:
                     r_col8.metric("Calvert Ratio", "N/D", help="Non calcolabile (Win Rate è 0%, 100%, o N/D).")
            # --- FINE BLOCCO CODICE CORRETTO ---

        # --- Visualizzazioni con Stile Aggiornato ---
        st.markdown("---")
        st.subheader("Visualizzazioni Grafiche")

        # Grafico 1: Distribuzione Esiti (Win/Loss)
        st.markdown("##### Distribuzione Esiti (Win/Loss)")
        if not df_results_filtered_final.empty:
            outcome_counts = df_results_filtered_final['Esito_Standard'].value_counts().reset_index(); outcome_counts.columns = ['Esito_Standard', 'Conteggio']
            fig_outcome_pie = px.pie(outcome_counts, names='Esito_Standard', values='Conteggio', title="Distribuzione Esiti Win/Loss (Periodo Filtrato)", hole=0.3, color='Esito_Standard', color_discrete_map=color_map_esiti)
            fig_outcome_pie.update_traces(marker_line_width=0); fig_outcome_pie.update_layout(template='plotly_white', legend_title_text='Esito')
            st.plotly_chart(fig_outcome_pie, use_container_width=True); st.caption("Mostra la proporzione tra scommesse vinte e perse nel periodo filtrato.")
        else: st.info("Nessun dato W/L disponibile per i filtri selezionati.")
        st.markdown("---")

        # Grafico P/L Cumulativo
        st.subheader("Andamento Cumulativo P/L nel Tempo (Periodo Filtrato)")
        if not df_results_filtered_final.empty and 'Cumulative P/L Filtered' in df_results_filtered_final.columns and df_results_filtered_final['Cumulative P/L Filtered'].notna().any():
            fig_cum_pl = px.line(df_results_filtered_final, x=colonna_data, y='Cumulative P/L Filtered', title="Andamento P/L Cumulativo (€) - Periodo Filtrato", markers=False)
            # MODIFICA: Rimosso smoothing
            fig_cum_pl.update_traces(line=dict(color=color_line2, shape='linear')); fig_cum_pl.update_layout(template='plotly_white')
            if italian_locale_set:
                 fig_cum_pl.update_yaxes(title="P/L Cumulativo (€)", showgrid=False, tickformat="~€")
            else:
                 fig_cum_pl.update_yaxes(title="P/L Cumulativo (€)", showgrid=False, ticksuffix=" €")
            fig_cum_pl.update_xaxes(title="Data", showgrid=False)
            st.plotly_chart(fig_cum_pl, use_container_width=True); st.caption("Mostra l'evoluzione del P/L totale nel tempo per il periodo selezionato.")
        else: st.info("Nessun dato P/L per grafico cumulativo filtrato.")
        st.markdown("---")

        # Grafico P/L e Conteggio Giornaliero
        st.subheader("Andamento Giornaliero P/L e Volume Scommesse (Periodo Filtrato)")
        if not df_results_filtered_final.empty:
            daily_summary_chart = df_results_filtered_final.groupby(pd.Grouper(key=colonna_data, freq='D')).agg(Daily_PL=('P/L', 'sum'), Num_Bets=(colonna_esito, 'size')).reset_index()
            daily_summary_chart = daily_summary_chart[daily_summary_chart['Num_Bets'] > 0]
            if not daily_summary_chart.empty:
                fig_daily = make_subplots(specs=[[{"secondary_y": True}]]); colors_pl = [color_win if pl >= 0 else color_loss for pl in daily_summary_chart['Daily_PL']]
                fig_daily.add_trace(go.Bar(x=daily_summary_chart[colonna_data], y=daily_summary_chart['Daily_PL'], name='P/L Giornaliero (€)', marker_color=colors_pl, marker_line_width=0, yaxis='y1'), secondary_y=False)
                # MODIFICA: Rimosso smoothing
                fig_daily.add_trace(go.Scatter(x=daily_summary_chart[colonna_data], y=daily_summary_chart['Num_Bets'], name='Num. Scommesse', mode='lines+markers', line=dict(color=color_line1, shape='linear'), marker=dict(size=5), yaxis='y2'), secondary_y=True)
                fig_daily.update_layout(title_text="Risultato Netto (€) e Num. Scommesse per Giorno (Periodo Filtrato)", xaxis_title="Data", yaxis_title="P/L Netto Giornaliero (€)", yaxis2_title="Numero Scommesse", legend_title="Legenda", barmode='relative', template='plotly_white')
                if italian_locale_set:
                    fig_daily.update_yaxes(title_text="P/L Netto Giornaliero (€)", tickformat="~€", secondary_y=False, showgrid=False)
                else:
                    fig_daily.update_yaxes(title_text="P/L Netto Giornaliero (€)", ticksuffix=" €", secondary_y=False, showgrid=False)
                fig_daily.update_yaxes(title_text="Numero Scommesse", secondary_y=True, showgrid=False); fig_daily.update_xaxes(showgrid=False)
                st.plotly_chart(fig_daily, use_container_width=True); st.caption("Mostra il P/L netto giornaliero e il numero di scommesse concluse.")
            else: st.info("Nessun dato giornaliero aggregato.")
        else: st.info("Nessun dato W/L per grafico P/L giornaliero.")
        st.markdown("---")

        # Grafico Errore Sovrastima vs Probabilità
        st.subheader(f"Relazione tra Errore Sovrastima Punti e {colonna_prob}")
        if colonna_prob in df_results_filtered_final.columns and 'Errore_Sovrastima_PT' in df_results_filtered_final.columns:
            df_plot_prob_error = df_results_filtered_final[df_results_filtered_final[colonna_prob].notna() & df_results_filtered_final['Errore_Sovrastima_PT'].notna()].copy()
            if not df_plot_prob_error.empty:
                 df_plot_prob_error['Prob Formattata'] = df_plot_prob_error[colonna_prob].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
                 df_plot_prob_error['Errore Sovrastima Formattato'] = df_plot_prob_error['Errore_Sovrastima_PT'].map(lambda x: f"{x:.1f}" if pd.notna(x) else 'N/A')
                 fig_prob_error = px.scatter(df_plot_prob_error, x=colonna_prob, y='Errore_Sovrastima_PT', color='Esito_Standard', title=f"Errore Sovrastima Punti vs. {colonna_prob} (Periodo Filtrato)", labels={ colonna_prob: 'Probabilità Stimata', 'Errore_Sovrastima_PT': 'Errore Sovrastima Punti', 'Esito_Standard': 'Esito' }, hover_name=colonna_sq_a, hover_data={ colonna_sq_b: True, 'Prob Formattata': True, 'Errore Sovrastima Formattato': True, colonna_prob: False, 'Errore_Sovrastima_PT': False }, color_discrete_map=color_map_esiti)
                 fig_prob_error.update_traces(marker=dict(line=dict(width=0)))
                 fig_prob_error.update_layout(template='plotly_white')
                 fig_prob_error.update_xaxes(title="Probabilità Stimata", showgrid=False, tickformat=".1%")
                 fig_prob_error.update_yaxes(title="Errore Sovrastima Punti", showgrid=False)

                 fig_prob_error.add_vline(x=0.60, line_dash="dash", line_color="gray")
                 fig_prob_error.add_vline(x=0.65, line_dash="dash", line_color="gray")
                 fig_prob_error.add_vline(x=0.70, line_dash="dash", line_color="gray")

                 st.plotly_chart(fig_prob_error, use_container_width=True)
                 st.caption("Mostra la relazione tra la sovrastima dei punti (Media Stimati - Punteggio Finale, solo se positivo) e la probabilità stimata dell'esito finale (Win/Loss). Valori Y=0 indicano stima corretta o sottostima.")
            else: st.info(f"Nessun dato valido (con Prob. Stimata ed Errore Sovrastima calcolabile) per questo grafico.")
            st.markdown("---")
        else: st.info(f"Le colonne '{colonna_prob}' o '{colonna_media_pt_stimati}'/'{colonna_ris_finale}' (per calcolare l'errore) non sono disponibili."); st.markdown("---")

        # Grafico Errore Sovrastima vs Edge Value
        st.subheader(f"Relazione tra Errore Sovrastima Punti e {colonna_edge}")
        if colonna_edge in df_results_filtered_final.columns and 'Errore_Sovrastima_PT' in df_results_filtered_final.columns:
            df_plot_edge_error = df_results_filtered_final[df_results_filtered_final[colonna_edge].notna() & df_results_filtered_final['Errore_Sovrastima_PT'].notna()].copy()
            if not df_plot_edge_error.empty:
                 df_plot_edge_error['Edge Formattato'] = df_plot_edge_error[colonna_edge].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
                 df_plot_edge_error['Errore Sovrastima Formattato'] = df_plot_edge_error['Errore_Sovrastima_PT'].map(lambda x: f"{x:.1f}" if pd.notna(x) else 'N/A')
                 fig_edge_error = px.scatter(df_plot_edge_error, x=colonna_edge, y='Errore_Sovrastima_PT', color='Esito_Standard',
                                     title=f"Errore Sovrastima Punti vs. {colonna_edge} (Periodo Filtrato)",
                                     labels={colonna_edge: 'Edge Value', 'Errore_Sovrastima_PT': 'Errore Sovrastima Punti', 'Esito_Standard': 'Esito'},
                                     hover_name=colonna_sq_a,
                                     hover_data={colonna_sq_b: True, 'Edge Formattato': True, 'Errore Sovrastima Formattato': True,
                                               colonna_edge: False, 'Errore_Sovrastima_PT': False},
                                     color_discrete_map=color_map_esiti)
                 fig_edge_error.update_traces(marker=dict(line=dict(width=0)))
                 fig_edge_error.update_layout(template='plotly_white')
                 fig_edge_error.update_xaxes(title="Edge Value", showgrid=False, tickformat=".1%")
                 fig_edge_error.update_yaxes(title="Errore Sovrastima Punti", showgrid=False)
                 # MODIFICA: Aggiunta linee verticali tratteggiate
                 fig_edge_error.add_vline(x=0.10, line_dash="dash", line_color="gray")
                 fig_edge_error.add_vline(x=0.15, line_dash="dash", line_color="gray")
                 fig_edge_error.add_vline(x=0.20, line_dash="dash", line_color="gray")
                 st.plotly_chart(fig_edge_error, use_container_width=True)
                 st.caption("Mostra la relazione tra la sovrastima dei punti (Media Stimati - Punteggio Finale, solo se positivo) e l'edge value. Valori Y=0 indicano stima corretta o sottostima.")
            else:
                st.info(f"Nessun dato valido (con Edge Value ed Errore Sovrastima calcolabile) per questo grafico.")
            st.markdown("---")
        else:
            st.info(f"Le colonne '{colonna_edge}' o '{colonna_media_pt_stimati}'/'{colonna_ris_finale}' (per calcolare l'errore) non sono disponibili.")
            st.markdown("---")

        # NUOVO GRAFICO: Edge Value vs Probabilità Stimata
        st.subheader(f"Relazione tra {colonna_edge} e {colonna_prob}")
        if colonna_edge in df_results_filtered_final.columns and colonna_prob in df_results_filtered_final.columns:
            df_plot_edge_prob = df_results_filtered_final[
                df_results_filtered_final[colonna_edge].notna() &
                df_results_filtered_final[colonna_prob].notna() &
                df_results_filtered_final['Esito_Standard'].isin(['Win', 'Loss']) # Considera solo W/L
            ].copy()
            if not df_plot_edge_prob.empty:
                df_plot_edge_prob['Edge Formattato'] = df_plot_edge_prob[colonna_edge].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
                df_plot_edge_prob['Prob Formattata'] = df_plot_edge_prob[colonna_prob].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')

                fig_edge_prob = px.scatter(
                    df_plot_edge_prob,
                    x=colonna_edge,
                    y=colonna_prob,
                    color='Esito_Standard',
                    title=f"{colonna_edge} vs. {colonna_prob} (Periodo Filtrato)",
                    labels={
                        colonna_edge: 'Edge Value',
                        colonna_prob: 'Probabilità Stimata',
                        'Esito_Standard': 'Esito'
                    },
                    hover_name=colonna_sq_a,
                    hover_data={
                        colonna_sq_b: True,
                        'Edge Formattato': True,
                        'Prob Formattata': True,
                        colonna_quota: ':.2f',
                        colonna_edge: False, # Nasconde valore grezzo se già formattato
                        colonna_prob: False  # Nasconde valore grezzo se già formattato
                    },
                    color_discrete_map=color_map_esiti
                )
                fig_edge_prob.update_traces(marker=dict(line=dict(width=0)))
                fig_edge_prob.update_layout(template='plotly_white')
                fig_edge_prob.update_xaxes(title="Edge Value", showgrid=False, tickformat=".0%") # Mostra come X%
                fig_edge_prob.update_yaxes(title="Probabilità Stimata", showgrid=False, tickformat=".0%") # Mostra come X%

                # MODIFICA: Aggiunta linee divisorie tratteggiate ogni 5%
                # Linee verticali per Edge Value (asse X)
                # Determina dinamicamente il range per le linee verticali basato sui dati, con step di 0.05
                min_edge_val = 0.0
                max_edge_val = df_plot_edge_prob[colonna_edge].max() if not df_plot_edge_prob.empty else 0.3 # Fallback a 30%
                if pd.notna(max_edge_val):
                    for val in np.arange(0.05, max_edge_val + 0.05, 0.05): # Inizia da 0.05 (5%)
                         # Non disegnare la linea se è troppo vicina a 0 per evitare sovrapposizione con l'asse
                        if val > 0.001 : fig_edge_prob.add_vline(x=val, line_dash="dash", line_color="lightgray", opacity=0.7)


                # Linee orizzontali per Probabilità Stimata (asse Y)
                # Determina dinamicamente il range per le linee orizzontali basato sui dati, con step di 0.05
                min_prob_val = df_plot_edge_prob[colonna_prob].min() if not df_plot_edge_prob.empty else 0.4 # Fallback a 40%
                max_prob_val = df_plot_edge_prob[colonna_prob].max() if not df_plot_edge_prob.empty else 1.0 # Fallback a 100%

                # Arrotonda min_prob_val al multiplo di 0.05 più vicino verso il basso, ma non meno di 0.05
                if pd.notna(min_prob_val):
                    start_prob_line = max(0.05, np.floor(min_prob_val / 0.05) * 0.05)
                else:
                    start_prob_line = 0.4

                if pd.notna(max_prob_val) and pd.notna(start_prob_line):
                     for val in np.arange(start_prob_line, max_prob_val + 0.05, 0.05):
                         # Non disegnare la linea se è troppo vicina a 0 o 1 per evitare sovrapposizione con i bordi del grafico
                         if val > 0.001 and val < 0.999: fig_edge_prob.add_hline(y=val, line_dash="dash", line_color="lightgray", opacity=0.7)


                st.plotly_chart(fig_edge_prob, use_container_width=True)
                st.caption(f"Mostra la relazione tra l'{colonna_edge} e la {colonna_prob}, colorata per esito della scommessa. Le linee tratteggiate indicano intervalli del 5%.")
            else:
                st.info(f"Nessun dato valido (con {colonna_edge}, {colonna_prob} e esito W/L) per questo grafico.")
            st.markdown("---")
        else:
            st.info(f"Le colonne '{colonna_edge}' o '{colonna_prob}' non sono disponibili per questo grafico.")
            st.markdown("---")


        # MODIFICA: SPOSTATO GRAFICO "Analisi della precisione di previsione" QUI
        st.subheader("Analisi della Precisione di Previsione")
        if not df_filtered_final.empty and colonna_prob in df_filtered_final.columns:
            # Crea bins di probabilità stimata e calcola la precisione reale
            df_prediction = df_filtered_final[
                (df_filtered_final[colonna_prob].notna()) &
                (df_filtered_final['Esito_Standard'].isin(['Win', 'Loss']))
            ].copy()

            if not df_prediction.empty:
                # Crea bins di probabilità (es. 50-55%, 55-60%, ecc.)
                # Assicurati che colonna_prob sia numerica
                df_prediction[colonna_prob] = pd.to_numeric(df_prediction[colonna_prob], errors='coerce')
                df_prediction.dropna(subset=[colonna_prob], inplace=True) # Rimuovi NaN dopo coercizione


                if not df_prediction.empty and df_prediction[colonna_prob].min() < 1.0 and df_prediction[colonna_prob].max() > 0 : # Assicurati che ci siano dati e che le probabilità siano valide
                    prob_bins_def = [0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
                    prob_labels_def = ['<50%','50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90-95%', '95-100%']

                    current_prob_bins = prob_bins_def
                    current_prob_labels = prob_labels_def

                    min_prob_data = df_prediction[colonna_prob].min()
                    if pd.notna(min_prob_data) and min_prob_data >= 0.5:
                        start_index = 0
                        for i, p_bin_val in enumerate(prob_bins_def):
                            if min_prob_data < p_bin_val:
                                start_index = max(0, i -1)
                                break
                        current_prob_bins = prob_bins_def[start_index:]
                        current_prob_labels = prob_labels_def[start_index:]
                    
                    if not current_prob_bins or len(current_prob_bins) < 2: # Assicurati che ci siano almeno due bin per pd.cut
                        st.info("Range di probabilità troppo ristretto per creare bin significativi.")
                    else:
                        df_prediction['Prob_Bin'] = pd.cut(
                            df_prediction[colonna_prob],
                            bins=current_prob_bins,
                            labels=current_prob_labels if len(current_prob_bins) -1 == len(current_prob_labels) else False, 
                            include_lowest=True,
                            right=True 
                        )


                        # Calcola la precisione reale per ogni bin
                        prediction_accuracy = df_prediction.groupby('Prob_Bin', observed=False).agg(
                            Scommesse_Totali=('Esito_Standard', 'size'),
                            Vincite=('Esito_Standard', lambda x: (x == 'Win').sum()),
                            Prob_Media=(colonna_prob, 'mean')
                        ).reset_index()


                        prediction_accuracy['Precisione_Reale'] = prediction_accuracy['Vincite'] / prediction_accuracy['Scommesse_Totali']
                        prediction_accuracy['Differenza'] = prediction_accuracy['Precisione_Reale'] - prediction_accuracy['Prob_Media']
                        prediction_accuracy.dropna(subset=['Prob_Media', 'Precisione_Reale'], inplace=True)

                        if not prediction_accuracy.empty:
                            fig_calibration = go.Figure()
                            perfect_cal_x_min = prediction_accuracy['Prob_Media'].min() if not prediction_accuracy.empty else 0.5
                            perfect_cal_x_max = prediction_accuracy['Prob_Media'].max() if not prediction_accuracy.empty else 1.0
                            perfect_cal_x = np.linspace(min(0.45, perfect_cal_x_min), max(1.0, perfect_cal_x_max), 10)


                            fig_calibration.add_trace(
                                go.Scatter(x=perfect_cal_x, y=perfect_cal_x, mode='lines', name='Calibrazione Perfetta',
                                          line=dict(color='black', width=1, dash='dash'))
                            )

                            sizes = prediction_accuracy['Scommesse_Totali'].clip(lower=5) * 1.5
                            fig_calibration.add_trace(
                                go.Scatter(
                                    x=prediction_accuracy['Prob_Media'],
                                    y=prediction_accuracy['Precisione_Reale'],
                                    mode='markers+text',
                                    marker=dict(
                                        size=sizes,
                                        color=prediction_accuracy['Differenza'],
                                        colorscale='RdYlGn',
                                        cmid=0,
                                        cmin=-0.2,
                                        cmax=0.2,
                                        colorbar=dict(title='Sovra/Sotto Stima'),
                                        line=dict(width=1, color='black')
                                    ),
                                    text=prediction_accuracy['Prob_Bin'].astype(str), 
                                    textposition="top center",
                                    name='Calibrazione Reale',
                                    hovertemplate='<b>Bin: %{text}</b><br>Probabilità Media Stimata: %{x:.1%}<br>Precisione Reale (Win Rate): %{y:.1%}<br>Differenza: %{marker.color:+.1%}<br>Scommesse nel Bin: %{customdata[0]}<extra></extra>',
                                    customdata=prediction_accuracy[['Scommesse_Totali']]
                                )
                            )

                            y_axis_max = 1.05
                            if not prediction_accuracy.empty and prediction_accuracy['Precisione_Reale'].notna().any():
                                y_axis_max = max(1.05, prediction_accuracy['Precisione_Reale'].max() + 0.05)


                            fig_calibration.update_layout(
                                title='Calibrazione della Previsione: Probabilità Stimata vs. Precisione Reale',
                                xaxis_title='Probabilità Media Stimata nel Bin',
                                yaxis_title='Precisione Reale (Win Rate) nel Bin',
                                template='plotly_white',
                                xaxis=dict(tickformat='.0%', range=[min(0.45, perfect_cal_x_min -0.05) , max(1.0, perfect_cal_x_max+0.05)]),
                                yaxis=dict(tickformat='.0%', range=[0, y_axis_max])
                            )

                            st.plotly_chart(fig_calibration, use_container_width=True)
                            st.caption("Analisi della calibrazione: confronta la probabilità stimata con la frequenza reale di vincita per diversi intervalli di probabilità. I punti sopra la diagonale indicano sottostima della probabilità reale, sotto indicano sovrastima. Idealmente, i punti dovrebbero essere vicini alla linea diagonale.")

                            st.markdown("##### Dettaglio Calibrazione Previsioni")
                            display_accuracy = prediction_accuracy.copy()
                            display_accuracy['Prob_Media'] = display_accuracy['Prob_Media'].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                            display_accuracy['Precisione_Reale'] = display_accuracy['Precisione_Reale'].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                            display_accuracy['Differenza'] = display_accuracy['Differenza'].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")

                            display_accuracy.columns = ['Range Probabilità Stimata', 'Totale Scommesse', 'Vincite', 'Prob. Media Stimata',
                                                          'Precisione Reale', 'Differenza (Reale - Stimata)']
                            st.dataframe(display_accuracy, use_container_width=True, hide_index=True)
                        else:
                            st.info("Nessun dato sufficiente per il grafico di calibrazione dopo il binning.")
                else:
                    st.info("Probabilità stimate non valide o fuori range per l'analisi di calibrazione (es. tutte NaN, <0 o >= 1.0).")
            else:
                st.info("Nessun dato con esito Win/Loss e probabilità stimata per l'analisi di calibrazione.")
            st.markdown("---")
        else:
            st.info(f"La colonna '{colonna_prob}' è necessaria per l'analisi di precisione della previsione ma non è disponibile.")
            st.markdown("---")


        # Grafici Edge Value vs P/L e Distribuzione Stake
        col_c, col_d = st.columns(2)
        with col_c:
             if colonna_edge in df_results_filtered_final.columns:
                 st.subheader(f"Rapporto {colonna_edge} vs P/L (Vittoria/Perdita)") # Titolo modificato per chiarezza
                 df_plot_edge_pl = df_results_filtered_final[df_results_filtered_final[colonna_edge].notna()].copy()
                 if not df_plot_edge_pl.empty:
                     df_plot_edge_pl['Stake Formattato'] = df_plot_edge_pl[colonna_stake].apply(format_currency)
                     df_plot_edge_pl['P/L Formattato'] = df_plot_edge_pl['P/L'].apply(format_currency)
                     df_plot_edge_pl['Edge Formattato'] = df_plot_edge_pl[colonna_edge].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
                     fig_edge_pl = px.scatter(df_plot_edge_pl, x=colonna_edge, y='P/L', color='Esito_Standard',
                                              title="P/L vs. Edge Value (Periodo Filtrato)",
                                              labels={colonna_edge: 'Edge Value', 'P/L': 'P/L (€)', 'Esito_Standard': 'Esito'},
                                              hover_name=colonna_sq_a,
                                              hover_data={colonna_sq_b:True, colonna_quota:':.2f', 'Stake Formattato': True,
                                                        'P/L Formattato':True, 'Edge Formattato':True,
                                                        colonna_edge:False, 'P/L':False},
                                              color_discrete_map=color_map_esiti)
                     fig_edge_pl.update_traces(marker=dict(line=dict(width=0)));
                     fig_edge_pl.update_layout(template='plotly_white')
                     fig_edge_pl.update_xaxes(title="Edge Value", showgrid=False, tickformat=".1%")
                     if italian_locale_set:
                         fig_edge_pl.update_yaxes(title="P/L (€)", showgrid=False, tickformat="~€")
                     else:
                         fig_edge_pl.update_yaxes(title="P/L (€)", showgrid=False, ticksuffix=" €")
                     # MODIFICA: Aggiunta linea verticale tratteggiata
                     fig_edge_pl.add_vline(x=0.20, line_dash="dash", line_color="gray")
                     st.plotly_chart(fig_edge_pl, use_container_width=True);
                     st.caption("Esplora la correlazione tra edge value e P/L.")
                 else: st.info("Nessun dato W/L con Edge Value valido.")
             else: st.info(f"Colonna '{colonna_edge}' non disponibile.")

        with col_d:
              if colonna_stake in df_filtered_final.columns:
                 st.subheader(f"Partecipazione Distribuzione (Tutti Filtrati > 0)") # Titolo aggiornato per coerenza
                 df_plot_stake = df_filtered_final[df_filtered_final[colonna_stake].notna() & (df_filtered_final[colonna_stake] > 0)]
                 if not df_plot_stake.empty:
                     fig_stake_hist = px.histogram(df_plot_stake, x=colonna_stake, nbins=15, title="Distribuzione Stake (€) (Periodo Filtrato)", labels={colonna_stake: 'Importo Puntato (€)'}, color_discrete_sequence=[color_line1])
                     fig_stake_hist.update_traces(marker_line_width=0); fig_stake_hist.update_layout(template='plotly_white', bargap=0.1); fig_stake_hist.update_yaxes(title="Numero Scommesse", showgrid=False)
                     if italian_locale_set:
                         fig_stake_hist.update_xaxes(title="Importo Puntato (€)", showgrid=False, tickformat="~€")
                     else:
                         fig_stake_hist.update_xaxes(title="Importo Puntato (€)", showgrid=False, ticksuffix=" €")
                     st.plotly_chart(fig_stake_hist, use_container_width=True); st.caption("Mostra come si distribuiscono gli importi puntati.")
                 else: st.info("Nessuna puntata valida (> 0 €).")
              else: st.info(f"Colonna '{colonna_stake}' non disponibile.")
        st.markdown("---") # Aggiunto un separatore per coerenza dopo la colonna

        # MODIFICA: Spostato il grafico "Conteggio Esiti per Quota" qui
        st.markdown(f"##### Conteggio Esiti (Vittoria/Perdita) per {colonna_quota}")
        if not df_results_filtered_final.empty and colonna_quota in df_results_filtered_final.columns:
             df_plot_odds = df_results_filtered_final.groupby([colonna_quota, 'Esito_Standard']).size().reset_index(name='Conteggio')
             fig_odds_outcome = px.bar(df_plot_odds, x=colonna_quota, y='Conteggio', color='Esito_Standard', barmode='group', title=f"Conteggio Win/Loss per {colonna_quota} (Periodo Filtrato)", labels={colonna_quota: 'Quota', 'Conteggio': 'Num. Scommesse', 'Esito_Standard': 'Esito'}, color_discrete_map=color_map_esiti)
             fig_odds_outcome.update_traces(marker_line_width=0); fig_odds_outcome.update_layout(template='plotly_white', bargap=0.2)
             fig_odds_outcome.update_yaxes(showgrid=False); fig_odds_outcome.update_xaxes(showgrid=False)
             st.plotly_chart(fig_odds_outcome, use_container_width=True); st.caption(f"Visualizza quante scommesse vinte/perse per fascia di quota.")
        else: st.info(f"Nessun dato per grafico esiti per {colonna_quota}.")
        st.markdown("---")


        # --- NUOVE FUNZIONALITÀ (Evoluzione e Simulazioni) ---
        st.subheader("Analisi Avanzate e Simulazioni")


        # Analisi dell'evoluzione del sistema
        st.markdown("---") # Separatore prima di questa sotto-sezione
        st.markdown("##### Evoluzione del Sistema nel Tempo") # Titolo più specifico

        if not df_results_filtered_final.empty and colonna_data in df_results_filtered_final.columns and df_results_filtered_final[colonna_data].notna().any():
            # Calcola metriche per finestra mobile (per vedere l'evoluzione)
            max_slider_val = len(df_results_filtered_final) if len(df_results_filtered_final) > 5 else 5
            default_slider_val = min(20, max_slider_val)

            window_size = st.slider("Finestra mobile (numero di scommesse)", min_value=5, max_value=max_slider_val, value=default_slider_val, 
                                  help="Numero di scommesse considerate per ogni punto della finestra mobile")

            if len(df_results_filtered_final) >= window_size:
                # Calcola ROI e win rate per finestra mobile
                df_sorted = df_results_filtered_final.sort_values(by=colonna_data).reset_index(drop=True)

                rolling_roi = []
                rolling_winrate = []
                rolling_dates = []

                for i in range(window_size, len(df_sorted) + 1):
                    window = df_sorted.iloc[i-window_size:i]

                    # ROI per finestra
                    window_stake = window[colonna_stake].sum()
                    window_pl = window['P/L'].sum()
                    window_roi = (window_pl / window_stake) * 100 if window_stake > 0 else 0

                    # Win rate per finestra
                    window_wins = (window['Esito_Standard'] == 'Win').sum()
                    window_winrate = window_wins / window_size

                    rolling_roi.append(window_roi)
                    rolling_winrate.append(window_winrate)
                    rolling_dates.append(window.iloc[-1][colonna_data])

                # Crea dataframe per grafico
                df_rolling = pd.DataFrame({
                    'Data': rolling_dates,
                    'ROI': rolling_roi,
                    'Win_Rate': rolling_winrate
                })

                # Grafico dell'evoluzione
                fig_evolution = make_subplots(specs=[[{"secondary_y": True}]])
                # MODIFICA: Rimosso smoothing
                fig_evolution.add_trace(
                    go.Scatter(x=df_rolling['Data'], y=df_rolling['ROI'],
                              name=f'ROI Mobile ({window_size} scommesse)',
                              line=dict(color='#4682B4', width=2, shape='linear')), # MODIFICATO shape
                    secondary_y=False
                )
                # MODIFICA: Rimosso smoothing
                fig_evolution.add_trace(
                    go.Scatter(x=df_rolling['Data'], y=df_rolling['Win_Rate'],
                              name=f'Win Rate Mobile ({window_size} scommesse)',
                              line=dict(color='#7B68EE', width=2, shape='linear')), # MODIFICATO shape
                    secondary_y=True
                )

                fig_evolution.update_layout(
                    title=f"Evoluzione del Sistema - Finestra Mobile di {window_size} scommesse",
                    template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )

                fig_evolution.update_yaxes(title_text="ROI (%)", secondary_y=False, showgrid=False)
                fig_evolution.update_yaxes(title_text="Win Rate", secondary_y=True, showgrid=False, tickformat=".1%")
                fig_evolution.update_xaxes(title_text="Data", showgrid=False)

                st.plotly_chart(fig_evolution, use_container_width=True)
                st.caption("Mostra l'evoluzione del ROI e Win Rate nel tempo usando una finestra mobile. Utile per verificare se il sistema migliora o peggiora nel tempo.")
            else:
                st.info(f"Insufficiente numero di scommesse per l'analisi con finestra mobile. Ne servono almeno {window_size} (dati filtrati: {len(df_results_filtered_final)}).")
        else:
            st.info("Nessun dato o date valide per l'analisi dell'evoluzione del sistema.")



        # Simulatore di strategie di staking
        st.markdown("---")
        st.markdown("##### Simulatore di Strategie di Staking") # Titolo più specifico

        if not df_results_filtered_final.empty:
            col1_sim, col2_sim = st.columns(2) # Rinomina per evitare conflitti

            # Selezione tipo di strategia
            staking_strategy = col1_sim.selectbox(
                "Seleziona Strategia di Staking",
                ["Stake Fisso", "Percentuale Bankroll", "Kelly Criterion", "Stake Proporzionale all'Edge"]
            )

            # Parametri di configurazione
            starting_bankroll = col2_sim.number_input("Bankroll Iniziale (€)", min_value=100.0, value=1000.0, step=100.0)


            # Descrizioni dettagliate per ciascuna strategia
            staking_descriptions = {
                "Stake Fisso": """
                **Stake Fisso**: Puntata sempre dello stesso importo, indipendentemente dal bankroll o dalle probabilità.

                **Principio**: Semplicità e costanza.

                **Grandezze considerate**:
                - Importo fisso deciso dall'utente

                **Vantaggi**:
                - Semplice da implementare
                - Facile da tracciare

                **Svantaggi**:
                - Non si adatta al bankroll crescente o calante
                - Non considera le probabilità o vantaggio atteso
                """,

                "Percentuale Bankroll": """
                **Percentuale Bankroll**: Puntata calcolata come percentuale fissa del bankroll attuale.

                **Principio**: Crescita o diminuzione proporzionale al bankroll.

                **Grandezze considerate**:
                - Bankroll attuale
                - Percentuale fissa decisa dall'utente

                **Vantaggi**:
                - Si adatta automaticamente alle vincite/perdite
                - Protegge contro il fallimento (stake diminuisce con le perdite)
                - Sfrutta i periodi vincenti (stake aumenta con le vincite)

                **Svantaggi**:
                - Non considera la qualità della singola scommessa (probabilità o vantaggio)
                """,

                "Kelly Criterion": """
                **Kelly Criterion**: Formula matematica che massimizza la crescita attesa del bankroll.

                **Principio**: Crescita ottimale del capitale nel lungo termine.

                **Formula (frazione da puntare del bankroll)**: Kelly = (p × b - (1-p)) / b
                dove:
                - p = probabilità di vincita stimata (es. 0.6 per 60%)
                - b = quota decimale - 1 (es. per quota 2.5, b = 1.5)

                **Grandezze considerate**:
                - Probabilità stimata di vincita
                - Quota offerta
                - Bankroll attuale
                - Frazione di Kelly (per moderare il rischio, es. Half Kelly = 50%)

                **Vantaggi**:
                - Matematicamente ottimale per crescita a lungo termine (se le probabilità sono stimate correttamente)
                - Considera sia probabilità che quota per massimizzare il valore

                **Svantaggi**:
                - Molto sensibile a errori di stima della probabilità (sovrastimare p porta a stake eccessivi)
                - Può suggerire stake molto elevati (rischiosi) con edge percepito alto
                - Richiede la colonna 'Probabilita Stimata' nel CSV.
                """,

                "Stake Proporzionale all'Edge": """
                **Stake Proporzionale all'Edge**: Puntata proporzionale al vantaggio atteso (edge) della scommessa.

                **Principio**: Più alto è il vantaggio, maggiore sarà la puntata (fino a un cap).

                **Calcolo Edge**: Edge = (probabilità stimata × quota decimale) - 1

                **Grandezze considerate**:
                - Probabilità stimata di vincita
                - Quota offerta
                - Moltiplicatore edge (per regolare l'aggressività)
                - Bankroll attuale (per limiti massimi, es. max 10% del bankroll)
                - Richiede la colonna 'Probabilita Stimata' nel CSV.
                """
            }

            # Mostra descrizione della strategia
            with st.expander(f"Dettagli sulla Strategia: {staking_strategy}"):
                st.markdown(staking_descriptions[staking_strategy])

            # Parametri specifici per strategia con range di valori
            stake_param = None # Inizializza stake_param
            if staking_strategy == "Stake Fisso":
                fixed_stake = st.slider("Stake Fisso (€)", min_value=1.0, max_value=min(100.0, starting_bankroll), value=min(10.0, starting_bankroll/10), step=1.0)
                stake_param = fixed_stake

                # Mostra info sul range di stake
                st.caption(f"Con questa configurazione, lo stake sarà sempre **{fixed_stake:.2f} €** per ogni scommessa (se il bankroll lo permette).")

            elif staking_strategy == "Percentuale Bankroll":
                bankroll_perc = st.slider("Percentuale Bankroll per Scommessa (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
                stake_param = bankroll_perc / 100

                # Calcola e mostra range di stake
                initial_stake_val = starting_bankroll * stake_param # Rinomina variabile
                min_stake_val = max(1.0, starting_bankroll * 0.1 * stake_param)  # Assumendo perdita del 90%
                max_stake_val = starting_bankroll * 2 * stake_param  # Assumendo raddoppio del bankroll


                st.caption(f"""
                **Range di Stake Indicativo**: Con bankroll iniziale di {starting_bankroll:.2f} € e percentuale del {bankroll_perc:.1f}%:
                - Stake iniziale: **{initial_stake_val:.2f} €**
                - Range atteso (varia con il bankroll): da **{min_stake_val:.2f} €** a **{max_stake_val:.2f} €**
                """)

            elif staking_strategy == "Kelly Criterion":
                if colonna_prob not in df_results_filtered_final.columns:
                    st.error(f"La strategia 'Kelly Criterion' richiede la colonna '{colonna_prob}' nel file CSV.")
                else:
                    kelly_fraction_perc = st.slider("Frazione di Kelly (%)", min_value=1, max_value=100, value=50, step=1,
                                             help="Percentuale del Kelly stake da utilizzare (100% = Kelly completo, 50% = Half Kelly). Riduce la volatilità.")
                    stake_param = kelly_fraction_perc / 100


                    # Calcola stake range per Kelly (esempio basato su medie)
                    avg_prob_kelly = df_results_filtered_final[colonna_prob].mean() if colonna_prob in df_results_filtered_final else None
                    avg_odds_kelly = df_results_filtered_final[colonna_quota].mean() if colonna_quota in df_results_filtered_final else None


                    if pd.notna(avg_prob_kelly) and pd.notna(avg_odds_kelly) and avg_odds_kelly > 1 and avg_prob_kelly > 0 and avg_prob_kelly < 1:
                        b_kelly = avg_odds_kelly - 1
                        full_kelly_frac = (avg_prob_kelly * b_kelly - (1 - avg_prob_kelly)) / b_kelly
                        full_kelly_frac = max(0, full_kelly_frac) # Kelly non può essere negativo

                        actual_kelly_frac_to_bet = full_kelly_frac * stake_param
                        example_stake = starting_bankroll * actual_kelly_frac_to_bet

                        st.caption(f"""
                        **Stake Kelly Indicativo (basato su medie filtrate)**:
                        - Prob. media: {avg_prob_kelly:.2%}, Quota media: {avg_odds_kelly:.2f}
                        - Frazione Kelly Completa Calcolata: {full_kelly_frac:.2%}
                        - Frazione Kelly da Puntare (con slider {kelly_fraction_perc}%): {actual_kelly_frac_to_bet:.2%} del bankroll
                        - Stake Esempio con Bankroll Iniziale: **{example_stake:.2f} €**
                        - Lo stake sarà 0 € se l'edge calcolato è nullo o negativo.
                        """)
                    else:
                        st.caption("Impossibile calcolare un esempio di stake Kelly con le medie dei dati filtrati (probabilità o quote non valide/disponibili).")


            else:  # Stake Proporzionale all'Edge
                if colonna_prob not in df_results_filtered_final.columns:
                     st.error(f"La strategia 'Stake Proporzionale all'Edge' richiede la colonna '{colonna_prob}' nel file CSV.")
                else:
                    edge_multiplier = st.slider("Moltiplicatore Stake per Unità di Edge", min_value=1, max_value=200, value=50, step=1,
                                              help="Quanto puntare per ogni punto percentuale di edge (es. Edge 0.10 * Multiplier 50 = Stake 5€, soggetto a cap).")
                    max_stake_perc_bankroll = st.slider("Cap Massimo Stake (% del Bankroll)", min_value=1.0, max_value=20.0, value=10.0, step=0.5,
                                                      help="Massima percentuale del bankroll da puntare per una singola scommessa, indipendentemente dall'edge.")
                    stake_param = {'multiplier': edge_multiplier, 'cap_perc': max_stake_perc_bankroll / 100}


                    # Calcola range di stake per Edge (esempio basato su medie)
                    avg_prob_edge = df_results_filtered_final[colonna_prob].mean() if colonna_prob in df_results_filtered_final else None
                    avg_odds_edge = df_results_filtered_final[colonna_quota].mean() if colonna_quota in df_results_filtered_final else None

                    if pd.notna(avg_prob_edge) and pd.notna(avg_odds_edge):
                        calculated_edge = (avg_prob_edge * avg_odds_edge) - 1
                        example_stake_edge = 0
                        if calculated_edge > 0:
                            example_stake_edge = min(calculated_edge * edge_multiplier, starting_bankroll * stake_param['cap_perc'])

                        st.caption(f"""
                        **Stake Edge Indicativo (basato su medie filtrate)**:
                        - Prob. media: {avg_prob_edge:.2%}, Quota media: {avg_odds_edge:.2f} => Edge Calcolato: {calculated_edge:.2%}
                        - Stake Esempio (con moltiplicatore {edge_multiplier} e cap {max_stake_perc_bankroll}%): **{example_stake_edge:.2f} €**
                        - Lo stake sarà 0 € se l'edge calcolato è nullo o negativo.
                        """)
                    else:
                        st.caption("Impossibile calcolare un esempio di stake Edge con le medie dei dati filtrati.")


            # Esegui simulazione
            if st.button("Simula Strategia"):
                # Verifica se le colonne necessarie per Kelly o Edge sono presenti, se la strategia è selezionata
                proceed_simulation = True
                if staking_strategy in ["Kelly Criterion", "Stake Proporzionale all'Edge"] and colonna_prob not in df_results_filtered_final.columns:
                    st.error(f"Simulazione interrotta: la colonna '{colonna_prob}' è necessaria per la strategia '{staking_strategy}' ma non è presente nei dati filtrati.")
                    proceed_simulation = False

                if proceed_simulation and stake_param is not None: # Assicurati che stake_param sia definito
                    # Ordina le scommesse per data
                    df_sim = df_results_filtered_final.sort_values(by=colonna_data).copy()
                    df_sim.reset_index(drop=True, inplace=True) # Resetta indice per iterrows

                    # Setup per simulazione
                    bankroll = float(starting_bankroll)
                    bankroll_history = [bankroll]
                    bet_amounts_sim = [] # Rinomina per evitare conflitti
                    stakes_as_perc_of_bankroll = []


                    for index_sim, bet_row in df_sim.iterrows(): # Rinomina variabili
                        current_stake = 0.0 # Inizializza stake per questo giro
                        # Controllo che il bankroll sia positivo
                        if bankroll <= 0.01: # Se il bankroll è virtualmente zero
                            current_stake = 0.0
                        # Calcola lo stake in base alla strategia
                        elif staking_strategy == "Stake Fisso":
                            current_stake = float(min(stake_param, bankroll))
                        elif staking_strategy == "Percentuale Bankroll":
                            current_stake = float(min(bankroll * float(stake_param), bankroll))
                        elif staking_strategy == "Kelly Criterion":
                            prob_kelly = bet_row[colonna_prob] if pd.notna(bet_row[colonna_prob]) else 0
                            quota_kelly = bet_row[colonna_quota] if pd.notna(bet_row[colonna_quota]) else 0

                            if prob_kelly > 0 and prob_kelly < 1 and quota_kelly > 1:
                                b_val_kelly = quota_kelly - 1 # Guadagno netto
                                kelly_full_fraction = (prob_kelly * b_val_kelly - (1 - prob_kelly)) / b_val_kelly
                                if kelly_full_fraction > 0:
                                    stake_fraction_to_bet = kelly_full_fraction * float(stake_param) # stake_param è kelly_fraction (0-1)
                                    current_stake = float(min(stake_fraction_to_bet * bankroll, bankroll)) # Non puntare più del bankroll
                                else:
                                    current_stake = 0.0 # Edge non positivo
                            else:
                                current_stake = 0.0 # Dati non validi per Kelly
                        else:  # Stake Proporzionale all'Edge
                            prob_edge_sim = bet_row[colonna_prob] if pd.notna(bet_row[colonna_prob]) else 0 # Rinomina
                            quota_edge_sim = bet_row[colonna_quota] if pd.notna(bet_row[colonna_quota]) else 0 # Rinomina


                            if prob_edge_sim > 0 and quota_edge_sim > 1:
                                edge_value = (prob_edge_sim * quota_edge_sim) - 1
                                if edge_value > 0:
                                    multiplier = float(stake_param['multiplier'])
                                    cap_fraction = float(stake_param['cap_perc'])

                                    stake_calculated_from_edge = edge_value * multiplier
                                    stake_capped_by_bankroll_perc = bankroll * cap_fraction
                                    current_stake = float(min(stake_calculated_from_edge, stake_capped_by_bankroll_perc, bankroll))
                                else:
                                    current_stake = 0.0 # Edge non positivo
                            else:
                                current_stake = 0.0 # Dati non validi


                        # Assicurati che lo stake sia un numero valido e limitato
                        if not pd.notna(current_stake): current_stake = 0.0
                        current_stake = float(max(0.0, min(float(current_stake), bankroll)))

                        bet_amounts_sim.append(current_stake)
                        stakes_as_perc_of_bankroll.append((current_stake / bankroll * 100) if bankroll > 0 else 0)


                        # Aggiorna il bankroll
                        if bet_row['Esito_Standard'] == 'Win':
                            quota_win_sim = float(bet_row[colonna_quota]) if pd.notna(bet_row[colonna_quota]) else 1.0 # Rinomina
                            profit = float(current_stake) * (quota_win_sim - 1.0)
                            bankroll += profit
                        elif bet_row['Esito_Standard'] == 'Loss': # Solo Loss, TBD non dovrebbe essere qui
                            bankroll -= float(current_stake)


                        # Assicurati che il bankroll sia un valore numerico valido
                        if not pd.notna(bankroll): bankroll = 0.0
                        bankroll = max(0.0, bankroll)

                        bankroll_history.append(bankroll)
                        if bankroll < 0.01 and index_sim < len(df_sim) -1: # Se il bankroll è esaurito prima dell'ultima scommessa
                            # Riempi il resto della storia e degli stake con zero per mantenere la stessa lunghezza
                            remaining_bets = len(df_sim) - 1 - index_sim
                            bankroll_history.extend([0.0] * remaining_bets)
                            bet_amounts_sim.extend([0.0] * remaining_bets)
                            stakes_as_perc_of_bankroll.extend([0.0] * remaining_bets)
                            break


                    # Aggiungi stake e bankroll alla dataframe (solo se le lunghezze corrispondono)
                    if len(bet_amounts_sim) == len(df_sim):
                        df_sim['Stake_Simulato'] = bet_amounts_sim
                        df_sim['Stake_Simulato_%Bankroll'] = stakes_as_perc_of_bankroll
                    else: # Se il bankroll è finito prima, bet_amounts_sim potrebbe essere più corto
                         # Riempi gli stake mancanti con 0 per allineare
                        if len(bet_amounts_sim) < len(df_sim):
                            diff_len = len(df_sim) - len(bet_amounts_sim)
                            bet_amounts_sim.extend([0.0] * diff_len)
                            stakes_as_perc_of_bankroll.extend([0.0] * diff_len)
                        df_sim['Stake_Simulato'] = bet_amounts_sim[:len(df_sim)]
                        df_sim['Stake_Simulato_%Bankroll'] = stakes_as_perc_of_bankroll[:len(df_sim)]


                    # Calcola statistiche della simulazione
                    final_bankroll = bankroll_history[-1]
                    total_staked_sim = sum(bet_amounts_sim)
                    profit_sim = final_bankroll - starting_bankroll
                    roi_sim = (profit_sim / total_staked_sim * 100) if total_staked_sim > 0 else 0

                    max_bankroll_hist = max(bankroll_history) if bankroll_history else starting_bankroll # Rinomina


                    # Calcolo max drawdown con gestione errori
                    max_drawdown_abs_sim = 0 # Rinomina
                    max_drawdown_perc_sim = 0 # Rinomina
                    if bankroll_history:
                        bh_series = pd.Series(bankroll_history)
                        cumulative_max_bh = bh_series.cummax()
                        drawdowns_bh = cumulative_max_bh - bh_series
                        if not drawdowns_bh.empty: # Controllo se drawdowns_bh è vuoto
                            max_drawdown_abs_sim = drawdowns_bh.max()

                            # Trova il picco da cui è iniziato il max drawdown
                            if max_drawdown_abs_sim > 0:
                                idx_max_dd_end = drawdowns_bh.idxmax() # Indice della fine del max drawdown
                                peak_for_max_dd = cumulative_max_bh[idx_max_dd_end]
                                max_drawdown_perc_sim = (max_drawdown_abs_sim / peak_for_max_dd * 100) if peak_for_max_dd > 0 else 0
                            else:
                                max_drawdown_perc_sim = 0
                        else: # Se drawdowns_bh è vuoto (es. bankroll_history ha un solo elemento)
                             max_drawdown_abs_sim = 0
                             max_drawdown_perc_sim = 0


                    # Visualizza risultati
                    st.markdown("##### Risultati Simulazione")
                    metric_col1_res, metric_col2_res, metric_col3_res, metric_col4_res = st.columns(4) # Rinomina
                    metric_col1_res.metric("Bankroll Finale", f"{final_bankroll:.2f} €", f"{profit_sim:+.2f} €")
                    metric_col2_res.metric("ROI Simulato", f"{roi_sim:.2f}%", help=f"Profitto / Totale Puntato ({format_currency(profit_sim)} / {format_currency(total_staked_sim)})")
                    metric_col3_res.metric("Bankroll Massimo Raggiunto", f"{max_bankroll_hist:.2f} €")
                    metric_col4_res.metric("Max Drawdown Simulato", f"{max_drawdown_abs_sim:.2f} € ({max_drawdown_perc_sim:.1f}%)", help="Massima perdita dal picco precedente durante la simulazione.")


                    # Grafico evoluzione bankroll
                    fig_bankroll = go.Figure()
                    # MODIFICA: Rimosso smoothing se presente (in questo caso non c'era, ma per coerenza)
                    fig_bankroll.add_trace(go.Scatter(
                        x=df_sim[colonna_data] if colonna_data in df_sim.columns and not df_sim[colonna_data].empty else df_sim.index, # Usa data o indice
                        y=bankroll_history[1:] if len(bankroll_history) > 1 else [], # Escludi il bankroll iniziale che non corrisponde a una scommessa
                        mode='lines',
                        name='Andamento Bankroll',
                        line=dict(color='#2E8B57', width=2, shape='linear') # Aggiunto shape='linear'
                    ))

                    fig_bankroll.add_hline(y=starting_bankroll, line_dash="dash", line_color="red",
                                          annotation_text="Bankroll Iniziale", annotation_position="bottom right")

                    fig_bankroll.update_layout(
                        title=f"Evoluzione Bankroll - Strategia: {staking_strategy}",
                        xaxis_title='Scommessa (ordinata per data)',
                        yaxis_title='Bankroll (€)',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_bankroll, use_container_width=True)

                    # Tabella con scommesse simulate
                    with st.expander("Mostra Dettaglio Scommesse Simulate"):
                        df_sim_display = df_sim[[colonna_data, colonna_sq_a, colonna_sq_b, colonna_quota, 'Esito_Standard', 'Stake_Simulato', 'Stake_Simulato_%Bankroll', 'P/L']].copy()
                        if len(bankroll_history[1:]) == len(df_sim_display): # Controllo lunghezza
                             df_sim_display['Bankroll_Dopo_Scommessa'] = bankroll_history[1:] # Escludi il primo valore (bankroll iniziale)
                        else: # Gestione disallineamento (es. bankroll esaurito)
                             df_sim_display['Bankroll_Dopo_Scommessa'] = bankroll_history[1:][:len(df_sim_display)]


                        # Formattazione
                        if colonna_data in df_sim_display.columns:
                            df_sim_display[colonna_data] = pd.to_datetime(df_sim_display[colonna_data], errors='coerce').dt.strftime('%Y-%m-%d')
                        df_sim_display['Stake_Simulato'] = df_sim_display['Stake_Simulato'].apply(format_currency)
                        df_sim_display['Stake_Simulato_%Bankroll'] = df_sim_display['Stake_Simulato_%Bankroll'].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                        df_sim_display['P/L'] = pd.to_numeric(df_sim_display['P/L'], errors='coerce').apply(format_currency) # P/L originale della scommessa
                        df_sim_display['Bankroll_Dopo_Scommessa'] = pd.to_numeric(df_sim_display['Bankroll_Dopo_Scommessa'], errors='coerce').apply(format_currency)

                        st.dataframe(df_sim_display, hide_index=True, use_container_width=True)
                elif proceed_simulation and stake_param is None:
                     st.warning("Parametri di staking non definiti correttamente. Controlla la configurazione della strategia.")
        else:
             st.info("Nessun dato disponibile per il simulatore di strategie di staking.")


        # Simulatore di regressione alla media
        st.markdown("---")
        st.markdown("##### Simulatore di Regressione alla Media (Beta)") # Titolo più specifico

        with st.container(border=True):
            st.markdown("""
            Questo simulatore stima come la performance (ROI) potrebbe evolvere nel futuro, considerando la varianza e la tendenza a regredire verso una media a lungo termine.
            **Nota**: Questa è una simulazione semplificata e i risultati sono indicativi.
            """)

            col1_regr, col2_regr = st.columns(2) # Rinomina
            
            # Parametri di input
            # Rendi i max_value dei slider dinamici in base al numero di scommesse filtrate
            max_sim_val = len(df_results_filtered_final) if not df_results_filtered_final.empty else 100
            default_future_bets = min(100, max_sim_val if max_sim_val > 0 else 100)


            num_simulations_regr = col1_regr.slider("Numero Simulazioni Monte Carlo", min_value=100, max_value=5000, value=1000, step=100, key="num_sim_regr") 
            future_bets_regr = col2_regr.slider("Scommesse Future per Simulazione", min_value=10, max_value=max(10, max_sim_val), value=default_future_bets, step=10, key="future_bets_regr")


            if st.button("Esegui Simulazione di Regressione"):
                if not df_results_filtered_final.empty and win_rate_wl is not None and pd.notna(win_rate_wl) and roi_wl is not None and pd.notna(roi_wl):
                    # Statistiche attuali dai dati filtrati
                    current_win_rate_regr = win_rate_wl 
                    current_roi_regr = roi_wl / 100  # Converti ROI da % a frazione 

                    # Estrai le quote delle scommesse vinte e perse dai dati filtrati
                    winning_bets_filtered = df_results_filtered_final[df_results_filtered_final['Esito_Standard'] == 'Win']
                    losing_bets_filtered = df_results_filtered_final[df_results_filtered_final['Esito_Standard'] == 'Loss']

                    if not winning_bets_filtered.empty and colonna_quota in winning_bets_filtered.columns and winning_bets_filtered[colonna_quota].notna().any():
                        # Per le vincite, il P/L per unità di stake è (quota - 1)
                        # Usiamo le quote reali delle vincite per campionare il profitto
                        unit_profit_from_wins = (winning_bets_filtered[colonna_quota] - 1).dropna().values
                        if len(unit_profit_from_wins) == 0: unit_profit_from_wins = np.array([current_roi_regr / current_win_rate_regr if current_win_rate_regr > 0 and pd.notna(current_win_rate_regr) else 0]) # Fallback
                    else: # Fallback se non ci sono vincite o la colonna quota manca
                        avg_profit_per_win_unit_stake = current_roi_regr / current_win_rate_regr if current_win_rate_regr > 0 and pd.notna(current_win_rate_regr) else 0
                        unit_profit_from_wins = np.array([avg_profit_per_win_unit_stake])


                    # Per le perdite, il P/L per unità di stake è -1
                    # Non serve campionare, la perdita unitaria è sempre -1

                    # Simulazione Monte Carlo
                    simulation_roi_results = [] 

                    for _ in range(num_simulations_regr):
                        sim_pl_total = 0
                        # Assumiamo uno stake unitario (1€) per ogni scommessa futura per calcolare il ROI
                        sim_total_stake_normalized = future_bets_regr

                        for _ in range(future_bets_regr):
                            # Determina se la scommessa è vinta in base al win rate attuale
                            if np.random.random() < current_win_rate_regr:
                                # Scommessa vinta - campiona un profitto unitario dalla distribuzione dei profitti unitari delle vincite
                                profit_unit = np.random.choice(unit_profit_from_wins) if len(unit_profit_from_wins) > 0 else 0
                                sim_pl_total += profit_unit
                            else:
                                # Scommessa persa - perdita unitaria
                                sim_pl_total -= 1

                        # Calcola ROI della simulazione
                        sim_roi = sim_pl_total / sim_total_stake_normalized if sim_total_stake_normalized > 0 else 0
                        simulation_roi_results.append(sim_roi)


                    # Analisi dei risultati
                    simulation_roi_results_np = np.array(simulation_roi_results) 
                    avg_future_roi_regr = np.mean(simulation_roi_results_np) 
                    median_future_roi_regr = np.median(simulation_roi_results_np) 
                    std_future_roi_regr = np.std(simulation_roi_results_np) 
                    percentile_5_regr = np.percentile(simulation_roi_results_np, 5) 
                    percentile_95_regr = np.percentile(simulation_roi_results_np, 95) 


                    # Visualizzazione risultati
                    res_col1_regr, res_col2_regr, res_col3_regr, res_col4_regr = st.columns(4) 
                    res_col1_regr.metric("ROI Attuale (dati filtrati)", f"{current_roi_regr:.2%}")
                    res_col2_regr.metric("ROI Futuro Stimato (Media Sim.)", f"{avg_future_roi_regr:.2%}",
                                      f"{avg_future_roi_regr - current_roi_regr:+.2%}")
                    res_col3_regr.metric("ROI Futuro (Mediana Sim.)", f"{median_future_roi_regr:.2%}")
                    res_col4_regr.metric("Dev. Std. ROI Sim.", f"{std_future_roi_regr:.2%}")


                    # Visualizzazione distribuzione
                    fig_dist_regr = go.Figure() 

                    # Istogramma dei risultati
                    fig_dist_regr.add_trace(go.Histogram(
                        x=simulation_roi_results_np,
                        histnorm='probability density',
                        name='Distribuzione ROI Simulato',
                        marker_color='#4682B4',
                        opacity=0.7
                    ))

                    # Linee verticali per i percentili e medie
                    fig_dist_regr.add_vline(x=avg_future_roi_regr, line_dash="solid", line_color="black",
                                      annotation_text="Media Sim.", annotation_position="top right")
                    fig_dist_regr.add_vline(x=current_roi_regr, line_dash="dash", line_color="red",
                                      annotation_text="ROI Attuale", annotation_position="top left")
                    fig_dist_regr.add_vline(x=percentile_5_regr, line_dash="dot", line_color="orange",
                                      annotation_text="5° percentile", annotation_position="bottom left")
                    fig_dist_regr.add_vline(x=percentile_95_regr, line_dash="dot", line_color="green",
                                      annotation_text="95° percentile", annotation_position="bottom right")


                    fig_dist_regr.update_layout(
                        title=f"Distribuzione ROI Simulato ({future_bets_regr} scommesse future, {num_simulations_regr} simulazioni)",
                        xaxis_title="ROI (Return on Investment)",
                        yaxis_title="Densità di Probabilità",
                        template='plotly_white',
                        xaxis=dict(tickformat='.1%')
                    )

                    st.plotly_chart(fig_dist_regr, use_container_width=True)

                    # Interpretazione
                    st.markdown("##### Interpretazione dei Risultati della Simulazione")
                    st.markdown(f"""
                    - **Intervallo di Confidenza al 90%**: Basandosi su questa simulazione, c'è una probabilità del 90% che il ROI futuro (per le prossime {future_bets_regr} scommesse) si collochi tra **{percentile_5_regr:.2%}** e **{percentile_95_regr:.2%}**.
                    - **Probabilità di ROI Positivo (Simulato)**: {(simulation_roi_results_np > 0).mean():.1%}
                    - **Probabilità di Mantenere ROI > 5% (Simulato)**: {(simulation_roi_results_np > 0.05).mean():.1%}
                    - **Probabilità di Mantenere ROI > 10% (Simulato)**: {(simulation_roi_results_np > 0.10).mean():.1%}

                    La differenza tra il ROI attuale ({current_roi_regr:.2%}) e il ROI futuro medio stimato ({avg_future_roi_regr:.2%}) può indicare una potenziale regressione (o progressione) verso la media attesa a lungo termine, data la varianza osservata.
                    """)
                else:
                    st.warning("Dati filtrati insufficienti o non validi (ROI, Win Rate) per eseguire la simulazione di regressione.")
        # --- Fine Simulatore di Regressione ---


        # --- Tabella Dati Filtrati ---
        st.markdown("---")
        st.subheader("Tabella Dati Completa (Filtrata e Formattata)")
        df_display_filtered = df_filtered_final.copy()
        def safe_format(value, format_str): return format(value, format_str) if pd.notna(value) else ""
        cols_to_format_percent = [colonna_prob, colonna_edge, 'EV']
        cols_to_format_currency_in_table = [colonna_stake, 'P/L']
        cols_to_format_points = [colonna_media_pt_stimati, colonna_ris_finale, colonna_confidenza, 'Errore_Sovrastima_PT']
        for col in cols_to_format_percent:
             if col in df_display_filtered.columns: df_display_filtered[col] = pd.to_numeric(df_display_filtered[col], errors='coerce').map(lambda x: safe_format(x, '.1%'))
        for col in cols_to_format_currency_in_table:
             if col in df_display_filtered.columns:
                 df_display_filtered[col] = pd.to_numeric(df_display_filtered[col], errors='coerce').apply(format_currency)
        for col in cols_to_format_points:
             if col in df_display_filtered.columns: df_display_filtered[col] = pd.to_numeric(df_display_filtered[col], errors='coerce').map(lambda x: safe_format(x, '.1f'))
        if colonna_quota in df_display_filtered.columns: df_display_filtered[colonna_quota] = pd.to_numeric(df_display_filtered[colonna_quota], errors='coerce').map(lambda x: safe_format(x, '.2f'))
        if 'Quota_BE' in df_display_filtered.columns: df_display_filtered['Quota_BE'] = pd.to_numeric(df_display_filtered['Quota_BE'], errors='coerce').map(lambda x: safe_format(x, '.2f'))
        if 'Kelly_Stake' in df_display_filtered.columns: df_display_filtered['Kelly_Stake'] = pd.to_numeric(df_display_filtered['Kelly_Stake'], errors='coerce').map(lambda x: safe_format(x, '.1%'))
        if colonna_data in df_display_filtered.columns:
            if pd.api.types.is_datetime64_any_dtype(df_display_filtered[colonna_data]): df_display_filtered[colonna_data] = df_display_filtered[colonna_data].dt.strftime('%Y-%m-%d')
            else: df_display_filtered[colonna_data] = pd.to_datetime(df_display_filtered[colonna_data], errors='coerce').dt.strftime('%Y-%m-%d')

        colonne_display_ordine = [
            colonna_sq_a, colonna_sq_b, colonna_data, colonna_ora, colonna_quota, 'Quota_BE',
            colonna_stake, colonna_prob, colonna_edge, 'EV', 'Kelly_Stake', colonna_confidenza,
            colonna_media_pt_stimati, colonna_ris_finale, 'Errore_Sovrastima_PT',
            'Esito_Standard', 'P/L'
        ]
        colonne_display_esistenti = [col for col in colonne_display_ordine if col in df_display_filtered.columns]
        st.dataframe(df_display_filtered[colonne_display_esistenti], hide_index=True, use_container_width=True)


        # --- Download Dati Puliti & Nota PDF ---
        st.markdown("---")
        st.subheader("Download ed Esportazione")
        csv_data = df_cleaned.to_csv(index=False, decimal='.', sep=',').encode('utf-8')
        st.download_button(label="💾 Scarica Dati Puliti e Calcolati (.csv)", data=csv_data, file_name='dati_scommesse_puliti_calcolati.csv', mime='text/csv', help="Scarica il dataset completo dopo la pulizia e l'aggiunta delle colonne calcolate.")
        st.markdown("---")
        st.subheader("Esportare come PDF")
        st.info("Per salvare la vista corrente della dashboard: usa File -> Stampa del tuo browser e scegli 'Salva come PDF'.")

# --- Gestione Errori Critici ---
except FileNotFoundError:
    st.error(f"ERRORE CRITICO: File '{file_path}' non trovato.")
    st.info("Assicurati che il file CSV sia nella stessa cartella dello script.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error(f"ERRORE CRITICO: Il file CSV '{file_path}' è vuoto.")
    st.stop()
except KeyError as e:
    st.error(f"ERRORE CRITICO: Colonna fondamentale non trovata nel CSV: {e}.")
    st.info(f"Controlla che il file '{file_path}' contenga tutte le colonne necessarie definite all'inizio dello script (es. {colonna_sq_a}, {colonna_esito}, ecc.).")
    if 'df' in locals() and hasattr(df, 'columns'): # Verifica se df esiste e ha l'attributo columns
        st.info(f"Colonne effettivamente lette dal CSV: {df.columns.tolist()}")
    st.stop()

except Exception as e:
    st.error(f"ERRORE IMPREVISTO NELL'APPLICAZIONE: {e}")
    st.error("Si è verificato un errore durante l'esecuzione dello script. Controlla i log o prova a ricaricare.")
    # Per il debug, puoi decommentare la riga seguente per vedere il traceback completo
    # st.exception(e)
    st.stop() # Ferma l'app in caso di errore imprevisto grave
