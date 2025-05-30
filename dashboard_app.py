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

    for ita, eng in italian_to_english_month.items():
        if ita.lower() in cleaned_date_str.lower():
            try:
                match = re.search(ita, cleaned_date_str, re.IGNORECASE)
                if match:
                    cleaned_date_str = cleaned_date_str.replace(match.group(0), eng)
                    break
            except Exception:
                cleaned_date_str = cleaned_date_str.replace(ita, eng).replace(ita.lower(), eng.lower()).replace(ita.title(), eng.title())
                break

    try:
        return pd.to_datetime(cleaned_date_str, format='%d %b %Y', errors='raise')
    except ValueError:
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
file_path = "risultati_bet.csv" # Assicurati che questo file sia presente
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
    df_cleaned[colonna_data] = df_cleaned[colonna_data].apply(parse_italian_date)
    invalid_dates_count = df_cleaned[colonna_data].isna().sum()
    if invalid_dates_count > 0:
        st.warning(f"Attenzione: {invalid_dates_count} righe hanno una data non valida dopo il parsing e potrebbero essere escluse dalle analisi temporali.")

    cols_to_clean_numeric = [colonna_quota, colonna_stake, colonna_prob, colonna_edge, colonna_media_pt_stimati, colonna_confidenza, colonna_ris_finale]
    for col in cols_to_clean_numeric:
        if col in df_cleaned.columns: df_cleaned[col] = df_cleaned[col].apply(clean_numeric_value)
    for col in [colonna_prob, colonna_edge]:
         if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
              if df_cleaned[col].max(skipna=True) > 1 and col == colonna_prob :
                  df_cleaned[col] = df_cleaned[col] / 100.0
              elif df_cleaned[col].max(skipna=True) > 1 and col == colonna_edge and (df_cleaned[col] > 1).any():
                  df_cleaned[col] = df_cleaned[col] / 100.0

    esito_map = {'W': 'Win', 'L': 'Loss', 'TBD': 'TBD'}
    df_cleaned['Esito_Standard'] = df_cleaned[colonna_esito].astype(str).str.strip().str.upper().map(esito_map).fillna('Unknown')

    # --- Calcoli Derivati ---
    df_cleaned['P/L'] = df_cleaned.apply(lambda row: calculate_pl(row, stake_col=colonna_stake, quota_col=colonna_quota, esito_col='Esito_Standard'), axis=1)

    df_cleaned['Errore_Sovrastima_PT'] = np.nan
    if colonna_media_pt_stimati in df_cleaned.columns and colonna_ris_finale in df_cleaned.columns:
        valid_error_calc = df_cleaned[colonna_media_pt_stimati].notna() & df_cleaned[colonna_ris_finale].notna()
        diff = df_cleaned.loc[valid_error_calc, colonna_media_pt_stimati] - df_cleaned.loc[valid_error_calc, colonna_ris_finale]
        df_cleaned.loc[valid_error_calc, 'Errore_Sovrastima_PT'] = diff.clip(lower=0)

    if colonna_prob in df_cleaned.columns and colonna_quota in df_cleaned.columns:
        df_cleaned['EV'] = df_cleaned.apply(lambda row: calculate_expected_value(row[colonna_prob], row[colonna_quota]), axis=1)
        df_cleaned['Quota_BE'] = df_cleaned[colonna_prob].apply(lambda x: 1/x if pd.notna(x) and x > 0 else np.nan)
        df_cleaned['Kelly_Stake'] = df_cleaned.apply(lambda row: calculate_kelly_stake(row[colonna_prob], row[colonna_quota]), axis=1)

    # --- Preparazione Dati Globali (pre-filtro sidebar) ---
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
    unique_outcomes = df_cleaned['Esito_Standard'].unique()
    options_outcomes = unique_outcomes
    selected_outcomes = st.sidebar.multiselect("Filtra per Esito", options=options_outcomes, default=list(options_outcomes))
    start_date, end_date = None, None
    if colonna_data in df_cleaned_valid_dates.columns and not df_cleaned_valid_dates.empty and df_cleaned_valid_dates[colonna_data].notna().any():
        min_date_dt = df_cleaned_valid_dates[colonna_data].dropna().min()
        max_date_dt = df_cleaned_valid_dates[colonna_data].dropna().max()
        if pd.NaT not in [min_date_dt, max_date_dt] and min_date_dt.date() != max_date_dt.date():
            min_date = min_date_dt.date()
            max_date = max_date_dt.date()
            selected_date_range = st.sidebar.date_input(f"Filtra per {colonna_data}", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if len(selected_date_range) == 2: start_date, end_date = selected_date_range
            elif len(selected_date_range) == 1: start_date = end_date = selected_date_range[0]
        elif pd.NaT not in [min_date_dt, max_date_dt]:
            single_date = min_date_dt.date()
            st.sidebar.info(f"Tutti i dati validi sono relativi al {single_date.strftime('%d/%m/%Y')}")
            start_date = end_date = single_date
        else:
             st.sidebar.info("Nessuna data valida disponibile per il filtro.")
    else:
        st.sidebar.info("Nessuna data disponibile per il filtro temporale.")

    # --- Applicazione Filtri ---
    df_filtered_final = df_cleaned_valid_dates.copy()

    if start_date and end_date:
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        df_filtered_final = df_filtered_final[
            (df_filtered_final[colonna_data] >= start_date_dt) &
            (df_filtered_final[colonna_data] < end_date_dt)
        ]
        df_results_filtered_final = df_results_global[
            (df_results_global[colonna_data] >= start_date_dt) &
            (df_results_global[colonna_data] < end_date_dt)
        ].copy()
    else:
        df_results_filtered_final = df_results_global.copy()

    if selected_outcomes:
        df_filtered_final = df_filtered_final[df_filtered_final['Esito_Standard'].isin(selected_outcomes)]
        df_results_filtered_final = df_results_filtered_final[df_results_filtered_final['Esito_Standard'].isin(selected_outcomes)]

    if not df_results_filtered_final.empty:
         df_results_filtered_final = df_results_filtered_final.sort_values(by=colonna_data) # Assicura l'ordinamento per data
         df_results_filtered_final['Cumulative P/L Filtered'] = df_results_filtered_final['P/L'].fillna(0).cumsum()
    else:
        if 'Cumulative P/L Filtered' not in df_results_filtered_final.columns:
            df_results_filtered_final['Cumulative P/L Filtered'] = np.nan

    # --- Visualizzazione Dashboard ---
    if df_filtered_final.empty:
        st.warning("Nessun dato corrisponde ai filtri selezionati.")
    else:
        st.subheader("Metriche Chiave (Periodo Filtrato)")
        total_bets_wl = 0; total_stake_wl = 0; total_pl_wl = 0; roi_wl = 0; win_rate_wl = 0
        avg_stake_wl = np.nan; most_winning_quota_data = None;
        avg_overestimation_error = np.nan
        sharpe_ratio_daily = np.nan; var_95_daily = np.nan; avg_ev = np.nan; avg_be_quota = np.nan
        std_dev_score_diff_loss = np.nan
        std_dev_score_diff_win = np.nan
        sortino_ratio = np.nan

        if not df_results_filtered_final.empty:
            total_bets_wl = len(df_results_filtered_final)
            total_stake_wl = df_results_filtered_final[colonna_stake].fillna(0).sum()
            total_pl_wl = df_results_filtered_final['P/L'].fillna(0).sum()
            if total_bets_wl > 0:
                win_count_wl = (df_results_filtered_final['Esito_Standard'] == 'Win').sum()
                win_rate_wl = win_count_wl / total_bets_wl if total_bets_wl > 0 else 0
                avg_stake_wl = df_results_filtered_final[colonna_stake].mean(skipna=True)
                roi_wl = (total_pl_wl / total_stake_wl) * 100 if total_stake_wl != 0 else 0
                if win_count_wl > 0:
                    winning_odds_counts = df_results_filtered_final[(df_results_filtered_final['Esito_Standard'] == 'Win') & df_results_filtered_final[colonna_quota].notna()][colonna_quota].value_counts()
                    if not winning_odds_counts.empty:
                        quota_val = winning_odds_counts.idxmax(); wins_at_quota = winning_odds_counts.max()
                        total_bets_at_quota_df = df_results_filtered_final[df_results_filtered_final[colonna_quota] == quota_val]
                        total_bets_at_quota = len(total_bets_at_quota_df) if not total_bets_at_quota_df.empty else 0
                        win_perc_at_quota = (wins_at_quota / total_bets_at_quota) * 100 if total_bets_at_quota > 0 else 0
                        win_perc_of_total_wins = (wins_at_quota / win_count_wl) * 100 if win_count_wl > 0 else 0
                        most_winning_quota_data = {"quota": quota_val, "wins": wins_at_quota, "total_bets": total_bets_at_quota, "win_perc": win_perc_at_quota, "perc_of_total": win_perc_of_total_wins}

                if 'EV' in df_results_filtered_final.columns: avg_ev = df_results_filtered_final['EV'].mean(skipna=True)
                if 'Quota_BE' in df_results_filtered_final.columns: avg_be_quota = df_results_filtered_final['Quota_BE'].mean(skipna=True)

                if not df_results_filtered_final.empty and \
                   colonna_ris_finale in df_results_filtered_final.columns and \
                   colonna_media_pt_stimati in df_results_filtered_final.columns:
                    df_loss_scores_for_std = df_results_filtered_final[
                        (df_results_filtered_final['Esito_Standard'] == 'Loss') &
                        (df_results_filtered_final[colonna_ris_finale].notna()) &
                        (df_results_filtered_final[colonna_media_pt_stimati].notna())
                    ].copy()
                    if not df_loss_scores_for_std.empty:
                        df_loss_scores_for_std['Differenza_Punteggio_Loss'] = df_loss_scores_for_std[colonna_ris_finale] - df_loss_scores_for_std[colonna_media_pt_stimati]
                        if len(df_loss_scores_for_std['Differenza_Punteggio_Loss']) >= 2:
                            std_dev_score_diff_loss = df_loss_scores_for_std['Differenza_Punteggio_Loss'].std(ddof=1)
                        elif len(df_loss_scores_for_std['Differenza_Punteggio_Loss']) == 1: std_dev_score_diff_loss = 0.0

                    df_win_scores_for_std = df_results_filtered_final[
                        (df_results_filtered_final['Esito_Standard'] == 'Win') &
                        (df_results_filtered_final[colonna_ris_finale].notna()) &
                        (df_results_filtered_final[colonna_media_pt_stimati].notna())
                    ].copy()
                    if not df_win_scores_for_std.empty:
                        df_win_scores_for_std['Differenza_Punteggio_Win'] = df_win_scores_for_std[colonna_ris_finale] - df_win_scores_for_std[colonna_media_pt_stimati]
                        if len(df_win_scores_for_std['Differenza_Punteggio_Win']) >= 2:
                            std_dev_score_diff_win = df_win_scores_for_std['Differenza_Punteggio_Win'].std(ddof=1)
                        elif len(df_win_scores_for_std['Differenza_Punteggio_Win']) == 1: std_dev_score_diff_win = 0.0

                if start_date and end_date:
                    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                elif not df_results_filtered_final.empty and colonna_data in df_results_filtered_final and df_results_filtered_final[colonna_data].notna().any():
                    min_date_val_dt = df_results_filtered_final[colonna_data].dropna().min()
                    max_date_val_dt = df_results_filtered_final[colonna_data].dropna().max()
                    if pd.NaT not in [min_date_val_dt, max_date_val_dt]:
                        full_date_range = pd.date_range(start=min_date_val_dt.date(), end=max_date_val_dt.date(), freq='D')
                    else: full_date_range = pd.Index([])
                else: full_date_range = pd.Index([])

                if not full_date_range.empty:
                    daily_pl = df_results_filtered_final.groupby(pd.Grouper(key=colonna_data, freq='D'))['P/L'].sum()
                    all_days_pl = pd.Series(0, index=full_date_range, dtype=float)
                    all_days_pl.update(daily_pl)
                else: all_days_pl = pd.Series(dtype=float)

                if not df_results_filtered_final.empty and not all_days_pl.empty:
                    avg_stake_val = df_results_filtered_final[colonna_stake].mean()
                    initial_bankroll = avg_stake_val * 20 if pd.notna(avg_stake_val) and avg_stake_val > 0 else 1000
                    daily_returns_pct = all_days_pl / initial_bankroll
                    if len(daily_returns_pct) > 1:
                        mean_daily_return = daily_returns_pct.mean()
                        std_dev_daily_return = daily_returns_pct.std(ddof=1)
                        risk_free_daily = 0.0
                        if std_dev_daily_return > 0 and pd.notna(std_dev_daily_return):
                            sharpe_ratio_daily = (mean_daily_return - risk_free_daily) / std_dev_daily_return
                        else: sharpe_ratio_daily = np.nan
                        var_95_daily = daily_returns_pct.quantile(0.05) * initial_bankroll
                        downside_returns = daily_returns_pct[daily_returns_pct < 0]
                        if not downside_returns.empty and downside_returns.notna().any():
                            downside_deviation = np.sqrt(np.mean(downside_returns[downside_returns.notna()]**2))
                            sortino_ratio = (mean_daily_return - risk_free_daily) / downside_deviation if downside_deviation > 0 else np.nan
                        else: sortino_ratio = np.nan
                    else:
                        sharpe_ratio_daily = np.nan; var_95_daily = np.nan; sortino_ratio = np.nan
                else:
                    sharpe_ratio_daily = np.nan; var_95_daily = np.nan; sortino_ratio = np.nan

        if 'Errore_Sovrastima_PT' in df_filtered_final.columns:
             overestimations = df_filtered_final[df_filtered_final['Errore_Sovrastima_PT'] > 0]['Errore_Sovrastima_PT']
             if not overestimations.empty: avg_overestimation_error = overestimations.mean()

        col_header = st.columns([3, 1])
        with col_header[0]: st.markdown("##### Metriche Base")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Scommesse Concluse (W/L)", f"{total_bets_wl}")
        col2.metric(f"{colonna_stake} Totale (W/L)", format_currency(total_stake_wl))
        col3.metric(f"P/L Totale (W/L)", format_currency(total_pl_wl))
        col4.metric("Dev. Std. Diff. Punti (Win)", f"{std_dev_score_diff_win:.2f}" if pd.notna(std_dev_score_diff_win) else "N/D", help=f"Deviazione standard della differenza ({colonna_ris_finale} - {colonna_media_pt_stimati}) per le sole partite vinte (W) con dati di punteggio disponibili.")
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("ROI (W/L)", f"{roi_wl:.2f}%" if pd.notna(roi_wl) else "N/D")
        col6.metric("Win Rate (W/L)", f"{win_rate_wl:.1%}" if pd.notna(win_rate_wl) else "N/D")
        col7.metric(f"{colonna_stake} Medio (W/L)", format_currency(avg_stake_wl))
        col8.metric("Dev. Std. Diff. Punti (Loss)", f"{std_dev_score_diff_loss:.2f}" if pd.notna(std_dev_score_diff_loss) else "N/D", help=f"Deviazione standard della differenza ({colonna_ris_finale} - {colonna_media_pt_stimati}) per le sole partite perse (L) con dati di punteggio disponibili.")

        with col_header[0]: st.markdown("##### Metriche Strategiche Avanzate")
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        adv_col1.metric("Expected Value Medio", f"{avg_ev:.2%}" if pd.notna(avg_ev) else "N/D", help="Media dell'Expected Value (EV): (probabilità * (quota - 1)) - (1 - probabilità)")
        adv_col2.metric("Quota Break-Even Media", f"{avg_be_quota:.2f}" if pd.notna(avg_be_quota) else "N/D", help="Quota media di break-even (1/probabilità). Se la quota reale è maggiore, c'è valore atteso positivo.")
        if not df_results_filtered_final.empty and win_rate_wl is not None and pd.notna(win_rate_wl):
            avg_odds_wins = df_results_filtered_final[df_results_filtered_final['Esito_Standard'] == 'Win'][colonna_quota].mean()
            if pd.notna(avg_odds_wins) and avg_odds_wins > 1:
                kelly_fraction = ((avg_odds_wins - 1) * win_rate_wl - (1 - win_rate_wl)) / (avg_odds_wins - 1)
                kelly_percentage = max(0, kelly_fraction) * 100
                adv_col3.metric("Kelly Criterion (%)", f"{kelly_percentage:.2f}%", help="Percentuale ottimale del bankroll da puntare secondo il criterio di Kelly")
            else: adv_col3.metric("Kelly Criterion (%)", "N/D", help="Non calcolabile con i dati attuali (es. quota media vincite <= 1).")
        max_win_streak = 0
        max_loss_streak = 0
        if not df_results_filtered_final.empty:
            df_results_filtered_final['Streak_Group'] = (df_results_filtered_final['Esito_Standard'] != df_results_filtered_final['Esito_Standard'].shift(1)).cumsum()
            win_streaks = df_results_filtered_final[df_results_filtered_final['Esito_Standard'] == 'Win'].groupby('Streak_Group').size()
            loss_streaks = df_results_filtered_final[df_results_filtered_final['Esito_Standard'] == 'Loss'].groupby('Streak_Group').size()
            max_win_streak = win_streaks.max() if not win_streaks.empty else 0
            max_loss_streak = loss_streaks.max() if not loss_streaks.empty else 0
            adv_col4.metric("Max Serie Vincente", f"{max_win_streak}", help="Massimo numero di scommesse vinte consecutivamente")

        if most_winning_quota_data:
            st.markdown("---");
            with st.container(border=True):
                 st.markdown(f"#### Quota Maggiormente Vincente: {most_winning_quota_data['quota']:.2f}")
                 q_col1, q_col2, q_col3 = st.columns(3)
                 q_col1.metric(label="Vittorie a q. Quota", value=f"{most_winning_quota_data['wins']}")
                 q_col2.metric(label="Su Tot. Giocate a Quota", value=f"{most_winning_quota_data['total_bets']} ({most_winning_quota_data['win_perc']:.1f}%)")
                 q_col3.metric(label="% sul Totale Vittorie", value=f"{most_winning_quota_data['perc_of_total']:.1f}%")

        st.markdown("---"); st.subheader("Indicatori Rischio/Performance")
        def metric_evaluation(value, ranges, na_text="N/D"):
            if pd.isna(value): return na_text
            evaluation = None; sorted_ranges = sorted(ranges, key=lambda x: x[0])
            for limit, desc, color in sorted_ranges:
                if value <= limit: evaluation = (desc, color); break
            if evaluation is None and sorted_ranges and sorted_ranges[-1][0] == float('inf'): evaluation = (sorted_ranges[-1][1], sorted_ranges[-1][2])
            return f"{value:.2f} <span style='color:{evaluation[1]}; font-weight:bold;'>({evaluation[0]})</span>" if evaluation else f"{value:.2f}"
        def create_metric_help(description, ranges, interpretation=""):
            tooltip = f"{description}\n\nValori interpretativi:"
            sorted_ranges = sorted(ranges, key=lambda x: x[0])
            for i, (limit, desc, _) in enumerate(sorted_ranges):
                prefix = f"≤ {limit:.2f}" if i == 0 else (f"> {sorted_ranges[i-1][0]:.2f}" if limit == float('inf') else f"{sorted_ranges[i-1][0]:.2f} < X ≤ {limit:.2f}")
                tooltip += f"\n- {desc}: {prefix}"
            if interpretation: tooltip += f"\n\n{interpretation}"
            return tooltip

        with st.container(border=True):
            r_col1, r_col2 = st.columns(2)
            simulated_peak_bankroll_for_dd = df_results_global['Cumulative P/L'].max() if not df_results_global.empty and 'Cumulative P/L' in df_results_global and df_results_global['Cumulative P/L'].notna().any() else (total_stake_wl * 5 if total_stake_wl > 0 else 1)
            if pd.isna(simulated_peak_bankroll_for_dd) or simulated_peak_bankroll_for_dd <=0 : simulated_peak_bankroll_for_dd = 1
            max_dd_perc = (max_drawdown_global / simulated_peak_bankroll_for_dd) * 100 if pd.notna(max_drawdown_global) and simulated_peak_bankroll_for_dd > 0 else np.nan
            drawdown_ranges = [(5, "Eccellente", "#00CC00"), (10, "Buono", "#88CC00"), (20, "Accettabile", "#CCCC00"), (30, "Problematico", "#CC8800"), (float('inf'), "Critico", "#CC0000")]
            dd_help = create_metric_help("Massima perdita storica dal picco precedente (calcolata su tutti i dati). Percentuale rispetto al picco massimo del P/L cumulativo.", drawdown_ranges, "Un drawdown basso indica un sistema stabile con perdite controllate.")
            max_drawdown_display_perc = f" ({max_dd_perc:.1f}% del picco P/L)" if pd.notna(max_dd_perc) else ""
            r_col1.metric("Max Drawdown Storico", f"{format_currency(max_drawdown_global)}{max_drawdown_display_perc}", help=dd_help)

            sharpe_ranges = [(0, "Negativo", "#CC0000"), (0.5, "Scarso", "#CC8800"), (1.0, "Accettabile", "#CCCC00"), (2.0, "Buono", "#88CC00"), (float('inf'), "Eccellente", "#00CC00")]
            sharpe_help = create_metric_help("Misura il rendimento aggiustato per il rischio (basato su rendimenti giornalieri). Più alto è, meglio è.", sharpe_ranges, "Un Sharpe Ratio > 1 è generalmente considerato buono.")
            sharpe_text_eval = metric_evaluation(sharpe_ratio_daily, sharpe_ranges)
            r_col2.markdown(f"**Sharpe Ratio (Giornaliero)**<br>{sharpe_text_eval}", unsafe_allow_html=True); r_col2.caption(sharpe_help)

            r_col3, r_col4 = st.columns(2)
            sortino_ranges = [(0, "Negativo", "#CC0000"), (0.75, "Scarso", "#CC8800"), (1.5, "Accettabile", "#CCCC00"), (2.5, "Buono", "#88CC00"), (float('inf'), "Eccellente", "#00CC00")]
            sortino_help = create_metric_help("Simile allo Sharpe, ma considera solo la volatilità negativa (rischio di perdite).", sortino_ranges, "Un Sortino Ratio elevato indica buon rendimento con basso rischio di perdite significative.")
            sortino_text_eval = metric_evaluation(sortino_ratio, sortino_ranges)
            r_col3.markdown(f"**Sortino Ratio**<br>{sortino_text_eval}", unsafe_allow_html=True); r_col3.caption(sortino_help)

            avg_stake_for_var = avg_stake_wl if pd.notna(avg_stake_wl) and avg_stake_wl > 0 else 1
            var_as_perc_of_avg_stake = abs(var_95_daily) / avg_stake_for_var * 100 if pd.notna(var_95_daily) and avg_stake_for_var > 0 else np.nan
            var_ranges = [(50, "Eccellente", "#00CC00"), (100, "Buono", "#88CC00"), (200, "Accettabile", "#CCCC00"), (300, "Problematico", "#CC8800"), (float('inf'), "Critico", "#CC0000")]
            var_help = create_metric_help("Perdita giornaliera massima attesa nel 5% dei casi peggiori (basata sui dati filtrati).", var_ranges, "Un VaR basso (come % dello stake medio) indica un rischio limitato di perdite estreme in un singolo giorno.")
            var_display_perc = f" ({var_as_perc_of_avg_stake:.0f}% stake medio)" if pd.notna(var_as_perc_of_avg_stake) else ""
            r_col4.metric("VaR 95% (Giornaliero)", f"{format_currency(var_95_daily)}{var_display_perc}", help=var_help)

            # Seconda riga di indicatori
            r_col5, r_col6, r_col7, r_col8 = st.columns(4)

            # --- INIZIO MODIFICA: Calcolo durata massima del drawdown in numero di scommesse ---
            if not df_results_filtered_final.empty:
                df_dd = df_results_filtered_final.copy()
                df_dd['P/L'] = pd.to_numeric(df_dd['P/L'], errors='coerce').fillna(0)
                df_dd['Cumulative P/L'] = df_dd['P/L'].cumsum()
                running_max_dd = df_dd['Cumulative P/L'].cummax()
                df_dd['Drawdown'] = running_max_dd - df_dd['Cumulative P/L']
                df_dd['In_Drawdown'] = df_dd['Drawdown'] > 0
                df_dd['Drawdown_Group'] = (df_dd['In_Drawdown'] != df_dd['In_Drawdown'].shift(1)).cumsum()

                drawdown_groups = df_dd[df_dd['In_Drawdown']].groupby('Drawdown_Group')
                if not drawdown_groups.groups:
                    max_drawdown_duration_bets = 0
                else:
                    drawdown_durations_bets = []
                    for _, group_df in drawdown_groups:
                        duration_in_bets = len(group_df) # Numero di scommesse nel drawdown
                        drawdown_durations_bets.append(duration_in_bets)
                    max_drawdown_duration_bets = max(drawdown_durations_bets) if drawdown_durations_bets else 0

                r_col5.metric("Max Durata Drawdown", f"{max_drawdown_duration_bets} scommesse",
                             help="Durata massima (in numero di scommesse consecutive) di un periodo continuo di perdita dal precedente picco (basato sui dati filtrati).")
            else:
                r_col5.metric("Max Durata Drawdown", "N/D", help="Nessun dato per calcolare la durata del drawdown.")
            # --- FINE MODIFICA ---

            r_col6.metric("Max Serie Perdente", f"{max_loss_streak}", help="Massimo numero di scommesse perse consecutivamente (basato sui dati filtrati).")

            if not df_results_filtered_final.empty:
                 avg_win = df_results_filtered_final[df_results_filtered_final['P/L'] > 0]['P/L'].mean()
                 avg_loss_abs_val = abs(df_results_filtered_final[df_results_filtered_final['P/L'] < 0]['P/L'].mean())
                 win_loss_ratio = avg_win / avg_loss_abs_val if pd.notna(avg_loss_abs_val) and avg_loss_abs_val != 0 and pd.notna(avg_win) else np.nan
                 r_col7.metric("Win/Loss Ratio", f"{win_loss_ratio:.2f}" if pd.notna(win_loss_ratio) else "N/D",
                              help="Rapporto tra media delle vincite e valore assoluto medio delle perdite.")
                 if pd.notna(win_rate_wl) and win_rate_wl < 1 and win_rate_wl > 0:
                     calvert_ratio = win_rate_wl / (1 - win_rate_wl)
                     r_col8.metric("Calvert Ratio", f"{calvert_ratio:.2f}",
                                  help="Rapporto tra probabilità di vincita e probabilità di perdita. Valori > 1 indicano che si vince più spesso di quanto si perde.")
                 else:
                     r_col8.metric("Calvert Ratio", "N/D", help="Non calcolabile (Win Rate è 0%, 100%, o N/D).")
            else: # Caso in cui df_results_filtered_final è vuoto per le ultime due metriche
                 r_col7.metric("Win/Loss Ratio", "N/D", help="Nessun dato per calcolare il Win/Loss Ratio.")
                 r_col8.metric("Calvert Ratio", "N/D", help="Nessun dato per calcolare il Calvert Ratio.")


        st.markdown("---"); st.subheader("Visualizzazioni Grafiche")
        st.markdown("##### Distribuzione Esiti (Win/Loss)")
        if not df_results_filtered_final.empty:
            outcome_counts = df_results_filtered_final['Esito_Standard'].value_counts().reset_index(); outcome_counts.columns = ['Esito_Standard', 'Conteggio']
            fig_outcome_pie = px.pie(outcome_counts, names='Esito_Standard', values='Conteggio', title="Distribuzione Esiti Win/Loss (Periodo Filtrato)", hole=0.3, color='Esito_Standard', color_discrete_map=color_map_esiti)
            fig_outcome_pie.update_traces(marker_line_width=0); fig_outcome_pie.update_layout(template='plotly_white', legend_title_text='Esito')
            st.plotly_chart(fig_outcome_pie, use_container_width=True); st.caption("Mostra la proporzione tra scommesse vinte e perse nel periodo filtrato.")
        else: st.info("Nessun dato W/L disponibile per i filtri selezionati.")
        st.markdown("---")

        st.subheader("Andamento Cumulativo P/L nel Tempo (Periodo Filtrato)")
        if not df_results_filtered_final.empty and 'Cumulative P/L Filtered' in df_results_filtered_final.columns and df_results_filtered_final['Cumulative P/L Filtered'].notna().any():
            fig_cum_pl = px.line(df_results_filtered_final, x=colonna_data, y='Cumulative P/L Filtered', title="Andamento P/L Cumulativo (€) - Periodo Filtrato", markers=False)
            fig_cum_pl.update_traces(line=dict(color=color_line2, shape='linear')); fig_cum_pl.update_layout(template='plotly_white')
            yaxis_title_cum_pl = "P/L Cumulativo (€)"; tickformat_cum_pl = "~€" if italian_locale_set else None; ticksuffix_cum_pl = None if italian_locale_set else " €"
            fig_cum_pl.update_yaxes(title=yaxis_title_cum_pl, showgrid=False, tickformat=tickformat_cum_pl, ticksuffix=ticksuffix_cum_pl)
            fig_cum_pl.update_xaxes(title="Data", showgrid=False)
            st.plotly_chart(fig_cum_pl, use_container_width=True); st.caption("Mostra l'evoluzione del P/L totale nel tempo per il periodo selezionato.")
        else: st.info("Nessun dato P/L per grafico cumulativo filtrato.")
        st.markdown("---")

        st.subheader("Andamento Giornaliero P/L e Volume Scommesse (Periodo Filtrato)")
        if not df_results_filtered_final.empty:
            daily_summary_chart = df_results_filtered_final.groupby(pd.Grouper(key=colonna_data, freq='D')).agg(Daily_PL=('P/L', 'sum'), Num_Bets=(colonna_esito, 'size')).reset_index()
            daily_summary_chart = daily_summary_chart[daily_summary_chart['Num_Bets'] > 0]
            if not daily_summary_chart.empty:
                fig_daily = make_subplots(specs=[[{"secondary_y": True}]]); colors_pl = [color_win if pl >= 0 else color_loss for pl in daily_summary_chart['Daily_PL']]
                fig_daily.add_trace(go.Bar(x=daily_summary_chart[colonna_data], y=daily_summary_chart['Daily_PL'], name='P/L Giornaliero (€)', marker_color=colors_pl, marker_line_width=0, yaxis='y1'), secondary_y=False)
                fig_daily.add_trace(go.Scatter(x=daily_summary_chart[colonna_data], y=daily_summary_chart['Num_Bets'], name='Num. Scommesse', mode='lines+markers', line=dict(color=color_line1, shape='linear'), marker=dict(size=5), yaxis='y2'), secondary_y=True)
                fig_daily.update_layout(title_text="Risultato Netto (€) e Num. Scommesse per Giorno (Periodo Filtrato)", xaxis_title="Data", legend_title="Legenda", barmode='relative', template='plotly_white')
                yaxis_title_daily = "P/L Netto Giornaliero (€)"; tickformat_daily = "~€" if italian_locale_set else None; ticksuffix_daily = None if italian_locale_set else " €"
                fig_daily.update_yaxes(title_text=yaxis_title_daily, tickformat=tickformat_daily, ticksuffix=ticksuffix_daily, secondary_y=False, showgrid=False)
                fig_daily.update_yaxes(title_text="Numero Scommesse", secondary_y=True, showgrid=False); fig_daily.update_xaxes(showgrid=False)
                st.plotly_chart(fig_daily, use_container_width=True); st.caption("Mostra il P/L netto giornaliero e il numero di scommesse concluse.")
            else: st.info("Nessun dato giornaliero aggregato.")
        else: st.info("Nessun dato W/L per grafico P/L giornaliero.")
        st.markdown("---")

        st.subheader(f"Relazione tra Errore Sovrastima Punti e {colonna_prob}")
        if colonna_prob in df_results_filtered_final.columns and 'Errore_Sovrastima_PT' in df_results_filtered_final.columns:
            df_plot_prob_error = df_results_filtered_final[df_results_filtered_final[colonna_prob].notna() & df_results_filtered_final['Errore_Sovrastima_PT'].notna()].copy()
            if not df_plot_prob_error.empty:
                 df_plot_prob_error['Prob Formattata'] = df_plot_prob_error[colonna_prob].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
                 df_plot_prob_error['Errore Sovrastima Formattato'] = df_plot_prob_error['Errore_Sovrastima_PT'].map(lambda x: f"{x:.1f}" if pd.notna(x) else 'N/A')
                 fig_prob_error = px.scatter(df_plot_prob_error, x=colonna_prob, y='Errore_Sovrastima_PT', color='Esito_Standard', title=f"Errore Sovrastima Punti vs. {colonna_prob} (Periodo Filtrato)", labels={ colonna_prob: 'Probabilità Stimata', 'Errore_Sovrastima_PT': 'Errore Sovrastima Punti', 'Esito_Standard': 'Esito' }, hover_name=colonna_sq_a, hover_data={ colonna_sq_b: True, 'Prob Formattata': True, 'Errore Sovrastima Formattato': True, colonna_prob: False, 'Errore_Sovrastima_PT': False }, color_discrete_map=color_map_esiti)
                 fig_prob_error.update_traces(marker=dict(line=dict(width=0))); fig_prob_error.update_layout(template='plotly_white')
                 fig_prob_error.update_xaxes(title="Probabilità Stimata", showgrid=False, tickformat=".1%"); fig_prob_error.update_yaxes(title="Errore Sovrastima Punti", showgrid=False)
                 for val_x_line in [0.60, 0.65, 0.70]: fig_prob_error.add_vline(x=val_x_line, line_dash="dash", line_color="gray")
                 st.plotly_chart(fig_prob_error, use_container_width=True); st.caption("Mostra la relazione tra la sovrastima dei punti (Media Stimati - Punteggio Finale, solo se positivo) e la probabilità stimata dell'esito finale (Win/Loss). Valori Y=0 indicano stima corretta o sottostima.")
            else: st.info(f"Nessun dato valido (con Prob. Stimata ed Errore Sovrastima calcolabile) per questo grafico.")
            st.markdown("---")
        else: st.info(f"Le colonne '{colonna_prob}' o '{colonna_media_pt_stimati}'/'{colonna_ris_finale}' (per calcolare l'errore) non sono disponibili."); st.markdown("---")

        st.subheader(f"Relazione tra Errore Sovrastima Punti e {colonna_edge}")
        if colonna_edge in df_results_filtered_final.columns and 'Errore_Sovrastima_PT' in df_results_filtered_final.columns:
            df_plot_edge_error = df_results_filtered_final[df_results_filtered_final[colonna_edge].notna() & df_results_filtered_final['Errore_Sovrastima_PT'].notna()].copy()
            if not df_plot_edge_error.empty:
                 df_plot_edge_error['Edge Formattato'] = df_plot_edge_error[colonna_edge].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
                 df_plot_edge_error['Errore Sovrastima Formattato'] = df_plot_edge_error['Errore_Sovrastima_PT'].map(lambda x: f"{x:.1f}" if pd.notna(x) else 'N/A')
                 fig_edge_error = px.scatter(df_plot_edge_error, x=colonna_edge, y='Errore_Sovrastima_PT', color='Esito_Standard', title=f"Errore Sovrastima Punti vs. {colonna_edge} (Periodo Filtrato)", labels={colonna_edge: 'Edge Value', 'Errore_Sovrastima_PT': 'Errore Sovrastima Punti', 'Esito_Standard': 'Esito'}, hover_name=colonna_sq_a, hover_data={colonna_sq_b: True, 'Edge Formattato': True, 'Errore Sovrastima Formattato': True, colonna_edge: False, 'Errore_Sovrastima_PT': False}, color_discrete_map=color_map_esiti)
                 fig_edge_error.update_traces(marker=dict(line=dict(width=0))); fig_edge_error.update_layout(template='plotly_white')
                 fig_edge_error.update_xaxes(title="Edge Value", showgrid=False, tickformat=".1%"); fig_edge_error.update_yaxes(title="Errore Sovrastima Punti", showgrid=False)
                 for val_x_line in [0.10, 0.15, 0.20]: fig_edge_error.add_vline(x=val_x_line, line_dash="dash", line_color="gray")
                 st.plotly_chart(fig_edge_error, use_container_width=True); st.caption("Mostra la relazione tra la sovrastima dei punti (Media Stimati - Punteggio Finale, solo se positivo) e l'edge value. Valori Y=0 indicano stima corretta o sottostima.")
            else: st.info(f"Nessun dato valido (con Edge Value ed Errore Sovrastima calcolabile) per questo grafico.")
            st.markdown("---")
        else: st.info(f"Le colonne '{colonna_edge}' o '{colonna_media_pt_stimati}'/'{colonna_ris_finale}' (per calcolare l'errore) non sono disponibili."); st.markdown("---")

        st.subheader(f"Relazione tra {colonna_edge} e {colonna_prob}")
        if colonna_edge in df_results_filtered_final.columns and colonna_prob in df_results_filtered_final.columns:
            df_plot_edge_prob = df_results_filtered_final[df_results_filtered_final[colonna_edge].notna() & df_results_filtered_final[colonna_prob].notna() & df_results_filtered_final['Esito_Standard'].isin(['Win', 'Loss'])].copy()
            if not df_plot_edge_prob.empty:
                df_plot_edge_prob['Edge Formattato'] = df_plot_edge_prob[colonna_edge].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
                df_plot_edge_prob['Prob Formattata'] = df_plot_edge_prob[colonna_prob].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
                fig_edge_prob = px.scatter(df_plot_edge_prob, x=colonna_edge, y=colonna_prob, color='Esito_Standard', title=f"{colonna_edge} vs. {colonna_prob} (Periodo Filtrato)", labels={colonna_edge: 'Edge Value', colonna_prob: 'Probabilità Stimata', 'Esito_Standard': 'Esito'}, hover_name=colonna_sq_a, hover_data={colonna_sq_b: True, 'Edge Formattato': True, 'Prob Formattata': True, colonna_quota: ':.2f', colonna_edge: False, colonna_prob: False}, color_discrete_map=color_map_esiti)
                fig_edge_prob.update_traces(marker=dict(line=dict(width=0))); fig_edge_prob.update_layout(template='plotly_white')
                fig_edge_prob.update_xaxes(title="Edge Value", showgrid=False, tickformat=".0%"); fig_edge_prob.update_yaxes(title="Probabilità Stimata", showgrid=False, tickformat=".0%")
                min_edge_val_data = df_plot_edge_prob[colonna_edge].min() if not df_plot_edge_prob.empty else -0.1; max_edge_val_data = df_plot_edge_prob[colonna_edge].max() if not df_plot_edge_prob.empty else 0.3
                if pd.notna(min_edge_val_data) and pd.notna(max_edge_val_data):
                    start_edge_line = np.ceil(min_edge_val_data / 0.05) * 0.05 if min_edge_val_data < 0 else np.floor(min_edge_val_data / 0.05) * 0.05; start_edge_line = max(start_edge_line, -0.3)
                    for val_x in np.arange(start_edge_line, max_edge_val_data + 0.05, 0.05):
                        if abs(val_x) > 0.001 : fig_edge_prob.add_vline(x=val_x, line_dash="dash", line_color="lightgray", opacity=0.7)
                min_prob_val_data = df_plot_edge_prob[colonna_prob].min() if not df_plot_edge_prob.empty else 0.4; max_prob_val_data = df_plot_edge_prob[colonna_prob].max() if not df_plot_edge_prob.empty else 1.0
                if pd.notna(min_prob_val_data) and pd.notna(max_prob_val_data):
                    start_prob_line = max(0.05, np.floor(min_prob_val_data / 0.05) * 0.05)
                    for val_y in np.arange(start_prob_line, max_prob_val_data + 0.05, 0.05):
                         if val_y > 0.001 and val_y < 0.999: fig_edge_prob.add_hline(y=val_y, line_dash="dash", line_color="lightgray", opacity=0.7)
                st.plotly_chart(fig_edge_prob, use_container_width=True); st.caption(f"Mostra la relazione tra l'{colonna_edge} e la {colonna_prob}, colorata per esito della scommessa. Le linee tratteggiate indicano intervalli del 5%.")
            else: st.info(f"Nessun dato valido (con {colonna_edge}, {colonna_prob} e esito W/L) per questo grafico.")
            st.markdown("---")
        else: st.info(f"Le colonne '{colonna_edge}' o '{colonna_prob}' non sono disponibili per questo grafico."); st.markdown("---")

        # Grafico "Analisi della Precisione di Previsione" ora basato su Edge Value
        st.subheader(f"Analisi della Precisione Reale per Range di {colonna_edge}")
        if not df_filtered_final.empty and colonna_edge in df_filtered_final.columns:
            df_analysis = df_filtered_final[
                (df_filtered_final[colonna_edge].notna()) &
                (df_filtered_final['Esito_Standard'].isin(['Win', 'Loss']))
            ].copy()

            if not df_analysis.empty:
                df_analysis[colonna_edge] = pd.to_numeric(df_analysis[colonna_edge], errors='coerce')
                df_analysis.dropna(subset=[colonna_edge], inplace=True)

                if not df_analysis.empty:
                    min_val_edge = df_analysis[colonna_edge].min(); max_val_edge = df_analysis[colonna_edge].max()
                    if pd.notna(min_val_edge) and pd.notna(max_val_edge) and min_val_edge < max_val_edge:
                        edge_bins = np.arange(np.floor(min_val_edge*20)/20, np.ceil(max_val_edge*20)/20 + 0.05, 0.05)
                        if len(edge_bins) < 2: edge_bins = np.arange(-0.20, 0.35, 0.05)
                    else: edge_bins = np.arange(-0.20, 0.35, 0.05)
                    edge_labels = [f"{edge_bins[i]*100:.0f}-{edge_bins[i+1]*100:.0f}%" for i in range(len(edge_bins)-1)]

                    if len(edge_bins) > 1 :
                        df_analysis['Value_Bin'] = pd.cut(df_analysis[colonna_edge], bins=edge_bins, labels=edge_labels, include_lowest=True, right=False)
                        accuracy_by_bin = df_analysis.groupby('Value_Bin', observed=False).agg(Scommesse_Totali=('Esito_Standard', 'size'), Vincite=('Esito_Standard', lambda x: (x == 'Win').sum()), Valore_Medio_Bin=(colonna_edge, 'mean')).reset_index()
                        accuracy_by_bin['Precisione_Reale'] = np.where(accuracy_by_bin['Scommesse_Totali'] > 0, accuracy_by_bin['Vincite'] / accuracy_by_bin['Scommesse_Totali'], 0)
                        accuracy_by_bin['Differenza_Rate_Valore'] = accuracy_by_bin['Precisione_Reale'] - accuracy_by_bin['Valore_Medio_Bin']
                        accuracy_by_bin.dropna(subset=['Valore_Medio_Bin', 'Precisione_Reale'], inplace=True)

                        if not accuracy_by_bin.empty:
                            fig_accuracy = go.Figure()
                            min_axis_val = min(accuracy_by_bin['Valore_Medio_Bin'].min(), accuracy_by_bin['Precisione_Reale'].min()) if not accuracy_by_bin.empty else -0.1
                            max_axis_val = max(accuracy_by_bin['Valore_Medio_Bin'].max(), accuracy_by_bin['Precisione_Reale'].max()) if not accuracy_by_bin.empty else 0.3
                            if pd.isna(min_axis_val): min_axis_val = -0.1;
                            if pd.isna(max_axis_val): max_axis_val = 0.3
                            fig_accuracy.add_trace(go.Scatter(x=[min_axis_val, max_axis_val], y=[min_axis_val, max_axis_val], mode='lines', name='Win Rate = Edge Medio', line=dict(color='black', width=1, dash='dash')))
                            fig_accuracy.add_hline(y=0.5, line_dash="dot", line_color="blue", opacity=0.7, annotation_text="50% Win Rate", annotation_position="bottom right")
                            sizes = accuracy_by_bin['Scommesse_Totali'].clip(lower=5) * 1.5
                            fig_accuracy.add_trace(go.Scatter(x=accuracy_by_bin['Valore_Medio_Bin'], y=accuracy_by_bin['Precisione_Reale'], mode='markers+text', marker=dict(size=sizes, color=accuracy_by_bin['Differenza_Rate_Valore'], colorscale='RdYlGn', cmid=0, cmin=min(-0.2, accuracy_by_bin['Differenza_Rate_Valore'].min() if accuracy_by_bin['Differenza_Rate_Valore'].notna().any() else -0.2), cmax=max(0.2, accuracy_by_bin['Differenza_Rate_Valore'].max() if accuracy_by_bin['Differenza_Rate_Valore'].notna().any() else 0.2), colorbar=dict(title='Win Rate - Edge Medio'), line=dict(width=1, color='black')), text=accuracy_by_bin['Value_Bin'].astype(str), textposition="top center", name='Precisione Reale per Bin di Edge', hovertemplate='<b>Bin Edge: %{text}</b><br>Edge Medio Stimato: %{x:.1%}<br>Precisione Reale (Win Rate): %{y:.1%}<br>Differenza (Rate - Edge): %{marker.color:+.1%}<br>Scommesse nel Bin: %{customdata[0]}<extra></extra>', customdata=accuracy_by_bin[['Scommesse_Totali']]))
                            x_axis_range = [min_axis_val - 0.05, max_axis_val + 0.05]; y_axis_range = [0, max(1.05, accuracy_by_bin['Precisione_Reale'].max() + 0.05 if accuracy_by_bin['Precisione_Reale'].notna().any() else 1.05)]
                            fig_accuracy.update_layout(title=f'Precisione Reale (Win Rate) vs. {colonna_edge} Medio nel Bin', xaxis_title=f'{colonna_edge} Medio Stimato nel Bin', yaxis_title='Precisione Reale (Win Rate) nel Bin', template='plotly_white', xaxis=dict(tickformat='.1%', range=x_axis_range), yaxis=dict(tickformat='.1%', range=y_axis_range))
                            st.plotly_chart(fig_accuracy, use_container_width=True); st.caption(f"Analisi della precisione: confronta l'{colonna_edge} medio stimato con la frequenza reale di vincita per diversi intervalli di Edge Value. Idealmente, per Edge Value positivi ci si aspetta un Win Rate che porti a profitto. La linea diagonale indica dove il Win Rate è numericamente uguale all'Edge Medio del bin. La linea blu indica un Win Rate del 50%.")
                            st.markdown(f"##### Dettaglio Precisione per Range di {colonna_edge}")
                            display_accuracy_data = accuracy_by_bin.copy()
                            display_accuracy_data['Valore_Medio_Bin'] = display_accuracy_data['Valore_Medio_Bin'].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                            display_accuracy_data['Precisione_Reale'] = display_accuracy_data['Precisione_Reale'].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                            display_accuracy_data['Differenza_Rate_Valore'] = display_accuracy_data['Differenza_Rate_Valore'].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")
                            display_accuracy_data.columns = [f'Range {colonna_edge}', 'Totale Scommesse', 'Vincite', f'{colonna_edge} Medio', 'Precisione Reale (Win Rate)', 'Differenza (Rate - Edge Medio)']
                            st.dataframe(display_accuracy_data, use_container_width=True, hide_index=True)
                        else: st.info(f"Nessun dato sufficiente per il grafico di analisi della precisione per {colonna_edge} dopo il binning.")
                    else: st.info(f"Impossibile creare bin significativi per l'{colonna_edge}.")
                else: st.info(f"Nessun {colonna_edge} valido per l'analisi di precisione.")
            else: st.info(f"Nessun dato con esito Win/Loss e {colonna_edge} per l'analisi di precisione.")
        else: st.info(f"La colonna '{colonna_edge}' è necessaria per questa analisi ma non è disponibile.")
        st.markdown("---")

        # --- MODIFIED GRAPH AND TABLE (BASED ON P/L PER BET) ---
        st.subheader(f"Analisi Impatto P/L Medio per Scommessa per Range di {colonna_edge}")
        if not df_filtered_final.empty and colonna_edge in df_filtered_final.columns and 'P/L' in df_filtered_final.columns:
            df_analysis_for_pl = df_filtered_final[
                (df_filtered_final[colonna_edge].notna()) &
                (df_filtered_final['P/L'].notna()) &
                (df_filtered_final['Esito_Standard'].isin(['Win', 'Loss']))
            ].copy()

            if not df_analysis_for_pl.empty:
                df_analysis_for_pl[colonna_edge] = pd.to_numeric(df_analysis_for_pl[colonna_edge], errors='coerce')
                df_analysis_for_pl.dropna(subset=[colonna_edge], inplace=True)

                if not df_analysis_for_pl.empty:
                    min_val_edge_pl = df_analysis_for_pl[colonna_edge].min()
                    max_val_edge_pl = df_analysis_for_pl[colonna_edge].max()
                    if pd.notna(min_val_edge_pl) and pd.notna(max_val_edge_pl) and min_val_edge_pl < max_val_edge_pl:
                        edge_bins_pl = np.arange(np.floor(min_val_edge_pl*20)/20, np.ceil(max_val_edge_pl*20)/20 + 0.05, 0.05)
                        if len(edge_bins_pl) < 2: edge_bins_pl = np.arange(-0.20, 0.35, 0.05)
                    else: edge_bins_pl = np.arange(-0.20, 0.35, 0.05)
                    edge_labels_pl = [f"{edge_bins_pl[i]*100:.0f}-{edge_bins_pl[i+1]*100:.0f}%" for i in range(len(edge_bins_pl)-1)]

                    if len(edge_bins_pl) > 1 :
                        df_analysis_for_pl['Value_Bin'] = pd.cut(
                            df_analysis_for_pl[colonna_edge], bins=edge_bins_pl, labels=edge_labels_pl, include_lowest=True, right=False
                        )
                        accuracy_by_bin_pl_impact = df_analysis_for_pl.groupby('Value_Bin', observed=False).agg(
                            Scommesse_Totali=('Esito_Standard', 'size'),
                            Vincite=('Esito_Standard', lambda x: (x == 'Win').sum()),
                            Valore_Medio_Bin=(colonna_edge, 'mean'),
                            PL_Totale_Bin=('P/L', 'sum')
                        ).reset_index()

                        accuracy_by_bin_pl_impact['Precisione_Reale'] = np.where(
                            accuracy_by_bin_pl_impact['Scommesse_Totali'] > 0,
                            accuracy_by_bin_pl_impact['Vincite'] / accuracy_by_bin_pl_impact['Scommesse_Totali'],
                            0
                        )
                        accuracy_by_bin_pl_impact['PL_Medio_Scommessa_Bin'] = np.where(
                            accuracy_by_bin_pl_impact['Scommesse_Totali'] > 0,
                            accuracy_by_bin_pl_impact['PL_Totale_Bin'] / accuracy_by_bin_pl_impact['Scommesse_Totali'],
                            0
                        )
                        accuracy_by_bin_pl_impact['Differenza_Rate_Valore'] = accuracy_by_bin_pl_impact['Precisione_Reale'] - accuracy_by_bin_pl_impact['Valore_Medio_Bin']
                        accuracy_by_bin_pl_impact.dropna(subset=['Valore_Medio_Bin', 'Precisione_Reale', 'PL_Totale_Bin'], inplace=True)

                        if not accuracy_by_bin_pl_impact.empty:
                            fig_accuracy_pl = go.Figure()
                            min_axis_val_pl_chart = min(accuracy_by_bin_pl_impact['Valore_Medio_Bin'].min(), accuracy_by_bin_pl_impact['Precisione_Reale'].min()) if not accuracy_by_bin_pl_impact.empty else -0.1
                            max_axis_val_pl_chart = max(accuracy_by_bin_pl_impact['Valore_Medio_Bin'].max(), accuracy_by_bin_pl_impact['Precisione_Reale'].max()) if not accuracy_by_bin_pl_impact.empty else 0.3
                            if pd.isna(min_axis_val_pl_chart): min_axis_val_pl_chart = -0.1
                            if pd.isna(max_axis_val_pl_chart): max_axis_val_pl_chart = 0.3
                            fig_accuracy_pl.add_trace(go.Scatter(x=[min_axis_val_pl_chart, max_axis_val_pl_chart], y=[min_axis_val_pl_chart, max_axis_val_pl_chart], mode='lines', name='Win Rate = Edge Medio', line=dict(color='black', width=1, dash='dash')))
                            fig_accuracy_pl.add_hline(y=0.5, line_dash="dot", line_color="blue", opacity=0.7, annotation_text="50% Win Rate", annotation_position="bottom right")

                            sizes_for_pl_chart = (accuracy_by_bin_pl_impact['PL_Medio_Scommessa_Bin'].abs().fillna(0.1) * 8.0).clip(lower=5, upper=80) # Aumentato fattore di scala e range di clip

                            custom_data_list = []
                            for _, row_data in accuracy_by_bin_pl_impact.iterrows():
                                custom_data_list.append([
                                    row_data['Scommesse_Totali'],
                                    format_currency(row_data['PL_Medio_Scommessa_Bin']),
                                    format_currency(row_data['PL_Totale_Bin'])
                                ])

                            fig_accuracy_pl.add_trace(
                                go.Scatter(
                                    x=accuracy_by_bin_pl_impact['Valore_Medio_Bin'],
                                    y=accuracy_by_bin_pl_impact['Precisione_Reale'],
                                    mode='markers+text',
                                    marker=dict(
                                        size=sizes_for_pl_chart,
                                        color=accuracy_by_bin_pl_impact['Differenza_Rate_Valore'],
                                        colorscale='RdYlGn', cmid=0,
                                        cmin=min(-0.2, accuracy_by_bin_pl_impact['Differenza_Rate_Valore'].min() if accuracy_by_bin_pl_impact['Differenza_Rate_Valore'].notna().any() else -0.2),
                                        cmax=max(0.2, accuracy_by_bin_pl_impact['Differenza_Rate_Valore'].max() if accuracy_by_bin_pl_impact['Differenza_Rate_Valore'].notna().any() else 0.2),
                                        colorbar=dict(title='Win Rate - Edge Medio'), line=dict(width=1, color='black')
                                    ),
                                    text=accuracy_by_bin_pl_impact['Value_Bin'].astype(str),
                                    textposition="top center", name='Precisione Reale per Bin di Edge (Dimensione ~ P/L Medio)',
                                    hovertemplate=(
                                        '<b>Bin Edge: %{text}</b><br>'
                                        'Edge Medio Stimato: %{x:.1%}<br>'
                                        'Precisione Reale (Win Rate): %{y:.1%}<br>'
                                        'Differenza (Rate - Edge): %{marker.color:+.1%}<br>'
                                        'Scommesse nel Bin: %{customdata[0]}<br>'
                                        'P/L Medio per Scommessa: %{customdata[1]}<br>'
                                        'P/L Totale nel Bin: %{customdata[2]}<extra></extra>'
                                    ),
                                    customdata=custom_data_list
                                )
                            )
                            x_axis_range_pl_chart = [min_axis_val_pl_chart - 0.05, max_axis_val_pl_chart + 0.05]
                            y_axis_range_pl_chart = [0, max(1.05, accuracy_by_bin_pl_impact['Precisione_Reale'].max() + 0.05 if accuracy_by_bin_pl_impact['Precisione_Reale'].notna().any() else 1.05)]
                            fig_accuracy_pl.update_layout(
                                title=f'Impatto P/L Medio per Scommessa vs. {colonna_edge} Medio (Dimensione Punti ~ P/L Medio Assoluto)',
                                xaxis_title=f'{colonna_edge} Medio Stimato nel Bin',
                                yaxis_title='Precisione Reale (Win Rate) nel Bin',
                                template='plotly_white',
                                xaxis=dict(tickformat='.1%', range=x_axis_range_pl_chart),
                                yaxis=dict(tickformat='.1%', range=y_axis_range_pl_chart)
                            )
                            st.plotly_chart(fig_accuracy_pl, use_container_width=True)
                            st.caption(f"Analisi dell'impatto P/L: confronta l'{colonna_edge} medio stimato con la frequenza reale di vincita. La dimensione dei punti è proporzionale al valore assoluto del P/L medio per scommessa in quel bin. Il colore indica la differenza tra Win Rate e Edge Medio.")

                            st.markdown(f"##### Dettaglio Impatto P/L Medio per Range di {colonna_edge}")
                            display_accuracy_pl_data = accuracy_by_bin_pl_impact.copy()

                            # Prepara le colonne formattate per la tabella
                            display_accuracy_pl_data_formatted = pd.DataFrame()
                            display_accuracy_pl_data_formatted[f'Range {colonna_edge}'] = display_accuracy_pl_data['Value_Bin'].astype(str)
                            display_accuracy_pl_data_formatted['Totale Scommesse'] = display_accuracy_pl_data['Scommesse_Totali']
                            display_accuracy_pl_data_formatted['Vincite'] = display_accuracy_pl_data['Vincite']
                            display_accuracy_pl_data_formatted[f'{colonna_edge} Medio'] = display_accuracy_pl_data['Valore_Medio_Bin'].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                            display_accuracy_pl_data_formatted['Precisione Reale (Win Rate)'] = display_accuracy_pl_data['Precisione_Reale'].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                            display_accuracy_pl_data_formatted['P/L Medio per Scommessa'] = display_accuracy_pl_data['PL_Medio_Scommessa_Bin'].apply(format_currency)
                            display_accuracy_pl_data_formatted['P/L Totale nel Bin'] = display_accuracy_pl_data['PL_Totale_Bin'].apply(format_currency)
                            display_accuracy_pl_data_formatted['Differenza (Rate - Edge Medio)'] = display_accuracy_pl_data['Differenza_Rate_Valore'].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")

                            st.dataframe(display_accuracy_pl_data_formatted, use_container_width=True, hide_index=True)

                        else: st.info(f"Nessun dato sufficiente per il grafico di analisi dell'impatto P/L Medio per {colonna_edge} dopo il binning.")
                    else: st.info(f"Impossibile creare bin significativi per l'{colonna_edge} per il grafico P/L Medio.")
                else: st.info(f"Nessun {colonna_edge} valido (dopo pulizia) per l'analisi di impatto P/L Medio.")
            else: st.info(f"Nessun dato con esito Win/Loss, {colonna_edge} e P/L per l'analisi di impatto P/L Medio.")
        else: st.info(f"Le colonne '{colonna_edge}' e/o 'P/L' sono necessarie per questa analisi ma non sono disponibili.")
        st.markdown("---")
        # --- FINE MODIFIED GRAPH AND TABLE ---

        col_c, col_d = st.columns(2)
        with col_c:
             if colonna_edge in df_results_filtered_final.columns:
                 st.subheader(f"Rapporto {colonna_edge} vs P/L (Vittoria/Perdita)")
                 df_plot_edge_pl = df_results_filtered_final[df_results_filtered_final[colonna_edge].notna()].copy()
                 if not df_plot_edge_pl.empty:
                     df_plot_edge_pl['Stake Formattato'] = df_plot_edge_pl[colonna_stake].apply(format_currency)
                     df_plot_edge_pl['P/L Formattato'] = df_plot_edge_pl['P/L'].apply(format_currency)
                     df_plot_edge_pl['Edge Formattato'] = df_plot_edge_pl[colonna_edge].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
                     fig_edge_pl = px.scatter(df_plot_edge_pl, x=colonna_edge, y='P/L', color='Esito_Standard', title="P/L vs. Edge Value (Periodo Filtrato)", labels={colonna_edge: 'Edge Value', 'P/L': 'P/L (€)', 'Esito_Standard': 'Esito'}, hover_name=colonna_sq_a, hover_data={colonna_sq_b:True, colonna_quota:':.2f', 'Stake Formattato': True, 'P/L Formattato':True, 'Edge Formattato':True, colonna_edge:False, 'P/L':False}, color_discrete_map=color_map_esiti)
                     fig_edge_pl.update_traces(marker=dict(line=dict(width=0))); fig_edge_pl.update_layout(template='plotly_white')
                     fig_edge_pl.update_xaxes(title="Edge Value", showgrid=False, tickformat=".1%")
                     yaxis_title_edge_pl = "P/L (€)"; tickformat_edge_pl = "~€" if italian_locale_set else None; ticksuffix_edge_pl = None if italian_locale_set else " €"
                     fig_edge_pl.update_yaxes(title=yaxis_title_edge_pl, showgrid=False, tickformat=tickformat_edge_pl, ticksuffix=ticksuffix_edge_pl)
                     fig_edge_pl.add_vline(x=0.20, line_dash="dash", line_color="gray")
                     st.plotly_chart(fig_edge_pl, use_container_width=True); st.caption("Esplora la correlazione tra edge value e P/L.")
                 else: st.info("Nessun dato W/L con Edge Value valido.")
             else: st.info(f"Colonna '{colonna_edge}' non disponibile.")

        with col_d:
              if colonna_stake in df_filtered_final.columns:
                 st.subheader(f"Partecipazione Distribuzione (Tutti Filtrati > 0)")
                 df_plot_stake = df_filtered_final[df_filtered_final[colonna_stake].notna() & (df_filtered_final[colonna_stake] > 0)]
                 if not df_plot_stake.empty:
                     fig_stake_hist = px.histogram(df_plot_stake, x=colonna_stake, nbins=15, title="Distribuzione Stake (€) (Periodo Filtrato)", labels={colonna_stake: 'Importo Puntato (€)'}, color_discrete_sequence=[color_line1])
                     fig_stake_hist.update_traces(marker_line_width=0); fig_stake_hist.update_layout(template='plotly_white', bargap=0.1); fig_stake_hist.update_yaxes(title="Numero Scommesse", showgrid=False)
                     xaxis_title_stake_hist = "Importo Puntato (€)"; tickformat_stake_hist = "~€" if italian_locale_set else None; ticksuffix_stake_hist = None if italian_locale_set else " €"
                     fig_stake_hist.update_xaxes(title=xaxis_title_stake_hist, showgrid=False, tickformat=tickformat_stake_hist, ticksuffix=ticksuffix_stake_hist)
                     st.plotly_chart(fig_stake_hist, use_container_width=True); st.caption("Mostra come si distribuiscono gli importi puntati.")
                 else: st.info("Nessuna puntata valida (> 0 €).")
              else: st.info(f"Colonna '{colonna_stake}' non disponibile.")
        st.markdown("---")

        st.markdown(f"##### Conteggio Esiti (Vittoria/Perdita) per {colonna_quota}")
        if not df_results_filtered_final.empty and colonna_quota in df_results_filtered_final.columns:
             df_plot_odds = df_results_filtered_final.groupby([colonna_quota, 'Esito_Standard']).size().reset_index(name='Conteggio')
             fig_odds_outcome = px.bar(df_plot_odds, x=colonna_quota, y='Conteggio', color='Esito_Standard', barmode='group', title=f"Conteggio Win/Loss per {colonna_quota} (Periodo Filtrato)", labels={colonna_quota: 'Quota', 'Conteggio': 'Num. Scommesse', 'Esito_Standard': 'Esito'}, color_discrete_map=color_map_esiti)
             fig_odds_outcome.update_traces(marker_line_width=0); fig_odds_outcome.update_layout(template='plotly_white', bargap=0.2)
             fig_odds_outcome.update_yaxes(showgrid=False); fig_odds_outcome.update_xaxes(showgrid=False)
             st.plotly_chart(fig_odds_outcome, use_container_width=True); st.caption(f"Visualizza quante scommesse vinte/perse per fascia di quota.")
        else: st.info(f"Nessun dato per grafico esiti per {colonna_quota}.")
        st.markdown("---")

        st.subheader("Analisi Avanzate e Simulazioni")
        st.markdown("---"); st.markdown("##### Evoluzione del Sistema nel Tempo")
        if not df_results_filtered_final.empty and colonna_data in df_results_filtered_final.columns and df_results_filtered_final[colonna_data].notna().any():
            max_slider_val = len(df_results_filtered_final) if len(df_results_filtered_final) > 5 else 5
            default_slider_val = min(20, max_slider_val)
            window_size = st.slider("Finestra mobile (numero di scommesse)", min_value=5, max_value=max_slider_val, value=default_slider_val, help="Numero di scommesse considerate per ogni punto della finestra mobile")
            if len(df_results_filtered_final) >= window_size:
                df_sorted = df_results_filtered_final.sort_values(by=colonna_data).reset_index(drop=True)
                rolling_roi = []; rolling_winrate = []; rolling_dates = []
                for i in range(window_size, len(df_sorted) + 1):
                    window = df_sorted.iloc[i-window_size:i]; window_stake = window[colonna_stake].sum(); window_pl = window['P/L'].sum()
                    window_roi = (window_pl / window_stake) * 100 if window_stake > 0 else 0
                    window_wins = (window['Esito_Standard'] == 'Win').sum(); window_winrate = window_wins / window_size if window_size > 0 else 0
                    rolling_roi.append(window_roi); rolling_winrate.append(window_winrate); rolling_dates.append(window.iloc[-1][colonna_data])
                if rolling_dates:
                    df_rolling = pd.DataFrame({'Data': rolling_dates, 'ROI': rolling_roi, 'Win_Rate': rolling_winrate})
                    fig_evolution = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_evolution.add_trace(go.Scatter(x=df_rolling['Data'], y=df_rolling['ROI'], name=f'ROI Mobile ({window_size} scommesse)', line=dict(color='#4682B4', width=2, shape='linear')), secondary_y=False)
                    fig_evolution.add_trace(go.Scatter(x=df_rolling['Data'], y=df_rolling['Win_Rate'], name=f'Win Rate Mobile ({window_size} scommesse)', line=dict(color='#7B68EE', width=2, shape='linear')), secondary_y=True)
                    fig_evolution.update_layout(title=f"Evoluzione del Sistema - Finestra Mobile di {window_size} scommesse", template='plotly_white', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
                    fig_evolution.update_yaxes(title_text="ROI (%)", secondary_y=False, showgrid=False); fig_evolution.update_yaxes(title_text="Win Rate", secondary_y=True, showgrid=False, tickformat=".1%")
                    fig_evolution.update_xaxes(title_text="Data", showgrid=False)
                    st.plotly_chart(fig_evolution, use_container_width=True); st.caption("Mostra l'evoluzione del ROI e Win Rate nel tempo usando una finestra mobile. Utile per verificare se il sistema migliora o peggiora nel tempo.")
                else: st.info("Non ci sono abbastanza dati per calcolare l'evoluzione con la finestra mobile selezionata.")
            else: st.info(f"Insufficiente numero di scommesse per l'analisi con finestra mobile. Ne servono almeno {window_size} (dati filtrati: {len(df_results_filtered_final)}).")
        else: st.info("Nessun dato o date valide per l'analisi dell'evoluzione del sistema.")

        st.markdown("---"); st.markdown("##### Simulatore di Strategie di Staking")
        if not df_results_filtered_final.empty:
            col1_sim, col2_sim = st.columns(2)
            staking_strategy = col1_sim.selectbox("Seleziona Strategia di Staking", ["Stake Fisso", "Percentuale Bankroll", "Kelly Criterion", "Stake Proporzionale all'Edge"])
            starting_bankroll = col2_sim.number_input("Bankroll Iniziale (€)", min_value=100.0, value=1000.0, step=100.0)
            staking_descriptions = {
                "Stake Fisso": "**Stake Fisso**: Si punta sempre lo stesso importo predefinito per ogni scommessa, indipendentemente dal bankroll corrente o dalla confidenza nella scommessa. È la strategia più semplice ma meno adattiva.",
                "Percentuale Bankroll": "**Percentuale Bankroll**: Si punta una percentuale fissa del bankroll corrente (es. 1-5%). L'importo puntato varia quindi con le fluttuazioni del bankroll, proteggendolo durante le serie negative e aumentando le puntate durante quelle positive.",
                "Kelly Criterion": "**Kelly Criterion**: Formula matematica ((probabilità * quota) - 1) / (quota - 1) per determinare la frazione ottimale del bankroll da puntare, massimizzando la crescita del bankroll nel lungo periodo. Richiede stime accurate di probabilità e quota. Spesso si usa una frazione di Kelly (es. 50%) per ridurre la volatilità.",
                "Stake Proporzionale all'Edge": "**Stake Proporzionale all'Edge**: L'importo puntato è proporzionale al vantaggio (edge) percepito nella scommessa (Edge = (Probabilità Stimata * Quota) - 1). Scommesse con edge maggiore ricevono uno stake maggiore, spesso con un cap massimo in percentuale del bankroll per controllare il rischio."
            }
            with st.expander(f"Dettagli sulla Strategia: {staking_strategy}"): st.markdown(staking_descriptions.get(staking_strategy, "Nessuna descrizione disponibile."))
            stake_param = None
            if staking_strategy == "Stake Fisso": stake_param = st.slider("Stake Fisso (€)", min_value=1.0, max_value=min(100.0, starting_bankroll), value=min(10.0, starting_bankroll/10 if starting_bankroll > 0 else 10.0), step=1.0)
            elif staking_strategy == "Percentuale Bankroll": stake_param = st.slider("Percentuale Bankroll per Scommessa (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1) / 100
            elif staking_strategy == "Kelly Criterion":
                if colonna_prob not in df_results_filtered_final.columns: st.error(f"Strategia '{staking_strategy}' richiede '{colonna_prob}'.")
                else: stake_param = st.slider("Frazione di Kelly (%)", min_value=1, max_value=100, value=50, step=1) / 100
            elif staking_strategy == "Stake Proporzionale all'Edge":
                if colonna_prob not in df_results_filtered_final.columns or colonna_edge not in df_results_filtered_final.columns: st.error(f"Strategia '{staking_strategy}' richiede '{colonna_prob}' e '{colonna_edge}'.")
                else: stake_param = {'multiplier': st.slider("Moltiplicatore Stake per Unità di Edge", min_value=1, max_value=200, value=50, step=1), 'cap_perc': st.slider("Cap Massimo Stake (% del Bankroll)", min_value=1.0, max_value=20.0, value=10.0, step=0.5) / 100}

            if st.button("Simula Strategia"):
                proceed_simulation = True
                if staking_strategy in ["Kelly Criterion", "Stake Proporzionale all'Edge"] and colonna_prob not in df_results_filtered_final.columns: proceed_simulation = False
                if staking_strategy == "Stake Proporzionale all'Edge" and colonna_edge not in df_results_filtered_final.columns: proceed_simulation = False
                if proceed_simulation and stake_param is not None:
                    df_sim = df_results_filtered_final.sort_values(by=colonna_data).copy(); df_sim.reset_index(drop=True, inplace=True)
                    bankroll = float(starting_bankroll); bankroll_history = [bankroll]; bet_amounts_sim = []; stakes_as_perc_of_bankroll = []
                    for index_sim, bet_row in df_sim.iterrows():
                        current_stake = 0.0
                        if bankroll <= 0.01: current_stake = 0.0
                        elif staking_strategy == "Stake Fisso": current_stake = float(min(stake_param, bankroll))
                        elif staking_strategy == "Percentuale Bankroll": current_stake = float(min(bankroll * float(stake_param), bankroll))
                        elif staking_strategy == "Kelly Criterion":
                            prob_kelly = bet_row.get(colonna_prob); quota_kelly = bet_row.get(colonna_quota)
                            if pd.notna(prob_kelly) and pd.notna(quota_kelly) and prob_kelly > 0 and prob_kelly < 1 and quota_kelly > 1:
                                b_val_kelly = quota_kelly - 1; kelly_full_fraction = (prob_kelly * b_val_kelly - (1 - prob_kelly)) / b_val_kelly
                                if kelly_full_fraction > 0: current_stake = float(min(kelly_full_fraction * float(stake_param) * bankroll, bankroll))
                        elif staking_strategy == "Stake Proporzionale all'Edge":
                            edge_val_sim = bet_row.get(colonna_edge)
                            if pd.notna(edge_val_sim) and edge_val_sim > 0:
                                stake_calculated_from_edge = edge_val_sim * float(stake_param['multiplier'])
                                stake_capped_by_bankroll_perc = bankroll * float(stake_param['cap_perc'])
                                current_stake = float(min(stake_calculated_from_edge, stake_capped_by_bankroll_perc, bankroll))
                        current_stake = float(max(0.0, min(float(current_stake if pd.notna(current_stake) else 0.0), bankroll)))
                        bet_amounts_sim.append(current_stake); stakes_as_perc_of_bankroll.append((current_stake / bankroll * 100) if bankroll > 0 else 0)
                        if bet_row['Esito_Standard'] == 'Win': bankroll += float(current_stake) * (float(bet_row.get(colonna_quota, 1.0)) - 1.0)
                        elif bet_row['Esito_Standard'] == 'Loss': bankroll -= float(current_stake)
                        bankroll = max(0.0, float(bankroll if pd.notna(bankroll) else 0.0)); bankroll_history.append(bankroll)
                        if bankroll < 0.01 and index_sim < len(df_sim) -1:
                            remaining_bets = len(df_sim) - 1 - index_sim; bankroll_history.extend([0.0] * remaining_bets); bet_amounts_sim.extend([0.0] * remaining_bets); stakes_as_perc_of_bankroll.extend([0.0] * remaining_bets); break
                    if len(bet_amounts_sim) != len(df_sim):
                        fill_count = len(df_sim) - len(bet_amounts_sim); bet_amounts_sim.extend([0.0] * fill_count); stakes_as_perc_of_bankroll.extend([0.0] * fill_count)
                    df_sim['Stake_Simulato'] = bet_amounts_sim; df_sim['Stake_Simulato_%Bankroll'] = stakes_as_perc_of_bankroll
                    final_bankroll = bankroll_history[-1]; total_staked_sim = sum(bet_amounts_sim); profit_sim = final_bankroll - starting_bankroll; roi_sim = (profit_sim / total_staked_sim * 100) if total_staked_sim > 0 else 0; max_bankroll_hist = max(bankroll_history) if bankroll_history else starting_bankroll
                    max_drawdown_abs_sim = 0; max_drawdown_perc_sim = 0
                    if bankroll_history:
                        bh_series = pd.Series(bankroll_history); cumulative_max_bh = bh_series.cummax(); drawdowns_bh = cumulative_max_bh - bh_series
                        if not drawdowns_bh.empty: max_drawdown_abs_sim = drawdowns_bh.max(); peak_for_max_dd = cumulative_max_bh[drawdowns_bh.idxmax()] if max_drawdown_abs_sim > 0 else 0; max_drawdown_perc_sim = (max_drawdown_abs_sim / peak_for_max_dd * 100) if peak_for_max_dd > 0 else 0
                    st.markdown("##### Risultati Simulazione"); metric_col1_res, metric_col2_res, metric_col3_res, metric_col4_res = st.columns(4)
                    metric_col1_res.metric("Bankroll Finale", f"{final_bankroll:.2f} €", f"{profit_sim:+.2f} €"); metric_col2_res.metric("ROI Simulato", f"{roi_sim:.2f}%", help=f"Profitto / Totale Puntato ({format_currency(profit_sim)} / {format_currency(total_staked_sim)})")
                    metric_col3_res.metric("Bankroll Massimo Raggiunto", f"{max_bankroll_hist:.2f} €"); metric_col4_res.metric("Max Drawdown Simulato", f"{max_drawdown_abs_sim:.2f} € ({max_drawdown_perc_sim:.1f}%)", help="Massima perdita dal picco precedente durante la simulazione.")
                    fig_bankroll = go.Figure(); fig_bankroll.add_trace(go.Scatter(x=df_sim[colonna_data] if colonna_data in df_sim.columns and not df_sim[colonna_data].empty else df_sim.index, y=bankroll_history[1:] if len(bankroll_history) > 1 else [], mode='lines', name='Andamento Bankroll', line=dict(color='#2E8B57', width=2, shape='linear')))
                    fig_bankroll.add_hline(y=starting_bankroll, line_dash="dash", line_color="red", annotation_text="Bankroll Iniziale", annotation_position="bottom right"); fig_bankroll.update_layout(title=f"Evoluzione Bankroll - Strategia: {staking_strategy}", xaxis_title='Scommessa (ordinata per data)', yaxis_title='Bankroll (€)', template='plotly_white')
                    st.plotly_chart(fig_bankroll, use_container_width=True)
                    with st.expander("Mostra Dettaglio Scommesse Simulate"):
                        df_sim_display = df_sim[[colonna_data, colonna_sq_a, colonna_sq_b, colonna_quota, 'Esito_Standard', 'Stake_Simulato', 'Stake_Simulato_%Bankroll']].copy()
                        df_sim_display['Bankroll_Dopo_Scommessa'] = bankroll_history[1:][:len(df_sim_display)] if len(bankroll_history) > 1 else ([0.0]*len(df_sim_display))
                        df_sim_display_pl_simulato = []
                        for i, row_sim_disp in df_sim_display.iterrows():
                            stake_s = df_sim['Stake_Simulato'].iloc[i]; pl_s = 0
                            if row_sim_disp['Esito_Standard'] == 'Win' and pd.notna(row_sim_disp[colonna_quota]) and row_sim_disp[colonna_quota] > 0 and pd.notna(stake_s) and stake_s > 0: pl_s = (stake_s * row_sim_disp[colonna_quota]) - stake_s
                            elif row_sim_disp['Esito_Standard'] == 'Loss' and pd.notna(stake_s) and stake_s > 0: pl_s = -stake_s
                            df_sim_display_pl_simulato.append(pl_s)
                        df_sim_display['P/L Simulato'] = [format_currency(val) for val in df_sim_display_pl_simulato]
                        if colonna_data in df_sim_display.columns: df_sim_display[colonna_data] = pd.to_datetime(df_sim_display[colonna_data], errors='coerce').dt.strftime('%Y-%m-%d')
                        df_sim_display['Stake_Simulato'] = df_sim_display['Stake_Simulato'].apply(format_currency); df_sim_display['Stake_Simulato_%Bankroll'] = df_sim_display['Stake_Simulato_%Bankroll'].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                        df_sim_display['Bankroll_Dopo_Scommessa'] = pd.to_numeric(df_sim_display['Bankroll_Dopo_Scommessa'], errors='coerce').apply(format_currency)
                        st.dataframe(df_sim_display[[colonna_data, colonna_sq_a, colonna_sq_b, colonna_quota, 'Esito_Standard', 'Stake_Simulato', 'Stake_Simulato_%Bankroll', 'P/L Simulato', 'Bankroll_Dopo_Scommessa']], hide_index=True, use_container_width=True)
                elif proceed_simulation and stake_param is None: st.warning("Parametri di staking non definiti correttamente.")
        else: st.info("Nessun dato disponibile per il simulatore di strategie di staking.")

        st.markdown("---"); st.markdown("##### Simulatore di Regressione alla Media (Beta)")
        with st.container(border=True):
            st.markdown("""Questo simulatore stima come la performance (ROI) potrebbe evolvere nel futuro basandosi sui dati storici filtrati. Utilizza simulazioni Monte Carlo per proiettare una distribuzione di possibili ROI futuri, aiutando a capire se il ROI attuale è sostenibile o potrebbe regredire verso una media.""")
            col1_regr, col2_regr = st.columns(2)
            max_sim_val_regr = len(df_results_filtered_final) if not df_results_filtered_final.empty else 100; default_future_bets_regr = min(100, max_sim_val_regr if max_sim_val_regr > 0 else 100)
            num_simulations_regr = col1_regr.slider("Numero Simulazioni Monte Carlo", min_value=100, max_value=5000, value=1000, step=100, key="num_sim_regr_3")
            future_bets_regr = col2_regr.slider("Scommesse Future per Simulazione", min_value=10, max_value=max(10, max_sim_val_regr), value=default_future_bets_regr, step=10, key="future_bets_regr_3")
            if st.button("Esegui Simulazione di Regressione"):
                if not df_results_filtered_final.empty and win_rate_wl is not None and pd.notna(win_rate_wl) and roi_wl is not None and pd.notna(roi_wl):
                    current_win_rate_regr = win_rate_wl; current_roi_regr = roi_wl / 100
                    winning_bets_filtered = df_results_filtered_final[df_results_filtered_final['Esito_Standard'] == 'Win']
                    unit_profit_from_wins = (winning_bets_filtered[colonna_quota] - 1).dropna().values if not winning_bets_filtered.empty and colonna_quota in winning_bets_filtered.columns and winning_bets_filtered[colonna_quota].notna().any() else np.array([current_roi_regr / current_win_rate_regr if current_win_rate_regr > 0 and pd.notna(current_win_rate_regr) else 0])
                    if len(unit_profit_from_wins) == 0 : unit_profit_from_wins = np.array([0])
                    simulation_roi_results = []
                    for _ in range(num_simulations_regr):
                        sim_pl_total = 0; sim_total_stake_normalized = future_bets_regr
                        for _ in range(future_bets_regr):
                            if np.random.random() < current_win_rate_regr: sim_pl_total += np.random.choice(unit_profit_from_wins) if len(unit_profit_from_wins) > 0 else 0
                            else: sim_pl_total -= 1
                        simulation_roi_results.append(sim_pl_total / sim_total_stake_normalized if sim_total_stake_normalized > 0 else 0)
                    simulation_roi_results_np = np.array(simulation_roi_results); avg_future_roi_regr = np.mean(simulation_roi_results_np); median_future_roi_regr = np.median(simulation_roi_results_np); std_future_roi_regr = np.std(simulation_roi_results_np); percentile_5_regr = np.percentile(simulation_roi_results_np, 5); percentile_95_regr = np.percentile(simulation_roi_results_np, 95)
                    res_col1_regr, res_col2_regr, res_col3_regr, res_col4_regr = st.columns(4)
                    res_col1_regr.metric("ROI Attuale (dati filtrati)", f"{current_roi_regr:.2%}"); res_col2_regr.metric("ROI Futuro Stimato (Media Sim.)", f"{avg_future_roi_regr:.2%}", f"{avg_future_roi_regr - current_roi_regr:+.2%}")
                    res_col3_regr.metric("ROI Futuro (Mediana Sim.)", f"{median_future_roi_regr:.2%}"); res_col4_regr.metric("Dev. Std. ROI Sim.", f"{std_future_roi_regr:.2%}")
                    fig_dist_regr = go.Figure(); fig_dist_regr.add_trace(go.Histogram(x=simulation_roi_results_np, histnorm='probability density', name='Distribuzione ROI Simulato', marker_color='#4682B4', opacity=0.7))
                    fig_dist_regr.add_vline(x=avg_future_roi_regr, line_dash="solid", line_color="black", annotation_text="Media Sim.", annotation_position="top right"); fig_dist_regr.add_vline(x=current_roi_regr, line_dash="dash", line_color="red", annotation_text="ROI Attuale", annotation_position="top left")
                    fig_dist_regr.add_vline(x=percentile_5_regr, line_dash="dot", line_color="orange", annotation_text="5° percentile", annotation_position="bottom left"); fig_dist_regr.add_vline(x=percentile_95_regr, line_dash="dot", line_color="green", annotation_text="95° percentile", annotation_position="bottom right")
                    fig_dist_regr.update_layout(title=f"Distribuzione ROI Simulato ({future_bets_regr} scommesse future, {num_simulations_regr} simulazioni)", xaxis_title="ROI", yaxis_title="Densità", template='plotly_white', xaxis=dict(tickformat='.1%'))
                    st.plotly_chart(fig_dist_regr, use_container_width=True); st.markdown(f"**Intervallo Confidenza 90% ROI Futuro**: {percentile_5_regr:.2%} – {percentile_95_regr:.2%}")
                else: st.warning("Dati insufficienti per la simulazione di regressione.")

        st.markdown("---"); st.subheader("Tabella Dati Completa (Filtrata e Formattata)")
        df_display_filtered = df_filtered_final.copy()
        def safe_format(value, format_str): return format(value, format_str) if pd.notna(value) else ""
        cols_to_format_percent = [colonna_prob, colonna_edge, 'EV', 'Kelly_Stake']
        cols_to_format_currency_in_table = [colonna_stake, 'P/L']
        cols_to_format_points = [colonna_media_pt_stimati, colonna_ris_finale, colonna_confidenza, 'Errore_Sovrastima_PT']
        for col in cols_to_format_percent:
             if col in df_display_filtered.columns: df_display_filtered[col] = pd.to_numeric(df_display_filtered[col], errors='coerce').map(lambda x: safe_format(x, '.1%'))
        for col in cols_to_format_currency_in_table:
             if col in df_display_filtered.columns: df_display_filtered[col] = pd.to_numeric(df_display_filtered[col], errors='coerce').apply(format_currency)
        for col in cols_to_format_points:
             if col in df_display_filtered.columns: df_display_filtered[col] = pd.to_numeric(df_display_filtered[col], errors='coerce').map(lambda x: safe_format(x, '.1f'))
        if colonna_quota in df_display_filtered.columns: df_display_filtered[colonna_quota] = pd.to_numeric(df_display_filtered[colonna_quota], errors='coerce').map(lambda x: safe_format(x, '.2f'))
        if 'Quota_BE' in df_display_filtered.columns: df_display_filtered['Quota_BE'] = pd.to_numeric(df_display_filtered['Quota_BE'], errors='coerce').map(lambda x: safe_format(x, '.2f'))
        if colonna_data in df_display_filtered.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_display_filtered[colonna_data]): df_display_filtered[colonna_data] = pd.to_datetime(df_display_filtered[colonna_data], errors='coerce')
            if pd.api.types.is_datetime64_any_dtype(df_display_filtered[colonna_data]): df_display_filtered[colonna_data] = df_display_filtered[colonna_data].dt.strftime('%Y-%m-%d')
        colonne_display_ordine = [colonna_sq_a, colonna_sq_b, colonna_data, colonna_ora, colonna_quota, 'Quota_BE', colonna_stake, colonna_prob, colonna_edge, 'EV', 'Kelly_Stake', colonna_confidenza, colonna_media_pt_stimati, colonna_ris_finale, 'Errore_Sovrastima_PT', 'Esito_Standard', 'P/L']
        colonne_display_esistenti = [col for col in colonne_display_ordine if col in df_display_filtered.columns]
        st.dataframe(df_display_filtered[colonne_display_esistenti], hide_index=True, use_container_width=True)

        st.markdown("---"); st.subheader("Download ed Esportazione")
        csv_data = df_cleaned.to_csv(index=False, decimal='.', sep=',').encode('utf-8')
        st.download_button(label="💾 Scarica Dati Puliti e Calcolati (.csv)", data=csv_data, file_name='dati_scommesse_puliti_calcolati.csv', mime='text/csv', help="Scarica il dataset completo dopo la pulizia e l'aggiunta delle colonne calcolate.")
        st.markdown("---"); st.subheader("Esportare come PDF"); st.info("Per salvare la vista corrente della dashboard: usa File -> Stampa del tuo browser e scegli 'Salva come PDF'.")
except FileNotFoundError: st.error(f"ERRORE CRITICO: File '{file_path}' non trovato."); st.info("Assicurati che il file CSV sia nella stessa cartella dello script."); st.stop()
except pd.errors.EmptyDataError: st.error(f"ERRORE CRITICO: Il file CSV '{file_path}' è vuoto."); st.stop()
except KeyError as e:
    st.error(f"ERRORE CRITICO: Colonna fondamentale non trovata nel CSV: {e}.")
    st.info(f"Controlla che il file '{file_path}' contenga tutte le colonne necessarie definite all'inizio dello script (es. {colonna_sq_a}, {colonna_esito}, ecc.).")
    if 'df' in locals() and hasattr(df, 'columns'): st.info(f"Colonne effettivamente lette dal CSV: {df.columns.tolist()}")
    st.stop()
except Exception as e:
    st.error(f"ERRORE IMPREVISTO NELL'APPLICAZIONE: {type(e).__name__}: {e}")
    st.error("Si è verificato un errore durante l'esecuzione dello script. Controlla i log o prova a ricaricare.")
    # import traceback
    # st.text_area("Traceback", traceback.format_exc(), height=300) # Decommentare per debug completo
    st.stop()
