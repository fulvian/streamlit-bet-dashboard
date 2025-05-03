# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import locale

# --- Impostazioni Iniziali ---
st.set_page_config(layout="wide")
st.title("📊 Dashboard Analisi Scommesse")

# --- MODIFICA: Gestione Localizzazione con Fallback ---
# Flag per sapere se la localizzazione italiana è stata impostata con successo
italian_locale_set = False
try:
    # Prova prima il formato UTF-8 comune su Linux/macOS
    locale.setlocale(locale.LC_ALL, 'it_IT.UTF-8')
    italian_locale_set = True # Successo!
except locale.Error:
    try:
        # Prova il formato comune su Windows
        locale.setlocale(locale.LC_ALL, 'Italian_Italy.1252')
        italian_locale_set = True # Successo!
    except locale.Error:
        # Se entrambi falliscono, mostra l'avviso
        st.warning("Localizzazione italiana ('it_IT.UTF-8' o 'Italian_Italy.1252') non disponibile sul sistema. Verrà usata una formattazione valuta di fallback (€ X.XX).")
# --- FINE MODIFICA ---

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

def parse_italian_date(date_str):
    """Converte una stringa di data in datetime."""
    if pd.isna(date_str): return pd.NaT
    try: return pd.to_datetime(date_str, format='%d %b %Y', errors='raise')
    except ValueError: return pd.to_datetime(date_str, errors='coerce')

def calculate_pl(row, stake_col='Stake', quota_col='Quota', esito_col='Esito_Standard'):
    """Calcola il Profit/Loss (P/L) per una riga."""
    if stake_col not in row or quota_col not in row or esito_col not in row: return 0.0
    esito = row[esito_col]; quota = row[quota_col]; stake = row[stake_col]
    if esito == 'Win' and pd.notna(quota) and quota > 0 and pd.notna(stake) and stake > 0:
        return round((stake * quota) - stake, 2)
    elif esito == 'Loss' and pd.notna(stake) and stake > 0:
        return round(-stake, 2)
    else: return 0.0

# --- MODIFICA: Funzione Formattazione Valuta con Fallback ---
def format_currency(value):
    """
    Formatta un valore come valuta.
    Usa la localizzazione italiana se disponibile, altrimenti usa un formato fallback (€ X.XX).
    """
    if pd.isna(value):
        return "N/D" # O "" a seconda delle preferenze
    if italian_locale_set:
        # Se la localizzazione italiana è attiva, usa locale.currency
        try:
            return locale.currency(value, grouping=True)
        except ValueError:
            # Fallback ulteriore nel caso locale.currency dia problemi anche con locale IT impostato
             return f"{value:.2f} €"
    else:
        # Altrimenti, usa la formattazione manuale fallback
        return f"{value:.2f} €"
# --- FINE MODIFICA ---


# --- Caricamento e Pulizia Dati ---
file_path = "risultati_bet.csv"
colonna_sq_a = "Squadra A"; colonna_sq_b = "Squadra B"; colonna_data = "Data"
colonna_ora = "Ora"; colonna_quota = "Quota"; colonna_stake = "Stake"
colonna_prob = "Probabilita Stimata"; colonna_edge = "Edge Value"; colonna_esito = "Esito"
colonna_media_pt_stimati = "Media Punti Stimati"; colonna_confidenza = "Confidenza"; colonna_ris_finale = "Punteggio finale"

try:
    # Assicurati che lo script usi il nome file corretto
    # Cambia 'dashboard_app.py' con il nome effettivo del tuo file se diverso
    # Il nome del file CSV è letto dalla variabile file_path
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

    df_cleaned['Errore_Sovrastima_PT'] = np.nan
    if colonna_media_pt_stimati in df_cleaned.columns and colonna_ris_finale in df_cleaned.columns:
        valid_error_calc = df_cleaned[colonna_media_pt_stimati].notna() & df_cleaned[colonna_ris_finale].notna()
        diff = df_cleaned.loc[valid_error_calc, colonna_media_pt_stimati] - df_cleaned.loc[valid_error_calc, colonna_ris_finale]
        df_cleaned.loc[valid_error_calc, 'Errore_Sovrastima_PT'] = diff.clip(lower=0)

    # --- Preparazione Dati Globali (pre-filtro sidebar) ---
    df_results_global = df_cleaned[df_cleaned['Esito_Standard'].isin(['Win', 'Loss']) & df_cleaned[colonna_data].notna()].copy()
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
    if colonna_data in df_cleaned.columns and df_cleaned[colonna_data].notna().any():
        valid_dates = df_cleaned[colonna_data].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date(); max_date = valid_dates.max().date()
            if min_date != max_date :
                 selected_date_range = st.sidebar.date_input(f"Filtra per {colonna_data}", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                 if len(selected_date_range) == 2: start_date, end_date = selected_date_range
                 elif len(selected_date_range) == 1: start_date = end_date = selected_date_range[0]
            else:
                st.sidebar.info(f"Tutti i dati sono relativi al {min_date.strftime('%d/%m/%Y')}")
                start_date = end_date = min_date

    # --- Applicazione Filtri ---
    df_filtered_final = df_cleaned.copy()
    df_results_filtered_final = df_results_global.copy()
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
        sharpe_ratio_daily = np.nan; var_95_daily = np.nan

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
                daily_pl_filtered = df_results_filtered_final.groupby(pd.Grouper(key=colonna_data, freq='D'))['P/L'].sum()
                daily_pl_filtered = daily_pl_filtered[daily_pl_filtered != 0]
                if len(daily_pl_filtered) > 1:
                    mean_daily_return = daily_pl_filtered.mean(); std_dev_daily_return = daily_pl_filtered.std()
                    if std_dev_daily_return != 0 and pd.notna(std_dev_daily_return): sharpe_ratio_daily = mean_daily_return / std_dev_daily_return
                    var_95_daily = daily_pl_filtered.quantile(0.05)

        if 'Errore_Sovrastima_PT' in df_filtered_final.columns:
             overestimations = df_filtered_final[df_filtered_final['Errore_Sovrastima_PT'] > 0]['Errore_Sovrastima_PT']
             if not overestimations.empty:
                 avg_overestimation_error = overestimations.mean()

        # --- Visualizzazione Metriche (Usa format_currency) ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Scommesse Concluse (W/L)", f"{total_bets_wl}")
        # --- MODIFICA: Usa la nuova funzione per formattare ---
        col2.metric(f"{colonna_stake} Totale (W/L)", format_currency(total_stake_wl))
        col3.metric(f"P/L Totale (W/L)", format_currency(total_pl_wl))
        # --- FINE MODIFICA ---
        col4.metric("Sovrastima Media PT", f"{avg_overestimation_error:.2f}" if pd.notna(avg_overestimation_error) else "N/D",
                    help=f"Media di ({colonna_media_pt_stimati} - {colonna_ris_finale}) solo quando la stima era superiore al risultato finale.")

        col5, col6, col7 = st.columns(3)
        col5.metric("ROI (W/L)", f"{roi_wl:.2f}%" if pd.notna(roi_wl) else "N/D")
        col6.metric("Win Rate (W/L)", f"{win_rate_wl:.1%}" if pd.notna(win_rate_wl) else "N/D")
        # --- MODIFICA: Usa la nuova funzione per formattare ---
        col7.metric(f"{colonna_stake} Medio (W/L)", format_currency(avg_stake_wl))
        # --- FINE MODIFICA ---

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
        with st.container(border=True):
             r_col1, r_col2, r_col3 = st.columns(3)
             # --- MODIFICA: Usa la nuova funzione per formattare ---
             r_col1.metric("Max Drawdown Storico (€)", format_currency(max_drawdown_global), help="Massima perdita storica dal picco precedente (calcolata su tutti i dati).")
             r_col2.metric("Sharpe Ratio (Giornaliero)", f"{sharpe_ratio_daily:.2f}" if pd.notna(sharpe_ratio_daily) else "N/D", help="Basato su P/L giornaliero dei dati filtrati. Risk-free rate = 0.")
             r_col3.metric("VaR Storico 95% (Giornaliero)", format_currency(var_95_daily), help="Perdita giornaliera massima attesa nel 5% dei casi peggiori (basata sui dati filtrati).")
             # --- FINE MODIFICA ---
        st.markdown("---")

        # --- Visualizzazioni con Stile Aggiornato ---
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

        # Grafico 2: Conteggio Esiti per Quota
        st.markdown(f"##### Conteggio Esiti (Win/Loss) per {colonna_quota}")
        if not df_results_filtered_final.empty and colonna_quota in df_results_filtered_final.columns:
             df_plot_odds = df_results_filtered_final.groupby([colonna_quota, 'Esito_Standard']).size().reset_index(name='Conteggio')
             fig_odds_outcome = px.bar(df_plot_odds, x=colonna_quota, y='Conteggio', color='Esito_Standard', barmode='group', title=f"Conteggio Win/Loss per {colonna_quota} (Periodo Filtrato)", labels={colonna_quota: 'Quota', 'Conteggio': 'Num. Scommesse', 'Esito_Standard': 'Esito'}, color_discrete_map=color_map_esiti)
             fig_odds_outcome.update_traces(marker_line_width=0); fig_odds_outcome.update_layout(template='plotly_white', bargap=0.2)
             fig_odds_outcome.update_yaxes(showgrid=False); fig_odds_outcome.update_xaxes(showgrid=False)
             st.plotly_chart(fig_odds_outcome, use_container_width=True); st.caption(f"Visualizza quante scommesse vinte/perse per fascia di quota.")
        else: st.info(f"Nessun dato per grafico esiti per {colonna_quota}.")
        st.markdown("---")

        # Grafico 3: P/L e Conteggio Giornaliero
        st.subheader("Andamento Giornaliero P/L e Volume Scommesse (Periodo Filtrato)")
        if not df_results_filtered_final.empty:
            daily_summary_chart = df_results_filtered_final.groupby(pd.Grouper(key=colonna_data, freq='D')).agg(Daily_PL=('P/L', 'sum'), Num_Bets=(colonna_esito, 'size')).reset_index()
            daily_summary_chart = daily_summary_chart[daily_summary_chart['Num_Bets'] > 0]
            if not daily_summary_chart.empty:
                fig_daily = make_subplots(specs=[[{"secondary_y": True}]]); colors_pl = [color_win if pl >= 0 else color_loss for pl in daily_summary_chart['Daily_PL']]
                fig_daily.add_trace(go.Bar(x=daily_summary_chart[colonna_data], y=daily_summary_chart['Daily_PL'], name='P/L Giornaliero (€)', marker_color=colors_pl, marker_line_width=0, yaxis='y1'), secondary_y=False)
                fig_daily.add_trace(go.Scatter(x=daily_summary_chart[colonna_data], y=daily_summary_chart['Num_Bets'], name='Num. Scommesse', mode='lines+markers', line=dict(color=color_line1, shape='spline', smoothing=0.7), marker=dict(size=5), yaxis='y2'), secondary_y=True)
                fig_daily.update_layout(title_text="Risultato Netto (€) e Num. Scommesse per Giorno (Periodo Filtrato)", xaxis_title="Data", yaxis_title="P/L Netto Giornaliero (€)", yaxis2_title="Numero Scommesse", legend_title="Legenda", barmode='relative', template='plotly_white')
                # --- MODIFICA: Usa format_currency per tick asse Y P/L ---
                # Nota: tickformat potrebbe sovrascrivere la formattazione completa del locale,
                # quindi usiamo un approccio diverso se il locale non è impostato.
                if italian_locale_set:
                    fig_daily.update_yaxes(title_text="P/L Netto Giornaliero (€)", tickformat="~€", secondary_y=False, showgrid=False) # Formato valuta approssimativo per tick
                else:
                    fig_daily.update_yaxes(title_text="P/L Netto Giornaliero (€)", ticksuffix=" €", secondary_y=False, showgrid=False) # Aggiunge solo simbolo
                # --- FINE MODIFICA ---
                fig_daily.update_yaxes(title_text="Numero Scommesse", secondary_y=True, showgrid=False); fig_daily.update_xaxes(showgrid=False)
                st.plotly_chart(fig_daily, use_container_width=True); st.caption("Mostra il P/L netto giornaliero e il numero di scommesse concluse.")
            else: st.info("Nessun dato giornaliero aggregato.")
        else: st.info("Nessun dato W/L per grafico P/L giornaliero.")
        st.markdown("---")

        # Grafico 4: P/L Cumulativo
        st.subheader("Andamento Cumulativo P/L nel Tempo (Periodo Filtrato)")
        if not df_results_filtered_final.empty and 'Cumulative P/L Filtered' in df_results_filtered_final.columns and df_results_filtered_final['Cumulative P/L Filtered'].notna().any():
            fig_cum_pl = px.line(df_results_filtered_final, x=colonna_data, y='Cumulative P/L Filtered', title="Andamento P/L Cumulativo (€) - Periodo Filtrato", markers=False)
            fig_cum_pl.update_traces(line=dict(color=color_line2, shape='spline', smoothing=0.7)); fig_cum_pl.update_layout(template='plotly_white')
             # --- MODIFICA: Usa format_currency per tick asse Y P/L ---
            if italian_locale_set:
                 fig_cum_pl.update_yaxes(title="P/L Cumulativo (€)", showgrid=False, tickformat="~€")
            else:
                 fig_cum_pl.update_yaxes(title="P/L Cumulativo (€)", showgrid=False, ticksuffix=" €")
            # --- FINE MODIFICA ---
            fig_cum_pl.update_xaxes(title="Data", showgrid=False)
            st.plotly_chart(fig_cum_pl, use_container_width=True); st.caption("Mostra l'evoluzione del P/L totale nel tempo per il periodo selezionato.")
        else: st.info("Nessun dato P/L per grafico cumulativo filtrato.")
        st.markdown("---")

        # Grafico 5: Quota vs Probabilità Stimata
        if colonna_quota in df_results_filtered_final.columns and colonna_prob in df_results_filtered_final.columns:
            st.subheader(f"Relazione tra {colonna_quota} e {colonna_prob}")
            df_plot_quota_prob = df_results_filtered_final[df_results_filtered_final[colonna_quota].notna() & df_results_filtered_final[colonna_prob].notna()].copy()
            if not df_plot_quota_prob.empty:
                df_plot_quota_prob['Prob Formattata'] = df_plot_quota_prob[colonna_prob].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
                df_plot_quota_prob['Quota Formattata'] = df_plot_quota_prob[colonna_quota].map(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
                fig_quota_prob = px.scatter(df_plot_quota_prob, x=colonna_quota, y=colonna_prob, color='Esito_Standard', title=f"{colonna_quota} vs. {colonna_prob} (Periodo Filtrato)", labels={colonna_quota: 'Quota', colonna_prob: 'Probabilità Stimata', 'Esito_Standard': 'Esito'}, hover_name=colonna_sq_a, hover_data={colonna_sq_b:True, 'Prob Formattata':True, 'Quota Formattata':True, colonna_prob:False, colonna_quota:False}, color_discrete_map=color_map_esiti)
                fig_quota_prob.update_traces(marker=dict(line=dict(width=0))); fig_quota_prob.update_layout(template='plotly_white')
                fig_quota_prob.update_xaxes(showgrid=False); fig_quota_prob.update_yaxes(showgrid=False, tickformat=".1%")
                st.plotly_chart(fig_quota_prob, use_container_width=True); st.caption("Mostra la relazione tra quota e probabilità stimata.")
            else: st.info(f"Nessun dato valido per grafico {colonna_quota} vs {colonna_prob}.")
            st.markdown("---")
        else: st.info(f"Le colonne '{colonna_quota}' o '{colonna_prob}' non sono disponibili."); st.markdown("---")

        # Grafico 6: Errore Sovrastima vs Probabilità
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
                 st.plotly_chart(fig_prob_error, use_container_width=True)
                 st.caption("Mostra la relazione tra la sovrastima dei punti (Media Stimati - Punteggio Finale, solo se positivo) e la probabilità stimata dell'esito finale (Win/Loss). Valori Y=0 indicano stima corretta o sottostima.")
            else: st.info(f"Nessun dato valido (con Prob. Stimata ed Errore Sovrastima calcolabile) per questo grafico.")
            st.markdown("---")
        else: st.info(f"Le colonne '{colonna_prob}' o '{colonna_media_pt_stimati}'/'{colonna_ris_finale}' (per calcolare l'errore) non sono disponibili."); st.markdown("---")

        # Grafici 7 e 8: Probabilità vs P/L e Distribuzione Stake
        col_c, col_d = st.columns(2)
        with col_c:
             if colonna_prob in df_results_filtered_final.columns:
                 st.subheader(f"Relazione {colonna_prob} vs P/L (Win/Loss)")
                 df_plot_prob_pl = df_results_filtered_final[df_results_filtered_final[colonna_prob].notna()].copy()
                 if not df_plot_prob_pl.empty:
                     # --- MODIFICA: Usa format_currency per tooltip ---
                     df_plot_prob_pl['Stake Formattato'] = df_plot_prob_pl[colonna_stake].apply(format_currency)
                     df_plot_prob_pl['P/L Formattato'] = df_plot_prob_pl['P/L'].apply(format_currency)
                     # --- FINE MODIFICA ---
                     df_plot_prob_pl['Prob Formattata'] = df_plot_prob_pl[colonna_prob].map(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
                     fig_prob_pl = px.scatter(df_plot_prob_pl, x=colonna_prob, y='P/L', color='Esito_Standard', title="P/L vs. Probabilità Stimata (Periodo Filtrato)", labels={colonna_prob: 'Probabilità Stimata', 'P/L': 'P/L (€)', 'Esito_Standard': 'Esito'}, hover_name=colonna_sq_a, hover_data={colonna_sq_b:True, colonna_quota:':.2f', 'Stake Formattato': True, 'P/L Formattato':True, 'Prob Formattata':True, colonna_prob:False, 'P/L':False}, color_discrete_map=color_map_esiti)
                     fig_prob_pl.update_traces(marker=dict(line=dict(width=0))); fig_prob_pl.update_layout(template='plotly_white')
                     fig_prob_pl.update_xaxes(title="Probabilità Stimata", showgrid=False, tickformat=".1%")
                     # --- MODIFICA: Usa format_currency per tick asse Y P/L ---
                     if italian_locale_set:
                         fig_prob_pl.update_yaxes(title="P/L (€)", showgrid=False, tickformat="~€")
                     else:
                         fig_prob_pl.update_yaxes(title="P/L (€)", showgrid=False, ticksuffix=" €")
                     # --- FINE MODIFICA ---
                     st.plotly_chart(fig_prob_pl, use_container_width=True); st.caption("Esplora la correlazione tra probabilità stimata e P/L.")
                 else: st.info("Nessun dato W/L con Probabilità Stimata valida.")
             else: st.info(f"Colonna '{colonna_prob}' non disponibile.")

        with col_d:
              if colonna_stake in df_filtered_final.columns:
                 st.subheader(f"Distribuzione {colonna_stake} (Tutti Filtrati > 0)")
                 df_plot_stake = df_filtered_final[df_filtered_final[colonna_stake].notna() & (df_filtered_final[colonna_stake] > 0)]
                 if not df_plot_stake.empty:
                     fig_stake_hist = px.histogram(df_plot_stake, x=colonna_stake, nbins=15, title="Distribuzione Stake (€) (Periodo Filtrato)", labels={colonna_stake: 'Importo Puntato (€)'}, color_discrete_sequence=[color_line1])
                     fig_stake_hist.update_traces(marker_line_width=0); fig_stake_hist.update_layout(template='plotly_white', bargap=0.1); fig_stake_hist.update_yaxes(title="Numero Scommesse", showgrid=False)
                     # --- MODIFICA: Usa format_currency per tick asse X Stake ---
                     if italian_locale_set:
                         fig_stake_hist.update_xaxes(title="Importo Puntato (€)", showgrid=False, tickformat="~€")
                     else:
                         fig_stake_hist.update_xaxes(title="Importo Puntato (€)", showgrid=False, ticksuffix=" €")
                     # --- FINE MODIFICA ---
                     st.plotly_chart(fig_stake_hist, use_container_width=True); st.caption("Mostra come si distribuiscono gli importi puntati.")
                 else: st.info("Nessuna puntata valida (> 0 €).")
              else: st.info(f"Colonna '{colonna_stake}' non disponibile.")


        # --- Tabella Dati Filtrati ---
        st.markdown("---")
        st.subheader("Tabella Dati Completa (Filtrata e Formattata)")
        df_display_filtered = df_filtered_final.copy()
        def safe_format(value, format_str): return format(value, format_str) if pd.notna(value) else ""
        cols_to_format_percent = [colonna_prob, colonna_edge]
        # --- MODIFICA: Usa format_currency per la tabella ---
        cols_to_format_currency_in_table = [colonna_stake, 'P/L']
        # --- FINE MODIFICA ---
        cols_to_format_points = [colonna_media_pt_stimati, colonna_ris_finale, colonna_confidenza, 'Errore_Sovrastima_PT']
        for col in cols_to_format_percent:
             if col in df_display_filtered.columns: df_display_filtered[col] = pd.to_numeric(df_display_filtered[col], errors='coerce').map(lambda x: safe_format(x, '.1%'))
        # --- MODIFICA: Applica format_currency alla tabella ---
        for col in cols_to_format_currency_in_table:
             if col in df_display_filtered.columns:
                 df_display_filtered[col] = pd.to_numeric(df_display_filtered[col], errors='coerce').apply(format_currency)
        # --- FINE MODIFICA ---
        for col in cols_to_format_points:
             if col in df_display_filtered.columns: df_display_filtered[col] = pd.to_numeric(df_display_filtered[col], errors='coerce').map(lambda x: safe_format(x, '.1f'))
        if colonna_quota in df_display_filtered.columns: df_display_filtered[colonna_quota] = pd.to_numeric(df_display_filtered[colonna_quota], errors='coerce').map(lambda x: safe_format(x, '.2f'))
        if colonna_data in df_display_filtered.columns:
            if pd.api.types.is_datetime64_any_dtype(df_display_filtered[colonna_data]): df_display_filtered[colonna_data] = df_display_filtered[colonna_data].dt.strftime('%Y-%m-%d')
            else: df_display_filtered[colonna_data] = pd.to_datetime(df_display_filtered[colonna_data], errors='coerce').dt.strftime('%Y-%m-%d')

        colonne_display_ordine = [
            colonna_sq_a, colonna_sq_b, colonna_data, colonna_ora, colonna_quota,
            colonna_stake, colonna_prob, colonna_edge, colonna_confidenza,
            colonna_media_pt_stimati, colonna_ris_finale,
            'Errore_Sovrastima_PT',
            'Esito_Standard', 'P/L'
        ]
        colonne_display_esistenti = [col for col in colonne_display_ordine if col in df_display_filtered.columns]
        st.dataframe(df_display_filtered[colonne_display_esistenti])


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
    st.error(f"ERRORE CRITICO: Colonna fondamentale non trovata: {e}.")
    if 'df' in locals():
        st.info(f"Colonne lette: {df.columns.tolist()}")
    st.stop()
except Exception as e:
    # Mostra l'errore specifico ma non interrompere bruscamente se possibile
    st.error(f"ERRORE IMPREVISTO: {e}")
    # Potresti voler commentare st.exception(e) in produzione per non mostrare tutto il traceback
    st.exception(e)
    # Considera se vuoi fermare l'app o provare a continuare mostrando solo un errore
    # st.stop() # Decommenta se preferisci fermare l'app in caso di qualsiasi errore
