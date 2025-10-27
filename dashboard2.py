# SkillQuest Dashboard Application
# Developed using Streamlit, Pandas, and Plotly for visualizing real-time session enrollment data.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Tableau de Bord SkillQuest",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Loads, cleans, and aggregates the session data from the Excel file."""
    
    # IMPORTANT: The Excel file must be present in the deployment directory.
    excel_file_name = "data.xlsx"
    sheet_name = "Sheet1" 
    
    try:
        df_raw_sessions = pd.read_excel(excel_file_name, sheet_name=sheet_name)

        # Rename columns for internal consistency
        df_sessions_clean = df_raw_sessions.rename(columns={
            'Intervenant': 'Enseignant',
            'Nbre places': 'Capacité',
            'Inscrits': 'Inscriptions'
        })

        # Convert date column to date object
        df_sessions_clean['Date'] = pd.to_datetime(df_sessions_clean['Date']).dt.date
        
        # Handle missing 'Enseignant' values: assign to 'Autonomie'
        df_sessions_clean['Enseignant'] = df_sessions_clean['Enseignant'].fillna('Autonomie').astype(str)
        
        # Standardize capitalization and correct common duplicates/typos
        df_sessions_clean['Enseignant'] = df_sessions_clean['Enseignant'].apply(
            lambda x: x.title() if x.lower() != 'autonomie' else x
        ).str.strip()
        
        # Specific name correction logic (as previously established)
        df_sessions_clean['Enseignant'] = df_sessions_clean['Enseignant'].replace({
            'De Araujo': 'Hamilton De Araujo',
            'Hamilton Araujo': 'Hamilton De Araujo',
            'Antoina Jabbour': 'Antonia Jabbour',
            'Antonia  Jabbour': 'Antonia Jabbour',
        })
        df_sessions_clean.loc[
            df_sessions_clean['Enseignant'].str.lower().str.contains('jabbour', na=False) & 
            ~df_sessions_clean['Enseignant'].str.lower().str.contains('autonomie', na=False),
            'Enseignant'
        ] = 'Antonia Jabbour'
        
        # Ensure numerical columns are integers
        df_sessions_clean['Capacité'] = pd.to_numeric(df_sessions_clean['Capacité'], errors='coerce').fillna(0).astype(int)
        df_sessions_clean['Inscriptions'] = pd.to_numeric(df_sessions_clean['Inscriptions'], errors='coerce').fillna(0).astype(int)
        
        # Filter out sessions with zero capacity AND zero enrollments
        df_sessions_clean = df_sessions_clean[
            (df_sessions_clean['Capacité'] > 0) | (df_sessions_clean['Inscriptions'] > 0)
        ]
        
        # Aggregate data by Date and Enseignant (important for daily/teacher metrics)
        df_sessions_agg = df_sessions_clean.groupby(['Date', 'Enseignant']).agg(
            Capacité=('Capacité', 'sum'),
            Inscriptions=('Inscriptions', 'sum')
        ).reset_index()

    except FileNotFoundError:
        st.error(f"Le fichier de données '{excel_file_name}' est introuvable. Veuillez vous assurer qu'il est dans le même dossier que l'application Streamlit.")
        st.stop()
    except ValueError as e:
        st.error(f"Erreur de lecture du fichier Excel: {e}. Vérifiez si la feuille de calcul s'appelle bien '{sheet_name}'.")
        st.stop()
    
    return df_sessions_agg, df_sessions_clean

# Load data
df_sessions, df_sessions_clean = load_data()


# --- 2. General KPI Calculation ---

total_capacity = df_sessions['Capacité'].sum()
total_enrollments = df_sessions['Inscriptions'].sum()
general_filling_rate = (total_enrollments / total_capacity) if total_capacity > 0 else 0
total_sessions_agg_count = df_sessions.shape[0]

# Split for Autonomie vs. Présentiel
df_presentiel = df_sessions[df_sessions['Enseignant'] != 'Autonomie']
df_autonomie = df_sessions[df_sessions['Enseignant'] == 'Autonomie']

inscrits_presentiel = df_presentiel['Inscriptions'].sum()
inscrits_autonomie = df_autonomie['Inscriptions'].sum()


# --- 3. Header and KPI Display ---

st.title("🎯 Tableau de Bord de Suivi des Inscriptions SkillQuest")
st.markdown("Vue d'ensemble et analyse des indicateurs clés de performance (KPI) basés sur le fichier **`data.xlsx`**.")
st.markdown("---")

st.header("🔑 Métriques Globales de Performance")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total des Sessions Planifiées",
        value=f"{total_sessions_agg_count:,}", 
    )

with col2:
    st.metric(
        label="Taux de Remplissage Général des Activités", 
        value=f"{general_filling_rate:.1%}",
    )
    
with col3:
    st.metric(
        label="Inscriptions Présentiel", 
        value=f"{inscrits_presentiel:,}",
    )

with col4:
    st.metric(
        label="Inscriptions Autonomie",
        value=f"{inscrits_autonomie:,}",
    )

st.markdown("---")


# --- 4. Temporal Tracking (Trends) ---

st.header("📈 Suivi Temporel des Inscriptions")

# Intervenant filter setup
all_intervenants = sorted(df_sessions['Enseignant'].unique())
all_intervenants.insert(0, 'Tous les Intervenants') 

selected_intervenant = st.selectbox(
    'Filtrer les données temporelles par Intervenant:',
    all_intervenants,
    index=0 
)

# Apply filter
if selected_intervenant == 'Tous les Intervenants':
    df_sessions_filtered = df_sessions.copy()
else:
    df_sessions_filtered = df_sessions[df_sessions['Enseignant'] == selected_intervenant].copy()
    
# Aggregate by date for plotting
df_time = df_sessions_filtered.groupby('Date').agg(
    Total_Inscriptions=('Inscriptions', 'sum'),
    Total_Capacite=('Capacité', 'sum'),
).reset_index()

# Merge Autonomie/Presentiel splits for global view only
if selected_intervenant == 'Tous les Intervenants':
    df_autonomie_daily = df_autonomie.groupby('Date')['Inscriptions'].sum().rename('Inscrits_Autonomie')
    df_presentiel_daily = df_presentiel.groupby('Date')['Inscriptions'].sum().rename('Inscrits_Presentiel')

    df_time = df_time.merge(df_autonomie_daily, on='Date', how='left').fillna(0)
    df_time = df_time.merge(df_presentiel_daily, on='Date', how='left').fillna(0)


# Filter for days with enrollments > 0
df_time = df_time[df_time['Total_Inscriptions'] > 0].copy()

# Recalculate filling rate for the filtered dataset
df_time['Taux_Remplissage'] = np.where(
    df_time['Total_Capacite'] > 0,
    df_time['Total_Inscriptions'] / df_time['Total_Capacite'],
    0
)
df_time.fillna(0, inplace=True)


# Chart 1: Capacity and Enrollments Evolution
st.subheader("1. Évolution Capacité et Inscriptions")

chart_title_evol = f"Évolution des Inscriptions vs. Capacité Totale des Sessions ({selected_intervenant})"

fig1 = px.line(
    df_time, 
    x='Date', 
    y=['Total_Inscriptions', 'Total_Capacite'], 
    title=chart_title_evol,
    labels={'value': 'Nombre', 'Date': 'Date', 'variable': 'Métrique'},
    color_discrete_map={'Total_Inscriptions': '#3498DB', 'Total_Capacite': '#E74C3C'}
)
fig1.update_layout(hovermode="x unified")
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Autonomie vs. Présentiel Proportion (Global Only)
st.subheader("2. Évolution Autonomie vs. Présentiel (Proportion)")
if selected_intervenant == 'Tous les Intervenants':
    
    df_time_long = df_time[['Date', 'Inscrits_Autonomie', 'Inscrits_Presentiel']].melt(
        id_vars=['Date'], 
        value_vars=['Inscrits_Autonomie', 'Inscrits_Presentiel'],
        var_name='Modalité',
        value_name='Inscriptions'
    )
    df_time_long['Modalité'] = df_time_long['Modalité'].replace({
        'Inscrits_Autonomie': 'Autonomie',
        'Inscrits_Presentiel': 'Présentiel'
    })

    fig_time_split = px.area(
        df_time_long,
        x='Date',
        y='Inscriptions',
        color='Modalité',
        title='Proportion Quotidienne des Inscriptions (Autonomie vs. Présentiel)',
        labels={'Inscriptions': 'Proportion (%)', 'Date': 'Date', 'Modalité': 'Modalité'},
        color_discrete_map={'Autonomie': '#F39C12', 'Présentiel': '#2ECC71'},
        groupnorm='percent'
    )

    fig_time_split.update_layout(
        yaxis_tickformat=".0%",
        hovermode="x unified",
        yaxis_title='Proportion Quotidienne (%)'
    )
    st.plotly_chart(fig_time_split, use_container_width=True)
else:
    st.info("Ce graphique est désactivé lorsque vous filtrez par un Intervenant spécifique, car il est conçu pour montrer la répartition globale.")

st.markdown("---")


# --- 5. Performance by Teacher ---

st.header("👤 Performance des Activités par Enseignant")
st.markdown("Analyse des performances des sessions encadrées par chaque intervenant (exclut 'Autonomie').")

# Calculate metrics per teacher (excluding 'Autonomie')
df_intervenants_raw = df_sessions_clean[df_sessions_clean['Enseignant'] != 'Autonomie'].copy()

df_performance = df_intervenants_raw.groupby('Enseignant').agg(
    Total_Capacite=('Capacité', 'sum'),
    Total_Inscriptions=('Inscriptions', 'sum'),
    Nombre_Sessions_Proposees=('Enseignant', 'size')
).reset_index()

# Filter out teachers with no capacity and no enrollment
df_performance = df_performance[(df_performance['Total_Capacite'] > 0) | (df_performance['Total_Inscriptions'] > 0)]

# Calculate filling rate
df_performance['Taux_Remplissage'] = np.where(
    df_performance['Total_Capacite'] > 0,
    df_performance['Total_Inscriptions'] / df_performance['Total_Capacite'],
    0
)


# Chart 3: Total Enrolled Students
st.subheader("3. Visualisation du Nombre d'Étudiants Formés")
fig3 = px.bar(
    df_performance.sort_values('Total_Inscriptions', ascending=False),
    x='Enseignant', 
    y='Total_Inscriptions',
    orientation='v', 
    title="Nombre d'Étudiants Formés (Total Inscriptions) par Enseignant",
    labels={'Total_Inscriptions': "Nombre d'Étudiants Formés", 'Enseignant': 'Enseignant'},
    text_auto=True,
    color='Total_Inscriptions',
    color_continuous_scale=px.colors.sequential.Viridis,
    hover_data={
        'Total_Inscriptions': True, # CORRECTION: Utilisation du nom de colonne exact.
        'Nombre_Sessions_Proposees': True,
        'Taux_Remplissage': ':.1%',
        'Enseignant': False 
    }
)
fig3.update_yaxes(title="Nombre d'Étudiants Formés") 
fig3.update_xaxes(title='Enseignant') 
fig3.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.8)
st.plotly_chart(fig3, use_container_width=True)

# Table 4: Detailed Metrics by Teacher
st.subheader("4. Détail des Métriques par Intervenant (Tableau Triable)")

df_table_display = df_performance.sort_values(by='Taux_Remplissage', ascending=False).rename(columns={
    'Enseignant': 'Intervenant',
    'Total_Inscriptions': 'Inscriptions Totales',
    'Total_Capacite': 'Capacité Totale',
    'Nombre_Sessions_Proposees': 'Nombre de Sessions',
    'Taux_Remplissage': 'Taux de Remplissage'
})

df_table_display['Taux de Remplissage'] = df_table_display['Taux de Remplissage'].apply(lambda x: f"{x:.1%}")

df_final_table = df_table_display[['Intervenant', 'Nombre de Sessions', 'Capacité Totale', 'Inscriptions Totales', 'Taux de Remplissage']]

st.dataframe(df_final_table, use_container_width=True, hide_index=True)
st.markdown("---")
