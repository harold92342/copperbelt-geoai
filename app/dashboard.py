import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
st.set_page_config(page_title="Copperbelt GeoAI", page_icon="⛏️", layout="wide")

st.title("Copperbelt GeoAI — DRC Exploration Dashboard")
st.markdown("Geochemical anomaly detection for Cu-Co exploration · Katanga, DRC")

@st.cache_data
def load_data():
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'Mines_Africa_Districts_cleaned.csv'))
    return df[df['countryname'] == 'Democratic Republic of the Congo']

drc = load_data()

# Sidebar
st.sidebar.header("Model settings")
contamination = st.sidebar.slider("Anomaly sensitivity", 0.05, 0.30, 0.10)
metals = st.sidebar.multiselect(
    "Metals to analyze",
    ['copper_mine','gold_mine','zinc_mine','nickel_mine'],
    default=['copper_mine','gold_mine','zinc_mine','nickel_mine']
)

# Model
drc2 = drc.copy()
scaler = StandardScaler()
scaled = scaler.fit_transform(drc2[metals].fillna(0))
model = IsolationForest(contamination=contamination, random_state=42)
drc2['anomaly'] = model.fit_predict(scaled)
drc2['label'] = drc2['anomaly'].apply(lambda x: 'Target' if x == -1 else 'Background')

anomalies = drc2[drc2['label'] == 'Target']

# Metrics row
col1, col2, col3 = st.columns(3)
col1.metric("Total DRC districts", len(drc2))
col2.metric("Exploration targets", len(anomalies))
col3.metric("Katanga targets", len(anomalies[anomalies['ADM1']=='Katanga']))

st.divider()

# Chart
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Anomaly scatter plot")
    fig, ax = plt.subplots(figsize=(7,5))
    colors = drc2['label'].map({'Target':'#E24B4A','Background':'#378ADD'})
    ax.scatter(drc2['copper_mine'], drc2['gold_mine'], c=colors, alpha=0.7, s=60)
    for _, row in anomalies.iterrows():
        ax.annotate(row['ADM2'], (row['copper_mine'], row['gold_mine']),
                   fontsize=8, xytext=(5,3), textcoords='offset points')
    ax.set_xlabel('Copper mines')
    ax.set_ylabel('Gold mines')
    ax.set_title('DRC districts — Isolation Forest')
    st.pyplot(fig)

with col_right:
    st.subheader("Top exploration targets")
    st.dataframe(
        anomalies[['ADM1','ADM2','copper_mine','gold_mine','label']]
        .sort_values('copper_mine', ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )