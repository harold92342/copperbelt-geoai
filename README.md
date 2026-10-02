# copperbelt-geoai
Geochemical anomaly detection for Cu-Co exploration - DRC Copperbelt

## Overview
Isolation Forest anomaly detection over DRC mining districts (copper, gold, zinc, nickel
mine counts), surfaced in an interactive Streamlit dashboard.

## Structure
- `data/` cleaned Africa mining districts dataset
- `notebooks/` exploratory analysis
- `src/` data loading helpers
- `app/dashboard.py` Streamlit app

## Run locally
```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```
