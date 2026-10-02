"""Isolation Forest anomaly detection for DRC mining districts."""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

METALS = ['copper_mine', 'gold_mine', 'zinc_mine', 'nickel_mine']


def detect_anomalies(df: pd.DataFrame, metals, contamination: float = 0.10,
                     random_state: int = 42) -> pd.DataFrame:
    """Return a copy of df with 'anomaly' (-1/1) and 'label' (Target/Background)."""
    if not metals:
        raise ValueError("At least one metal is required")
    out = df.copy()
    scaled = StandardScaler().fit_transform(out[list(metals)].fillna(0))
    model = IsolationForest(contamination=contamination, random_state=random_state)
    out['anomaly'] = model.fit_predict(scaled)
    out['label'] = out['anomaly'].map({-1: 'Target', 1: 'Background'})
    return out
