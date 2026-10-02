import numpy as np
import pandas as pd
import pytest

from src.model import detect_anomalies


def _df(n=50):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'copper_mine': rng.integers(0, 3, n), 'gold_mine': rng.integers(0, 3, n)})
    df.loc[0, ['copper_mine', 'gold_mine']] = [100, 100]
    return df


def test_labels_and_outlier_flagged():
    out = detect_anomalies(_df(), ['copper_mine', 'gold_mine'], contamination=0.05)
    assert set(out['label']) <= {'Target', 'Background'}
    assert out.loc[0, 'label'] == 'Target'


def test_input_not_mutated():
    df = _df()
    detect_anomalies(df, ['copper_mine'])
    assert 'label' not in df.columns


def test_empty_metals_raises():
    with pytest.raises(ValueError):
        detect_anomalies(_df(), [])
