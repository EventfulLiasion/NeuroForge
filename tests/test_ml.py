def test_psi():
    import pandas as pd
    from src.ml.drift import psi, drift_report
    s = pd.Series([1, 2, 3, 4, 5] * 20)
    assert psi(s, s) < 0.05
    rep = drift_report(pd.DataFrame({"a": s}), pd.DataFrame({"a": s * 10}))
    assert "a" in rep

def test_tabular_train_small():
    import pandas as pd
    from src.ml.train import train_tabular
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=100), "x2": rng.normal(size=100),
                       "cat": rng.choice(["a", "b"], 100),
                       "y": rng.choice([0, 1], 100)})
    _, m = train_tabular(df, "y", test_size=0.3, n_trials=2, out_model="data/test_model.pkl")
    assert "f1_weighted" in m
