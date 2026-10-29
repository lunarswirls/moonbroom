import pandas as pd
import pytest

from moonbroom import plot_timeseries


def test_plot_obeys_window_and_returns_axes(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    df = pd.DataFrame({"utc": ["1999-03-01", "1999-03-02", "1999-03-03"], "field": [1, 2, 3]})
    ax = plot_timeseries(df, "field", start="1999-03-02", stop="1999-03-02", ylabel="nT")
    assert ax.lines[0].get_ydata().tolist() == [2]
    path = tmp_path / "field.png"
    ax.figure.savefig(path)
    assert path.stat().st_size > 0
    plt.close(ax.figure)
