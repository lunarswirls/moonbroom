"""Time-series plotting without event annotations or import-time figures"""

from .processing import select_time_window


def plot_timeseries(df, columns, *, start=None, stop=None, utc_col="utc",
                    ylabel=None, title=None, ax=None):
    """Plot named MAG or ER columns and return the axes without showing a window"""
    import matplotlib.pyplot as plt

    if isinstance(columns, str):
        columns = [columns]
    if not columns:
        raise ValueError("Select at least one column")
    selected = select_time_window(df, start, stop, utc_col=utc_col).sort_values(utc_col)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    for column in columns:
        ax.plot(selected[utc_col], selected[column], label=column)
    ax.set_xlabel("Time [UTC]")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax
