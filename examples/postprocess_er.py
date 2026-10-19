"""Edit paths and the sample interval for a labelled high-resolution ER table"""

from pathlib import Path

from moonbroom import plot_timeseries, read_er, select_time_window, write_csv


input_file = Path("data/er/EH990301.xml")
out_dir = Path("output/er")
start = "1999-03-01T00:00:00Z"
stop = "1999-03-02T23:59:59Z"
columns = ["eflux1", "eflux2"]

if __name__ == "__main__":
    df = select_time_window(read_er(input_file), start, stop)
    write_csv(df, out_dir / f"{input_file.stem}.csv")
    ax = plot_timeseries(df, columns, ylabel="Electron flux", title="Lunar Prospector ER")
    ax.figure.savefig(out_dir / f"{input_file.stem}.png", dpi=200, bbox_inches="tight")
