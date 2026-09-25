"""Run the daily risk batch and write the risk pack to outputs/.

uv run python scripts/run_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from derivatives_risk_engine.engine import run, summary_tables

OUT = Path(__file__).resolve().parent.parent / "outputs"


def fmt(df: pd.DataFrame) -> str:
    def cell(x):
        if isinstance(x, (int, float, np.number)) and not isinstance(x, bool):
            return f"{x:,.3f}" if abs(x) < 10 else f"{x:,.0f}"
        return str(x)

    return df.map(cell).to_markdown(disable_numparse=True, stralign="right")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    r = run()
    tables = summary_tables(r)
    lines = ["# Daily risk pack (synthetic book)\n", "All figures in USD.\n"]
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv")
        lines += [f"## {name}\n", fmt(df), ""]
    (OUT / "risk_pack.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
