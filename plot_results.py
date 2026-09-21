"""Generate equity_curves.png from results_curves.csv (produced by pairs_trading.py)."""
import pandas as pd
import matplotlib.pyplot as plt

curves = pd.read_csv("results_curves.csv", index_col=0, parse_dates=True)

fig, ax = plt.subplots(figsize=(9, 5))
for col in curves.columns:
    ax.plot(curves.index, curves[col], label=col, linewidth=1.4)

ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
ax.set_title("Out-of-sample cumulative equity (net of costs)")
ax.set_ylabel("Portfolio value (base 1.0)")
ax.legend(loc="lower left")
ax.grid(alpha=0.25)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("equity_curves.png", dpi=150)
print("saved equity_curves.png")
