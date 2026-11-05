"""Save both daily temperature series over their shared date range."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from kalix.data import load_processed_data


def main() -> None:
    root = Path(__file__).resolve().parent
    daily = load_processed_data(root)
    series = {
        "Nikkaluokta": daily["nikkaluokta_temperature_c"],
        "Tarfala": daily["tarfala_temperature_c"],
    }

    start = max(values.first_valid_index() for values in series.values())
    end = min(values.last_valid_index() for values in series.values())
    if start > end:
        raise ValueError("The temperature datasets have no shared timeframe.")

    # The prepared daily calendar preserves gaps in missing observations.
    fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
    for label, values in series.items():
        shared = values.loc[start:end]
        ax.plot(shared.index, shared, label=label, linewidth=0.6, alpha=0.75)

    ax.set(
        title=f"Daily mean temperature | {start:%Y-%m-%d} to {end:%Y-%m-%d}",
        xlabel="Date",
        ylabel="Temperature (°C)",
        xlim=(start, end),
    )
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.grid(alpha=0.2)
    ax.legend()

    output = root / "outputs" / "figures" / "temperature_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
