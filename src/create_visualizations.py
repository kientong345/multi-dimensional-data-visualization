"""
Script to generate multi‑dimensional data visualizations for the
CMP_SC‑8630 data visualization assignment.  The script loads three
real‑world datasets related to climate and hydrology and produces
visualizations that explore patterns across multiple variables and
dimensions.  The resulting figures are saved to the ``output``
directory.  The datasets used here include:

* ``weather_data.csv`` – daily weather observations for multiple
  cities in New Zealand (2016–2017) containing temperature,
  humidity, wind, pressure and precipitation variables.  Source:
  mosaicData package within the Rdatasets collection.
* ``global_temp.csv`` – NASA Goddard Institute for Space Studies
  (GISTEMP) global land–ocean temperature anomalies from 1880 to
  2025.  Monthly anomalies relative to the 1951–1980 baseline are
  provided.  Source: NASA GISS via data.giss.nasa.gov.
* ``minnesota_weather.csv`` – monthly weather summary for six
  Minnesota agricultural sites (1927–1936) including cooling and
  heating degree days, precipitation and temperature extremes.
  Source: agridat package within Rdatasets.

The visualizations include heatmaps, scatter plots and line charts
to illustrate how variables such as temperature, humidity and
precipitation vary over time and across different locations.
"""

import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np


def ensure_output_dir(path: str) -> None:
    """Ensure that the output directory exists."""
    os.makedirs(path, exist_ok=True)


def plot_weather_heatmap(df: pd.DataFrame, outdir: str) -> str:
    """Create a heatmap of average temperature by city and month.

    Parameters
    ----------
    df : pandas.DataFrame
        Weather data with columns ``city``, ``month`` and ``avg_temp``.
    outdir : str
        Directory to write the output image.

    Returns
    -------
    str
        Path to the saved figure.
    """
    # Compute average monthly temperature for each city
    monthly_avg = df.groupby(["city", "month"])["avg_temp"].mean().reset_index()

    # Pivot: city as index, month as columns, avg temp as values
    pivot = monthly_avg.pivot(index="city", columns="month", values="avg_temp")

    # Sort columns in calendar order (1–12)
    pivot = pivot[sorted(pivot.columns)]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="coolwarm",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Average temperature"},
    )

    # Labels and title
    ax.set_title("Average monthly temperature by city", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("City")

    # Save figure
    out_path = os.path.join(outdir, "weather_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_weather_scatter(df: pd.DataFrame, outdir: str) -> str:
    """Create a scatter plot exploring relationships between humidity,
    temperature and precipitation.

    Each point represents a daily observation.  The x‑axis shows
    average humidity, the y‑axis shows average temperature in Fahrenheit,
    the marker size encodes precipitation and colour encodes the city.
    Separate legends are provided for city and precipitation to avoid
    overlap.

    Parameters
    ----------
    df : pandas.DataFrame
        Weather data with columns ``avg_humidity``, ``avg_temp``,
        ``precip`` and ``city``.
    outdir : str
        Directory to write the output image.

    Returns
    -------
    str
        Path to the saved figure.
    """
    # Clean 'precip': coerce to numeric, fill NaN with 0.0
    df = df.copy()
    df["precip"] = pd.to_numeric(df["precip"], errors="coerce").fillna(0.0)

    # Set up figure
    fig, ax = plt.subplots(figsize=(9, 6))

    # Marker size range
    size_range = (20, 300)

    # Determine fixed order for cities
    cities = sorted(df["city"].unique())

    # Scatter plot (legend=False so we draw custom legends)
    sns.scatterplot(
        data=df,
        x="avg_humidity",
        y="avg_temp",
        hue="city",
        hue_order=cities,
        size="precip",
        sizes=size_range,
        alpha=0.65,
        ax=ax,
        legend=False,
    )

    # Custom legend for cities (hue)
    palette = sns.color_palette(n_colors=df["city"].nunique())
    
    city_handles = [
        mlines.Line2D(
            [], [],
            marker="o",
            color="w",
            markerfacecolor=palette[i],
            markersize=8,
            label=city,
        )
        for i, city in enumerate(cities)
    ]
    legend_city = ax.legend(
        handles=city_handles,
        title="City",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )
    ax.add_artist(legend_city)

    # Custom legend for precipitation sizes
    precip_max = df["precip"].max()
    precip_vals = np.linspace(0, precip_max, 4)
    size_handles = []
    for pv in precip_vals:
        mapped_size = float(np.interp(pv, [0, precip_max], list(size_range)))
        h = ax.scatter(
            [], [],
            s=mapped_size,
            color="gray",
            alpha=0.6,
            label=f"{pv:.2f}",
        )
        size_handles.append(h)
    ax.legend(
        handles=size_handles,
        title="Precipitation",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        frameon=True,
    )

    # Labels and title
    ax.set_xlabel("Average relative humidity (%)")
    ax.set_ylabel("Average temperature (°F)")
    ax.set_title(
        "Daily weather: temperature vs humidity with precipitation (size)",
        fontsize=12,
        fontweight="bold",
    )

    # Save figure
    out_path = os.path.join(outdir, "weather_scatter.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_global_temp_heatmap(df: pd.DataFrame, outdir: str) -> str:
    """Create a heatmap of global temperature anomalies by year and month.

    Parameters
    ----------
    df : pandas.DataFrame
        Global temperature anomalies where rows correspond to years and
        columns to months (Jan–Dec).  The DataFrame should include
        numeric values for anomalies.  Missing values are allowed and
        will appear as blank cells.
    outdir : str
        Directory to write the output image.

    Returns
    -------
    str
        Path to the saved figure.
    """
    month_abbrs = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Melt wide → long, keeping only month columns Jan–Dec
    long_df = df.melt(
        id_vars=["Year"],
        value_vars=month_abbrs,
        var_name="Month",
        value_name="Anomaly",
    )

    # Map month abbreviation → month number
    month_map = {m: i + 1 for i, m in enumerate(month_abbrs)}
    long_df["MonthNum"] = long_df["Month"].map(month_map)

    # Pivot back to matrix: Year × MonthNum
    matrix = long_df.pivot(index="Year", columns="MonthNum", values="Anomaly")

    # Sort by year ascending
    matrix = matrix.sort_index(ascending=True)

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw heatmap
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="coolwarm",
        vmin=-1.5,
        vmax=1.5,
        linewidths=0,
        linecolor="white",
        cbar_kws={"label": "Temperature anomaly (°C relative to 1951–1980)"},
    )

    # Customize x-ticks: month abbreviations, rotated 45°
    ax.set_xticks(np.arange(len(month_abbrs)) + 0.5)
    ax.set_xticklabels(month_abbrs, rotation=45, ha="right")

    # Labels and title
    ax.set_title(
        "Global land–ocean temperature anomalies (1880–2025)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    # Save figure
    out_path = os.path.join(outdir, "global_temp_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_minnesota_precip_line(df: pd.DataFrame, outdir: str) -> str:
    """Create a line chart of monthly precipitation by site over time.

    This figure shows how precipitation varies across the six Minnesota
    sites from 1927 to 1936.  Each line corresponds to a site and
    month; values are aggregated by year and month.

    Parameters
    ----------
    df : pandas.DataFrame
        Minnesota weather data with columns ``site``, ``year``, ``mo`` (month) and
        ``precip``.
    outdir : str
        Directory to write the output image.

    Returns
    -------
    str
        Path to the saved figure.
    """
    df = df.copy()

    # Create 'date' column from year and month (day=1)
    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["mo"], day=1)
    )

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Line plot
    sns.lineplot(
        data=df,
        x="date",
        y="precip",
        hue="site",
        ax=ax,
    )

    # Labels and title
    ax.set_xlabel("Year")
    ax.set_ylabel("Precipitation (inches)")

    # Title
    ax.set_title(
        "Monthly precipitation by Minnesota site (1927–1936)",
        fontsize=13,
        fontweight="bold",
    )

    # Legend outside plot
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Site",
        frameon=True,
    )

    # Save figure
    out_path = os.path.join(outdir, "minnesota_precip_line.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> List[str]:
    """Run all visualizations and return a list of generated file paths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "output")
    ensure_output_dir(out_dir)
    figures: List[str] = []

    # Load and plot weather data
    weather_path = os.path.join(data_dir, "weather_data.csv")
    weather_df = pd.read_csv(weather_path)
    # Plot heatmap and scatter
    figures.append(plot_weather_heatmap(weather_df, out_dir))
    figures.append(plot_weather_scatter(weather_df, out_dir))

    # Load and plot global temperature anomalies
    global_path = os.path.join(data_dir, "global_temp.csv")
    global_df = pd.read_csv(global_path, skiprows=1)
    # Replace *** with NA and convert to numeric
    global_df = global_df.replace("***", pd.NA)
    for col in global_df.columns[1:]:
        global_df[col] = pd.to_numeric(global_df[col], errors="coerce")
    figures.append(plot_global_temp_heatmap(global_df, out_dir))

    # Load and plot Minnesota weather data
    minn_path = os.path.join(data_dir, "minnesota_weather.csv")
    minn_df = pd.read_csv(minn_path)
    figures.append(plot_minnesota_precip_line(minn_df, out_dir))
    return figures


if __name__ == "__main__":
    generated = main()
    print("Generated figures:")
    for path in generated:
        print(path)