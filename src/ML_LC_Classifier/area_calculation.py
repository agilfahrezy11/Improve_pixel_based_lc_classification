"""Fast raster class area computation and optional bar plotting."""

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Mapping, Any
import numpy as np
import pandas as pd
import rasterio

from .geometry import compute_pixel_area_ha

RasterPath = str | PathLike[str]
ClassSpec = Mapping[int, Mapping[str, Any]]


@dataclass
class ClassAreaResult:
    """Return value of :func:`calculate_class_area`.

    Attributes
    ----------
    area_ha : pandas.DataFrame
        One row per class with columns ``class_id``, ``class_name``,
        ``pixel_count``, and ``area_ha``. Sorted by ``class_id``.
    figure : object or None
        Matplotlib bar chart of area per class, or ``None`` if plotting was
        disabled or matplotlib is not installed.
    csv_path : Path or None
        Resolved path of the written CSV, or ``None`` if *csv_path* was not
        supplied.
    """

    area_ha: pd.DataFrame
    figure: Any = field(default=None)
    csv_path: Path | None = field(default=None)


def calculate_class_area(
    raster_path: RasterPath,
    class_spec: ClassSpec | None = None,
    *,
    block_size: int = 2048,
    csv_path: RasterPath | None = None,
    plot: bool = False,
    figsize: tuple[float, float] = (10.0, 6.0),
    title: str = "Land-Cover Area by Class",
    xlabel: str = "Land-Cover Class",
    ylabel: str = "Area (ha)",
) -> ClassAreaResult:
    """Calculate per-class area from a single-band land-cover raster in streamed blocks.

    Parameters
    ----------
    raster_path : str or PathLike
        Single-band uint8/integer GeoTIFF — the land-cover classification.
    class_spec : dict[int, dict[str, Any]] or None
        Optional mapping from class ID to display metadata (e.g., name, color).
    block_size : int
        Window block size for streamed processing to keep RAM usage bounded.
    csv_path : str or PathLike or None
        If provided, writes the area table to this path as UTF-8 CSV.
    plot : bool
        If True, generates a matplotlib bar chart and stores it in ``figure``.
    figsize, title, xlabel, ylabel :
        Plotting customization options when ``plot=True``.

    Returns
    -------
    ClassAreaResult
    """
    raster_path = Path(raster_path)
    total_counts = np.zeros(256, dtype=np.int64)
    total_area_per_class = np.zeros(256, dtype=np.float64)

    with rasterio.open(raster_path) as src:
        if src.count != 1:
            raise ValueError(f"Expected a single-band raster, got {src.count} bands")

        crs = src.crs
        transform = src.transform
        nodata = src.nodata
        height = src.height
        width = src.width

        is_geo = crs is not None and crs.is_geographic
        if not is_geo:
            const_pixel_area_ha = float(compute_pixel_area_ha(crs, transform, height))

        for row_start in range(0, height, block_size):
            h_win = min(block_size, height - row_start)
            if is_geo:
                row_indices = np.arange(row_start, row_start + h_win, dtype=np.float64)
                win_pixel_area_ha = compute_pixel_area_ha(crs, transform, height, row_indices=row_indices)
                # Reshape to (h_win, 1) for broadcasting across cols
                win_pixel_area_ha = np.asarray(win_pixel_area_ha).reshape(-1, 1)

            for col_start in range(0, width, block_size):
                w_win = min(block_size, width - col_start)
                window = rasterio.windows.Window(col_start, row_start, w_win, h_win)
                data = src.read(1, window=window, masked=True)

                if nodata is not None and not np.isnan(nodata):
                    valid_mask = ~data.mask & (data.data != nodata) & (data.data != 0)
                else:
                    valid_mask = ~data.mask & (data.data != 0)

                valid_labels = data.data[valid_mask]
                if valid_labels.size == 0:
                    continue

                if not np.equal(valid_labels, np.floor(valid_labels)).all():
                    raise ValueError("Raster contains non-integer valid class values")

                int_labels = valid_labels.astype(np.int64)
                if (int_labels < 0).any() or (int_labels > 255).any():
                    raise ValueError("Class values must be integers between 0 and 255")

                bincount = np.bincount(int_labels, minlength=256)
                total_counts += bincount

                if is_geo:
                    # Accumulate latitude-dependent area
                    for cid in np.unique(int_labels):
                        class_cell_mask = (data.data == cid) & valid_mask
                        total_area_per_class[cid] += float(np.sum(win_pixel_area_ha[np.where(class_cell_mask)[0]]))
                else:
                    total_area_per_class += bincount * const_pixel_area_ha

    class_ids_present = np.where(total_counts > 0)[0]
    rows_list = []
    for cid in class_ids_present:
        count = int(total_counts[cid])
        area = round(float(total_area_per_class[cid]), 4)
        spec = (class_spec or {}).get(int(cid), {})
        name = spec.get("name", f"Class {cid}")
        rows_list.append(
            {
                "class_id": int(cid),
                "class_name": name,
                "pixel_count": count,
                "area_ha": area,
            }
        )

    area_df = pd.DataFrame(
        rows_list, columns=["class_id", "class_name", "pixel_count", "area_ha"]
    ).sort_values("class_id").reset_index(drop=True)

    fig = None
    if plot:
        fig = plot_class_area(
            area_df,
            class_spec=class_spec,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    resolved_csv: Path | None = None
    if csv_path is not None:
        resolved_csv = Path(csv_path)
        resolved_csv.parent.mkdir(parents=True, exist_ok=True)
        area_df.to_csv(resolved_csv, index=False, encoding="utf-8")
        resolved_csv = resolved_csv.resolve()

    return ClassAreaResult(area_ha=area_df, figure=fig, csv_path=resolved_csv)


def plot_class_area(
    area_df: pd.DataFrame,
    class_spec: ClassSpec | None = None,
    *,
    figsize: tuple[float, float] = (10.0, 6.0),
    title: str = "Land-Cover Area by Class",
    xlabel: str = "Land-Cover Class",
    ylabel: str = "Area (ha)",
) -> Any:
    """Generate a matplotlib bar chart from a class area DataFrame."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_class_area. "
            "Install it with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=figsize)

    bar_colors = []
    for _, row in area_df.iterrows():
        spec = (class_spec or {}).get(int(row["class_id"]), {})
        bar_colors.append(spec.get("color", None))

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    resolved_colors = []
    cycle_idx = 0
    for c in bar_colors:
        if c is None:
            resolved_colors.append(prop_cycle[cycle_idx % len(prop_cycle)])
            cycle_idx += 1
        else:
            resolved_colors.append(c)

    x_positions = range(len(area_df))
    bars = ax.bar(x_positions, area_df["area_ha"], color=resolved_colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(area_df["class_name"], rotation=45, ha="right", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:,.0f}"))

    for bar, val in zip(bars, area_df["area_ha"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 1.01,
            f"{val:,.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
