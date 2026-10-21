#!/usr/bin/env python3
"""Publication-quality flood susceptibility maps for four study events.

Reads existing M21 susceptibility GeoTIFFs.
Does not retrain the model or modify the source rasters.

Outputs:
  outputs/figures/m21_susceptibility_<event>.png
  outputs/figures/m21_susceptibility_<event>.pdf
  outputs/figures/m21_susceptibility_all_events.png
  outputs/figures/m21_susceptibility_all_events.pdf
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "outputs" / "susceptibility"
OUTPUT_DIR = ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS = [
    ("assam_2020", "Assam", "2020"),
    ("assam_2022", "Assam", "2022"),
    ("bihar_2020", "Bihar", "2020"),
    ("kerala_2018", "Kerala", "2018"),
]

NODATA_FALLBACK = -9999.0
CMAP_NAME = "YlOrRd"
DPI = 300

# Consistent typography and clean paper-ready styling.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "#edf0f2",
    "savefig.facecolor": "white",
    "savefig.dpi": DPI,
    "axes.edgecolor": "#53616c",
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

cmap = plt.get_cmap(CMAP_NAME).copy()
cmap.set_bad("#e7ebef")

NORM = Normalize(vmin=0.0, vmax=1.0)


def read_reprojected_raster(path: Path):
    """Reproject an existing probability raster to geographic coordinates."""

    with rasterio.open(path) as src:
        src_data = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else NODATA_FALLBACK

        invalid = (
            ~np.isfinite(src_data)
            | (src_data == nodata)
            | (src_data < 0.0)
            | (src_data > 1.0)
        )
        src_data[invalid] = nodata

        dst_crs = "EPSG:4326"
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        dst_data = np.full(
            (dst_height, dst_width),
            nodata,
            dtype=np.float32,
        )

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )

        mask = (
            ~np.isfinite(dst_data)
            | (dst_data == nodata)
            | (dst_data < 0.0)
            | (dst_data > 1.0)
        )
        data = np.ma.array(dst_data, mask=mask)

        left = dst_transform.c
        top = dst_transform.f
        right = left + dst_width * dst_transform.a
        bottom = top + dst_height * dst_transform.e

        extent = [left, right, bottom, top]

        return data, extent, src.crs, src.width, src.height


def style_axis(ax, title, extent):
    """Apply consistent cartographic styling."""

    ax.set_title(title, loc="left", pad=10)

    ax.set_xlabel("Longitude (°)", labelpad=5)
    ax.set_ylabel("Latitude (°)", labelpad=5)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.2f}°")
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:.2f}°")
    )

    ax.tick_params(
        direction="out",
        length=3,
        width=0.6,
        colors="#34404a",
        pad=4,
    )

    ax.grid(
        True,
        color="white",
        linewidth=0.65,
        linestyle="--",
        alpha=0.8,
        zorder=3,
    )

    for spine in ax.spines.values():
        spine.set_color("#667580")
        spine.set_linewidth(0.7)


def add_colorbar(fig, mappable, cax, compact=False):
    """Use the same probability scale for every event."""

    ticks = np.linspace(0, 1, 6)

    cb = fig.colorbar(
        mappable,
        cax=cax,
        orientation="vertical",
        ticks=ticks,
    )

    cb.set_label(
        "Predicted probability of\nnew-open-water label",
        labelpad=10,
        fontsize=9 if not compact else 8,
    )

    cb.ax.tick_params(labelsize=8 if not compact else 7, length=3)
    cb.outline.set_linewidth(0.7)

    return cb


def save_figure(fig, stem):
    """Save a high-resolution raster image and a vector PDF."""

    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.12,
    )

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.12,
    )

    print(f"Saved: {png_path.relative_to(ROOT)}")
    print(f"Saved: {pdf_path.relative_to(ROOT)}")


def main():
    loaded = []
    missing = []

    for event, region, year in EVENTS:
        path = INPUT_DIR / f"m21_susceptibility_{event}.tif"

        if not path.exists():
            missing.append(str(path))
            continue

        try:
            data, extent, crs, width, height = read_reprojected_raster(path)

            if data.count() == 0:
                raise ValueError("Raster contains no valid probability pixels")

            loaded.append({
                "event": event,
                "region": region,
                "year": year,
                "data": data,
                "extent": extent,
                "source_crs": crs,
                "width": width,
                "height": height,
            })

            print(
                f"Loaded {event}: {width:,} x {height:,}; "
                f"display-valid pixels={data.count():,}"
            )

        except Exception as exc:
            warnings.warn(f"Could not read {path}: {exc}")

    if missing:
        raise FileNotFoundError(
            "Missing expected raster(s):\n" + "\n".join(missing)
        )

    if len(loaded) != len(EVENTS):
        raise RuntimeError(
            f"Expected {len(EVENTS)} event rasters; loaded {len(loaded)}."
        )

    # ---------------------------------------------------------------
    # Individual event maps
    # ---------------------------------------------------------------
    for item in loaded:
        fig = plt.figure(figsize=(8.2, 7.0), constrained_layout=True)

        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[1, 0.045],
            wspace=0.04,
        )

        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        image = ax.imshow(
            item["data"],
            extent=item["extent"],
            origin="upper",
            cmap=cmap,
            norm=NORM,
            interpolation="nearest",
            aspect="auto",
            zorder=1,
        )

        title = f'{item["region"]} | {item["year"]}'
        style_axis(ax, title, item["extent"])

        fig.suptitle(
            "Flood susceptibility",
            x=0.09,
            y=1.035,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )

        fig.text(
            0.09,
            0.985,
            "M21 Extra Trees · 21 predictors · 90 m source resolution",
            ha="left",
            va="top",
            fontsize=9,
            color="#56616b",
        )

        add_colorbar(fig, image, cax)

        fig.text(
            0.09,
            -0.015,
            "Gray background indicates NoData. Predictions represent "
            "agreement with a Sentinel-1-derived new-open-water label; "
            "they are not independently verified flood probabilities.",
            fontsize=7.5,
            color="#4e5962",
            ha="left",
            wrap=True,
        )

        save_figure(fig, f'm21_susceptibility_{item["event"]}')
        plt.close(fig)

    # ---------------------------------------------------------------
    # Four-event comparison: one common color scale
    # ---------------------------------------------------------------
    fig = plt.figure(figsize=(14.2, 11.0), constrained_layout=True)

    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 0.055],
        height_ratios=[1, 1],
        wspace=0.09,
        hspace=0.16,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    cax = fig.add_subplot(gs[:, 2])
    last_image = None

    for ax, item in zip(axes, loaded):
        last_image = ax.imshow(
            item["data"],
            extent=item["extent"],
            origin="upper",
            cmap=cmap,
            norm=NORM,
            interpolation="nearest",
            aspect="auto",
            zorder=1,
        )

        title = f'{item["region"]} | {item["year"]}'
        style_axis(ax, title, item["extent"])

    fig.suptitle(
        "Flood susceptibility across four flood events",
        fontsize=19,
        fontweight="bold",
        x=0.055,
        ha="left",
    )

    fig.text(
        0.057,
        0.955,
        "M21 Extra Trees · Common probability scale (0–1) · "
        "90 m source rasters",
        fontsize=10,
        color="#56616b",
        ha="left",
    )

    add_colorbar(fig, last_image, cax, compact=True)

    fig.text(
        0.057,
        0.012,
        "Gray background indicates NoData. The model predicts agreement "
        "with a Sentinel-1-derived new-open-water label; independent "
        "flood-extent validation is still required.",
        fontsize=8,
        color="#4e5962",
        ha="left",
        wrap=True,
    )

    save_figure(fig, "m21_susceptibility_all_events")
    plt.close(fig)

    print("\nAll susceptibility figures regenerated.")
    print("Source GeoTIFFs and trained model were not modified.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
