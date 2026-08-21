#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3
"""
Create the manuscript reference map showing:

(a) Arabian Peninsula study cities and regional geography,
(b) ENSO monitoring regions, including Niño 3.4,
(c) Indian Ocean Dipole western and eastern poles.

The Arabian Peninsula panel uses ETOPO 2022 elevation/bathymetry.
The script produces the manuscript reference-map PDF.

Example
-------
From the repository root:

    python map_generation/reference_map.py

Optional custom paths:

    python map_generation/reference_map.py \
        --etopo /path/to/ETOPO_2022_v1_60s_N90W180_bed.nc \
        --output /path/to/reference_map.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import xarray as xr


# ============================================================
# MAP PROJECTIONS
# ============================================================

GEO = ccrs.PlateCarree()
PACIFIC_PROJ = ccrs.PlateCarree(central_longitude=180)


# ============================================================
# NATURAL EARTH FEATURES
# ============================================================

LAND = cfeature.NaturalEarthFeature(
    "physical",
    "land",
    "50m",
    facecolor="#dbe8c7",
    edgecolor="none",
)

OCEAN = cfeature.NaturalEarthFeature(
    "physical",
    "ocean",
    "50m",
    facecolor="#cfe4f7",
    edgecolor="none",
)

BORDERS = cfeature.NaturalEarthFeature(
    "cultural",
    "admin_0_boundary_lines_land",
    "10m",
    facecolor="none",
    edgecolor="0.35",
)

COASTLINE = cfeature.NaturalEarthFeature(
    "physical",
    "coastline",
    "10m",
    facecolor="none",
    edgecolor="0.15",
)


# ============================================================
# STUDY LOCATIONS
# ============================================================

COUNTRY_LABELS = {
    "Saudi Arabia": (45.0, 23.4),
    "Yemen": (47.0, 15.7),
    "Oman": (56.5, 20.6),
    "UAE": (54.2, 23.7),
    "Qatar": (51.2, 26.0),
}

CITIES = {
    "Riyadh": (46.6753, 24.7136),
    "Jeddah": (39.1925, 21.4858),
    "Aden": (45.0187, 12.8855),
    "Muscat": (58.3829, 23.5880),
    "Doha": (51.5200, 25.2760),
    "Dubai": (55.2962, 25.2770),
    "Dammam": (50.0888, 26.4207),
    "Medina": (39.5692, 24.5247),
    "Kuwait City": (47.9774, 29.3759),
    "Basra": (47.7835, 30.5085),
}


# ============================================================
# HELPERS
# ============================================================

def load_etopo(path: Path) -> xr.DataArray:
    """
    Load and subset ETOPO 2022 to the Arabian Peninsula domain.

    Longitudes are converted to 0--360 coordinates for consistency
    with the plotting configuration.
    """
    with xr.open_dataset(path) as ds:

        variable = "z" if "z" in ds.data_vars else list(ds.data_vars)[0]

        ds = (
            ds.assign_coords(
                lon=((ds.lon + 360) % 360)
            )
            .sortby("lon")
        )

        if ds.lat[0] > ds.lat[-1]:
            ds = ds.sortby("lat")

        etopo = (
            ds[variable]
            .sel(
                lon=slice(35, 62),
                lat=slice(10, 35),
            )
            .load()
        )

    return etopo


def add_etopo(
    ax,
    extent,
    etopo: xr.DataArray,
    alpha: float = 0.65,
):
    """Plot ETOPO elevation and bathymetry on the AP panel."""
    sub = etopo.sel(
        lon=slice(extent[0], extent[1]),
        lat=slice(extent[2], extent[3]),
    )

    lon = sub.lon.values
    lat = sub.lat.values
    elevation = sub.values

    # Downsample sufficiently for fast vector-PDF creation.
    step = max(
        1,
        int(max(elevation.shape) / 450),
    )

    lon = lon[::step]
    lat = lat[::step]
    elevation = elevation[::step, ::step]

    norm = TwoSlopeNorm(
        vmin=-5000,
        vcenter=0,
        vmax=3000,
    )

    return ax.pcolormesh(
        lon,
        lat,
        elevation,
        cmap=cmocean.cm.topo,
        norm=norm,
        shading="auto",
        alpha=alpha,
        transform=GEO,
        zorder=0,
        rasterized=True,
    )


def add_base(
    ax,
    extent,
    title,
    etopo: xr.DataArray | None = None,
    extent_crs=GEO,
):
    """Add geographic base layers, gridlines, and panel title."""
    ax.set_extent(
        extent,
        crs=extent_crs,
    )

    if etopo is not None:
        mesh = add_etopo(
            ax,
            extent,
            etopo,
        )
    else:
        mesh = None
        ax.add_feature(
            OCEAN,
            zorder=0,
        )
        ax.add_feature(
            LAND,
            zorder=1,
        )

    ax.add_feature(
        BORDERS,
        linewidth=0.45,
        zorder=5,
    )

    ax.add_feature(
        COASTLINE,
        linewidth=0.7,
        zorder=6,
    )

    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.2,
        color="0.45",
        alpha=0.35,
        linestyle="--",
        zorder=7,
    )

    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 7}
    gridlines.ylabel_style = {"size": 7}

    ax.set_title(
        title,
        fontsize=11,
    )

    return mesh


def add_label(
    ax,
    text,
    lon,
    lat,
    size=7.5,
    weight="normal",
    ha="center",
    va="center",
    style="normal",
    rotation=0,
):
    """Add a lightly backed geographic label."""
    ax.text(
        lon,
        lat,
        text,
        fontsize=size,
        fontweight=weight,
        fontstyle=style,
        ha=ha,
        va=va,
        rotation=rotation,
        transform=GEO,
        zorder=10,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.55,
            "pad": 1.2,
        },
    )


def add_panel_label(ax, label: str) -> None:
    """Add manuscript panel label."""
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=100,
    )


# ============================================================
# FIGURE
# ============================================================

def make_figure(
    etopo_path: Path,
    output_path: Path,
) -> None:

    etopo = load_etopo(etopo_path)

    fig = plt.figure(
        figsize=(11, 6),
        constrained_layout=True,
    )

    grid = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.45, 1.0],
        height_ratios=[1, 1],
    )

    ax_ap = fig.add_subplot(
        grid[:, 0],
        projection=GEO,
    )

    ax_enso = fig.add_subplot(
        grid[0, 1],
        projection=PACIFIC_PROJ,
    )

    ax_iod = fig.add_subplot(
        grid[1, 1],
        projection=GEO,
    )

    ap_extent = [35, 62, 10, 35]
    enso_extent = [130, 290, -20, 20]
    iod_extent = [35, 120, -25, 25]

    mesh_ap = add_base(
        ax_ap,
        ap_extent,
        "Arabian Peninsula",
        etopo=etopo,
    )

    add_base(
        ax_enso,
        enso_extent,
        "ENSO Region",
        extent_crs=GEO,
    )

    add_base(
        ax_iod,
        iod_extent,
        "IOD Region",
    )

    # --------------------------------------------------------
    # ENSO regions
    # --------------------------------------------------------

    # Niño 3.4 is the region used for the RONI analysis.
    nino34 = Rectangle(
        (190, -5),
        50,
        10,
        linewidth=1.4,
        edgecolor="black",
        facecolor="none",
        transform=GEO,
        zorder=5,
    )

    ax_enso.add_patch(nino34)

    ax_enso.text(
        215,
        7,
        "Niño 3.4",
        ha="center",
        va="bottom",
        fontsize=9,
        transform=GEO,
        zorder=6,
    )

    # Additional canonical Niño regions are shown for context.
    enso_regions = {
        "Niño 1+2": {
            "xy": (270, -10),
            "width": 10,
            "height": 10,
            "label_xy": (275, -13),
        },
        "Niño 3": {
            "xy": (210, -5),
            "width": 60,
            "height": 10,
            "label_xy": (240, -8),
        },
        "Niño 4": {
            "xy": (160, -5),
            "width": 50,
            "height": 10,
            "label_xy": (185, -8),
        },
    }

    for name, spec in enso_regions.items():

        box = Rectangle(
            spec["xy"],
            spec["width"],
            spec["height"],
            linewidth=1.1,
            edgecolor="goldenrod",
            facecolor="none",
            transform=GEO,
            zorder=5,
        )

        ax_enso.add_patch(box)

        ax_enso.text(
            spec["label_xy"][0],
            spec["label_xy"][1],
            name,
            ha="center",
            va="top",
            fontsize=7.5,
            color="goldenrod",
            transform=GEO,
            zorder=6,
        )

    # --------------------------------------------------------
    # IOD regions
    # --------------------------------------------------------

    dmi_west = Rectangle(
        (50, -10),
        20,
        20,
        linewidth=1.3,
        edgecolor="black",
        facecolor="none",
        transform=GEO,
        zorder=5,
    )

    ax_iod.add_patch(dmi_west)

    ax_iod.text(
        60,
        -12.5,
        "Western pole",
        ha="center",
        va="top",
        fontsize=8,
        transform=GEO,
        zorder=6,
    )

    dmi_east = Rectangle(
        (90, -10),
        20,
        10,
        linewidth=1.3,
        edgecolor="black",
        facecolor="none",
        transform=GEO,
        zorder=5,
    )

    ax_iod.add_patch(dmi_east)

    ax_iod.text(
        100,
        -12.5,
        "Eastern pole",
        ha="center",
        va="top",
        fontsize=8,
        transform=GEO,
        zorder=6,
    )

    # --------------------------------------------------------
    # AP labels
    # --------------------------------------------------------

    for name, (lon, lat) in COUNTRY_LABELS.items():
        add_label(
            ax_ap,
            name,
            lon,
            lat,
            size=7.5,
            weight="bold",
        )

    for city, (lon, lat) in CITIES.items():

        ax_ap.scatter(
            lon,
            lat,
            s=34,
            color="black",
            edgecolor="white",
            linewidth=0.7,
            transform=GEO,
            zorder=20,
        )

        add_label(
            ax_ap,
            city,
            lon + 0.45,
            lat + 0.25,
            size=7.4,
            ha="left",
        )

    # Seas and gulfs
    add_label(
        ax_ap,
        "Red Sea",
        38.5,
        20.5,
        size=8,
        style="italic",
        rotation=-62,
    )

    add_label(
        ax_ap,
        "Gulf of Aden",
        47.5,
        12.0,
        size=8,
        style="italic",
    )

    add_label(
        ax_ap,
        "Gulf of Oman",
        58.5,
        24.8,
        size=8,
        style="italic",
    )

    add_label(
        ax_ap,
        "Persian \nGulf",
        50.2,
        28.0,
        size=8,
        style="italic",
    )

    add_label(
        ax_ap,
        "Arabian Sea",
        57.5,
        13.5,
        size=8,
        style="italic",
    )

    # Geographic features
    add_label(
        ax_ap,
        "Bab el-Mandeb",
        42.7,
        12.7,
        size=6.8,
    )

    add_label(
        ax_ap,
        "Strait of Hormuz",
        56.2,
        26.2,
        size=6.8,
    )

    add_label(
        ax_ap,
        "Arabian Desert",
        44.8,
        26.8,
        size=8,
    )

    add_label(
        ax_ap,
        "Rub' al Khali",
        50.8,
        20.0,
        size=8,
    )

    add_panel_label(
        ax_ap,
        "(a)",
    )

    add_panel_label(
        ax_enso,
        "(b)",
    )

    add_panel_label(
        ax_iod,
        "(c)",
    )

    colorbar = fig.colorbar(
        mesh_ap,
        ax=ax_ap,
        orientation="horizontal",
        fraction=0.045,
        pad=0.045,
    )

    colorbar.set_label(
        "Elevation / bathymetry (m)",
        fontsize=8,
    )

    colorbar.ax.tick_params(
        labelsize=7,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {output_path}")

    plt.show()


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Create the AP/ENSO/IOD manuscript reference map."
    )

    parser.add_argument(
        "--etopo",
        type=Path,
        default=(
            repo_root
            / "data"
            / "elevation"
            / "ETOPO_2022_v1_60s_N90W180_bed.nc"
        ),
        help="Path to the ETOPO 2022 NetCDF file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo_root
            / "figures"
            / "reference_map"
            / "reference_map_AP_ENSO_IOD.pdf"
        ),
        help="Output PDF path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    make_figure(
        etopo_path=args.etopo.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()


# In[ ]:




