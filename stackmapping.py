import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

STACK_DIR = "data/datastacks"

# Which day to draw. A string like "2025-05-01", or None for the day with
# the best chlorophyll coverage.
DAY = None

# name -> (file, variable, colormap, log color scale, colorbar label)
PRODUCTS = {
    "Chlorophyll": ("stacked_chlorophyll.nc", "chlor_a", "viridis", True, "chlor_a [mg m$^{-3}$]"),
    "SST": ("stacked_sst.nc", "sst", "coolwarm", False, "SST [°C]"),
    "Wind": ("stacked_wind_s1.nc", "wind_speed", "plasma", False, "wind speed [m s$^{-1}$]"),
}


def open_stack(fname):
    ds = xr.open_dataset(os.path.join(STACK_DIR, fname))
    # In the stacked files the dimension is called "day" and "time" is a
    # separate variable on it. Swapping makes "time" the index, so we can
    # select a date instead of a position.
    return ds.swap_dims({"day": "time"})


def day_index(ds, day):
    """Position of a given date in this stack, or None if the date is absent."""
    dates = ds["time"].values.astype("datetime64[D]")
    hits = np.flatnonzero(dates == day)
    return int(hits[0]) if hits.size else None


stacks = {name: open_stack(spec[0]) for name, spec in PRODUCTS.items()}

# ---- choose the day ----
chl = stacks["Chlorophyll"]
if DAY is None:
    coverage = chl["chlor_a"].notnull().mean(dim=("lat", "lon")).values
    day = chl["time"].values[int(np.argmax(coverage))].astype("datetime64[D]")
else:
    day = np.datetime64(DAY, "D")
print("Plotting day:", day)

# ---- draw ----
lon = chl["lon"].values
lat = chl["lat"].values
extent = [lon.min(), lon.max(), lat.min(), lat.max()]

fig, axes = plt.subplots(1, len(PRODUCTS), figsize=(5.5 * len(PRODUCTS), 5.5),
                         subplot_kw={"projection": ccrs.PlateCarree()},
                         constrained_layout=True)

for ax, (name, (fname, var, cmap, use_log, label)) in zip(axes, PRODUCTS.items()):
    ds = stacks[name]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(f"{name}  {day}")

    k = day_index(ds, day)
    if k is None:
        ax.text(0.5, 0.5, "no data this day", ha="center", va="center", transform=ax.transAxes)
        continue

    field = ds[var].isel(time=k)
    if use_log:
        field = field.where(field > 0)
    finite = field.values[np.isfinite(field.values)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "all NaN", ha="center", va="center", transform=ax.transAxes)
        continue

    vmin, vmax = np.nanpercentile(finite, [5, 95])
    kwargs = {"norm": colors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)} if use_log \
        else {"vmin": vmin, "vmax": vmax}

    graph = ax.pcolormesh(ds["lon"], ds["lat"], field, transform=ccrs.PlateCarree(),
                          cmap=cmap, shading="nearest", **kwargs)

    # 10 m land: the default 110 m outline is so coarse that it would cover
    # real ocean pixels around Monterey Bay.
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.5, zorder=3)
    fig.colorbar(graph, ax=ax, label=label, shrink=0.8)

plt.savefig(os.path.join(STACK_DIR, "day_maps_cartopy.png"), dpi=150)
plt.show()