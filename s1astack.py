"""
Regrid Sentinel-1 Level-2 OCN wind (OWI) granules onto the shared Monterey Bay
grid and stack them into one file with dimensions (day, lat, lon).

Input : extracted measurement .nc files (one per granule) in INPUT_DIR
Output: OUTPUT_FILE with wind_speed, wind_u, wind_v, n_obs on the same
        40 x 50 grid used by the chlorophyll and SST stacks.
"""
import os
import re
import warnings
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import netCDF4 as nc
import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
INPUT_DIR = "./data/wind2_nc"
OUTPUT_DIR = "./data/datastacks"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "stacked_wind_s1.nc")

# Shared grid. Must stay identical to the chlorophyll / SST script.
LAT_EDGES = np.linspace(36.4, 37.2, 41)       # 40 cells of 0.02 deg
LON_EDGES = np.linspace(-122.7, -121.7, 51)   # 50 cells of 0.02 deg
LAT_CENTERS = 0.5 * (LAT_EDGES[:-1] + LAT_EDGES[1:])
LON_CENTERS = 0.5 * (LON_EDGES[:-1] + LON_EDGES[1:])
NY, NX = len(LAT_CENTERS), len(LON_CENTERS)

# Hours added to the UTC granule time before taking the date. -8 puts each
# pass on its local (Pacific) day, so the ~02 UTC evening pass lands on the
# same day as that afternoon's chlorophyll / SST. Use 0 for plain UTC dates.
DAY_OFFSET_HOURS = -8

# "both", "morning" (local hour < 12) or "evening" (local hour >= 12)
PASSES = "both"

# Drop pixels whose owiWindQuality exceeds this (0 is best). None = no filter.
MAX_WIND_QUALITY = None

MIN_PIXELS = 1      # source pixels required in a cell to keep it
MAX_SPEED = 60.0    # m/s; anything above is treated as bad data

REQUIRED = ("owiLat", "owiLon", "owiWindSpeed", "owiWindDirection")
TIME_RE = re.compile(r"(\d{8})t(\d{6})", re.IGNORECASE)
_warned = set()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def parse_start_time(filename):
    """UTC start time from e.g. s1a-iw-ocn-vv-20251209t020739-...-001.nc"""
    m = TIME_RE.search(filename)
    if m is None:
        raise ValueError("no timestamp found in filename")
    return datetime.strptime(m.group(1) + m.group(2),
                             "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def as_float(var):
    return np.ma.filled(var[:].astype("f8"), np.nan)


def warn_once(key, message):
    if key not in _warned:
        _warned.add(key)
        print(f"WARNING: {message}")


def read_granule(path):
    """Return lon, lat, speed, direction for valid ocean pixels inside the grid."""
    with nc.Dataset(path) as src:
        v = src.variables
        missing = [n for n in REQUIRED if n not in v]
        if missing:
            raise KeyError(f"missing variables {missing}")

        lat = as_float(v["owiLat"])
        lon = as_float(v["owiLon"])
        speed = as_float(v["owiWindSpeed"])
        wdir = as_float(v["owiWindDirection"])

        with np.errstate(invalid="ignore"):
            good = (np.isfinite(lat) & np.isfinite(lon)
                    & (speed >= 0) & (speed <= MAX_SPEED)
                    & (wdir >= 0) & (wdir <= 360))

            if "owiMask" in v:
                good &= as_float(v["owiMask"]) == 0        # 0 = valid ocean
            else:
                warn_once("mask", "owiMask not found; no land mask applied.")

            if MAX_WIND_QUALITY is not None:
                if "owiWindQuality" in v:
                    good &= as_float(v["owiWindQuality"]) <= MAX_WIND_QUALITY
                else:
                    warn_once("qual", "owiWindQuality not found; no quality filter.")

    lon = ((lon + 180.0) % 360.0) - 180.0
    with np.errstate(invalid="ignore"):
        good &= ((lat >= LAT_EDGES[0]) & (lat < LAT_EDGES[-1])
                 & (lon >= LON_EDGES[0]) & (lon < LON_EDGES[-1]))

    return lon[good], lat[good], speed[good], wdir[good]


# ---------------------------------------------------------------------------
# Regridding: average every source pixel that falls inside each grid cell
# ---------------------------------------------------------------------------
def bin_to_grid(lon, lat, fields):
    iy = np.clip(np.searchsorted(LAT_EDGES, lat, side="right") - 1, 0, NY - 1)
    ix = np.clip(np.searchsorted(LON_EDGES, lon, side="right") - 1, 0, NX - 1)
    cell = iy * NX + ix

    count = np.bincount(cell, minlength=NY * NX)
    enough = count >= max(MIN_PIXELS, 1)

    grids = []
    for f in fields:
        total = np.bincount(cell, weights=f, minlength=NY * NX)
        mean = np.full(NY * NX, np.nan)
        mean[enough] = total[enough] / count[enough]
        grids.append(mean.reshape(NY, NX))
    return grids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".nc"))
    if not files:
        print(f"No .nc files found in {INPUT_DIR}.")
        return
    print(f"--- Stacking Sentinel-1 wind: {len(files)} files ---")

    daily = defaultdict(lambda: {"speed": [], "u": [], "v": [], "t": []})
    empty = errors = filtered = 0

    for i, filename in enumerate(files, start=1):
        try:
            t = parse_start_time(filename)
            local = t + timedelta(hours=DAY_OFFSET_HOURS)
            is_morning = local.hour < 12
            if (PASSES == "morning" and not is_morning) or \
                    (PASSES == "evening" and is_morning):
                filtered += 1
                continue

            lon, lat, speed, wdir = read_granule(os.path.join(INPUT_DIR, filename))
            if speed.size == 0:
                empty += 1
                continue

            # Direction is where the wind blows FROM, clockwise from north
            rad = np.deg2rad(wdir)
            u = -speed * np.sin(rad)     # eastward component
            v = -speed * np.cos(rad)     # northward component

            g_speed, g_u, g_v = bin_to_grid(lon, lat, [speed, u, v])
            if np.all(np.isnan(g_speed)):
                empty += 1
                continue

            d = daily[local.date()]
            d["speed"].append(g_speed)
            d["u"].append(g_u)
            d["v"].append(g_v)
            d["t"].append(t.timestamp())
        except Exception as e:
            errors += 1
            print(f"Error at file {filename}: {e}")

        if i % 25 == 0 or i == len(files):
            print(f"Progress: [{i}/{len(files)}] files processed.")

    days = sorted(daily)
    print(f"{len(days)} days with data; {empty} granules empty in the box; "
          f"{filtered} skipped by PASSES; {errors} errors.")
    if not days:
        print("Nothing to write.")
        return

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    out = nc.Dataset(OUTPUT_FILE, "w", format="NETCDF4")
    out.createDimension("day", None)
    out.createDimension("lat", NY)
    out.createDimension("lon", NX)

    v_lat = out.createVariable("lat", "f8", ("lat",))
    v_lat.units = "degrees_north"
    v_lat[:] = LAT_CENTERS
    v_lon = out.createVariable("lon", "f8", ("lon",))
    v_lon.units = "degrees_east"
    v_lon[:] = LON_CENTERS

    v_time = out.createVariable("time", "i4", ("day",))
    v_time.units = "days since 1970-01-01 00:00:00"
    v_time.calendar = "standard"
    v_time.long_name = f"day after shifting UTC by {DAY_OFFSET_HOURS} h"

    v_obs = out.createVariable("obs_time", "f8", ("day",))
    v_obs.units = "seconds since 1970-01-01 00:00:00"
    v_obs.long_name = "mean UTC start time of the granules used for this day"

    dims = ("day", "lat", "lon")
    v_speed = out.createVariable("wind_speed", "f4", dims, fill_value=np.nan)
    v_speed.units = "m s^-1"
    v_speed.long_name = "10 m wind speed (mean of scalar speed)"
    v_u = out.createVariable("wind_u", "f4", dims, fill_value=np.nan)
    v_u.units = "m s^-1"
    v_u.long_name = "eastward 10 m wind component"
    v_v = out.createVariable("wind_v", "f4", dims, fill_value=np.nan)
    v_v.units = "m s^-1"
    v_v.long_name = "northward 10 m wind component"
    v_n = out.createVariable("n_obs", "i2", dims)
    v_n.long_name = "number of granules contributing to each cell-day"

    out.setncattr("created_by", "stack_s1_wind.py")
    out.setncattr("source", "Sentinel-1 Level-2 OCN, OWI component")
    out.setncattr("regrid_method", "mean of source pixels inside each grid cell")
    out.setncattr("passes", PASSES)
    out.setncattr("day_offset_hours", DAY_OFFSET_HOURS)

    filled = []
    for k, day in enumerate(days):
        d = daily[day]
        speed = np.stack(d["speed"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN cells
            m_speed = np.nanmean(speed, axis=0)
            m_u = np.nanmean(np.stack(d["u"]), axis=0)
            m_v = np.nanmean(np.stack(d["v"]), axis=0)

        v_time[k] = (day - date(1970, 1, 1)).days
        v_obs[k] = float(np.mean(d["t"]))
        v_speed[k, :, :] = m_speed
        v_u[k, :, :] = m_u
        v_v[k, :, :] = m_v
        v_n[k, :, :] = np.sum(~np.isnan(speed), axis=0)
        filled.append(int(np.isfinite(m_speed).sum()))

    out.close()
    print(f"Wrote {OUTPUT_FILE}: {len(days)} days ({days[0]} to {days[-1]}) "
          f"on a {NY}x{NX} grid.")
    print(f"Cells with wind per day: median {int(np.median(filled))} "
          f"of {NY * NX} (min {min(filled)}, max {max(filled)}).")

    # Confirm the grid matches the other stacks, if they exist
    for other in ("stacked_chlorophyll.nc", "stacked_sst.nc"):
        path = os.path.join(OUTPUT_DIR, other)
        if os.path.exists(path):
            with nc.Dataset(path) as ds:
                same = (np.array_equal(np.asarray(ds["lat"][:]), LAT_CENTERS)
                        and np.array_equal(np.asarray(ds["lon"][:]), LON_CENTERS))
            print(f"Grid identical to {other}: {same}")


if __name__ == "__main__":
    main()