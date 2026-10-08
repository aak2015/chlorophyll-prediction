import os
import warnings
from collections import defaultdict
from datetime import datetime, timezone, date

import netCDF4 as nc
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

OUTPUT_DIR = "./data/datastacks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# NEW: the single shared grid (Monterey Bay). Defined once, used by all jobs.
# ---------------------------------------------------------------------------
LAT_EDGES = np.linspace(36.4, 37.2, 41)       # 40 cells of 0.02 deg
LON_EDGES = np.linspace(-122.7, -121.7, 51)   # 50 cells of 0.02 deg
LAT_CENTERS = 0.5 * (LAT_EDGES[:-1] + LAT_EDGES[1:])
LON_CENTERS = 0.5 * (LON_EDGES[:-1] + LON_EDGES[1:])
LON_GRID, LAT_GRID = np.meshgrid(LON_CENTERS, LAT_CENTERS)   # both (40, 50)
LAT_MIN, LAT_MAX = LAT_EDGES[0], LAT_EDGES[-1]
LON_MIN, LON_MAX = LON_EDGES[0], LON_EDGES[-1]

JOBS = [
    {
        "input_dir": "./data/chlorophyll",
        "output_file": os.path.join(OUTPUT_DIR, "stacked_chlorophyll.nc"),
        "primary_variable": "chlor_a",
        "units": "mg m^-3",
        "time_format": "pace",
        "layout": "grouped",
        "data_group": "geophysical_data",
        "nav_group": "navigation_data",
        "lat_name": "latitude",
        "lon_name": "longitude",
        # NEW keys
        "log": True,                      # interpolate/average log10(chlor_a)
        "flag_var": "l2_flags",
        "bad_flags": ["ATMFAIL", "LAND", "HIGLINT", "HILT", "HISATZEN",
                      "STRAYLIGHT", "CLDICE", "NAVFAIL"],  # verify names in your files
        "qual_var": None,
        "qual_max": None,
        "margin_deg": 0.2,                # extra source data kept around the box
        "max_gap_deg": 0.02,              # blank cell if nearest valid pixel is farther
    },
    {
        "input_dir": "./data/sea-surface-temp",
        "output_file": os.path.join(OUTPUT_DIR, "stacked_sst.nc"),
        "primary_variable": "sst",
        "units": "degree_C",
        "time_format": "pace",
        "layout": "grouped",
        "data_group": "geophysical_data",
        "nav_group": "navigation_data",
        "lat_name": "latitude",
        "lon_name": "longitude",
        "log": False,
        "flag_var": "l2_flags",
        "bad_flags": ["LAND", "NAVFAIL"],  # verify names in your files
        "qual_var": "qual_sst",            # 0 best ... 3 bad (verify in your files)
        "qual_max": 1,                     # keep best and good; relax to 2 if too sparse
        "margin_deg": 0.2,
        "max_gap_deg": 0.02,
    },
    {
        "input_dir": "./data/wind",
        "output_file": os.path.join(OUTPUT_DIR, "stacked_wind.nc"),
        "primary_variable": "wind_speed",
        "units": "m s^-1",
        "time_format": "ascat",
        "layout": "flat",
        "lat_name": "lat",
        "lon_name": "lon",
        "log": False,
        "flag_var": None,                  # set to "wvc_quality_flag" once you know its bits
        "bad_flags": [],
        "qual_var": None,
        "qual_max": None,
        "margin_deg": 0.3,
        "max_gap_deg": 0.15,               # ~12.5 km product; ~0.3 for the 25 km product
    },
]


# ---------------------------------------------------------------------------
# UNCHANGED from your script
# ---------------------------------------------------------------------------
def parse_time_pace(filename):
    time_str = filename.split(".")[1]
    return datetime.strptime(time_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def parse_time_ascat(filename):
    parts = filename.split("_")
    date_str, time_str = parts[1], parts[2]
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


TIME_PARSERS = {
    "pace": parse_time_pace,
    "ascat": parse_time_ascat,
}


def get_pool(src, job):
    if job["layout"] == "grouped":
        return src.groups[job["data_group"]].variables
    else:
        return src.variables


def get_nav(src, job):
    if job["layout"] == "grouped":
        nav = src.groups[job["nav_group"]]
        return nav.variables[job["lat_name"]], nav.variables[job["lon_name"]]
    else:
        return src.variables[job["lat_name"]], src.variables[job["lon_name"]]


# ---------------------------------------------------------------------------
# NEW: quality masks (replace discover_variables)
# ---------------------------------------------------------------------------
def bad_flag_mask(pool, job):
    """True where a bit-flag variable (e.g. l2_flags) marks a pixel unusable."""
    var = pool[job["flag_var"]]
    if not (hasattr(var, "flag_masks") and hasattr(var, "flag_meanings")):
        print(f"WARNING: {job['flag_var']} has no flag_masks/flag_meanings; not masking.")
        return np.zeros(var.shape, dtype=bool)

    masks = np.atleast_1d(var.flag_masks)
    meanings = var.flag_meanings.split()
    wanted = {n.upper() for n in job["bad_flags"]}

    missing = wanted - {n.upper() for n in meanings}
    if missing:
        print(f"WARNING: flag names not in file: {sorted(missing)}")

    bad_bits = 0
    for m, name in zip(masks, meanings):
        if name.upper() in wanted:
            bad_bits |= int(m)

    flags = np.ma.filled(var[:], 0).astype(np.int64)
    return (flags & bad_bits) != 0


# ---------------------------------------------------------------------------
# NEW: read one granule as flat arrays of valid pixels near the box
# (replaces the reading half of write_granule_group)
# ---------------------------------------------------------------------------
def read_swath(src, job):
    pool = get_pool(src, job)
    lat_var, lon_var = get_nav(src, job)

    values = np.ma.filled(pool[job["primary_variable"]][:].astype("f8"), np.nan)
    lat = np.ma.filled(lat_var[:].astype("f8"), np.nan)
    lon = np.ma.filled(lon_var[:].astype("f8"), np.nan)

    # If coordinates are 1D axes, expand to 2D to match the data
    if lat.ndim == 1 and values.ndim == 2:
        lon, lat = np.meshgrid(lon, lat)

    # Force longitude to -180..180 (ASCAT may be 0..360)
    lon = ((lon + 180.0) % 360.0) - 180.0

    # Quality masking
    if job["flag_var"]:
        values[bad_flag_mask(pool, job)] = np.nan
    if job["qual_var"]:
        qual = np.ma.filled(pool[job["qual_var"]][:].astype("f8"), np.inf)
        values[qual > job["qual_max"]] = np.nan

    # Log transform for chlorophyll
    if job["log"]:
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(values > 0, np.log10(values), np.nan)

    # Keep finite pixels inside the box plus a margin
    m = job["margin_deg"]
    keep = (
            np.isfinite(values) & np.isfinite(lat) & np.isfinite(lon)
            & (lat >= LAT_MIN - m) & (lat <= LAT_MAX + m)
            & (lon >= LON_MIN - m) & (lon <= LON_MAX + m)
    )
    return lon[keep], lat[keep], values[keep]


# ---------------------------------------------------------------------------
# NEW: the SciPy interpolation itself
# ---------------------------------------------------------------------------
def regrid_interp(lon, lat, vals, job):
    empty = np.full(LAT_GRID.shape, np.nan)
    if vals.size < 3:
        return empty

    pts = np.column_stack([lon, lat])
    targets = np.column_stack([LON_GRID.ravel(), LAT_GRID.ravel()])

    try:
        out = LinearNDInterpolator(pts, vals)(targets)
    except Exception as e:   # e.g. collinear points -> Qhull error
        print(f"  interpolation failed: {e}")
        return empty

    # Blank cells that are far from any real pixel (clouds, gaps)
    dist, _ = cKDTree(pts).query(targets)
    out[dist > job["max_gap_deg"]] = np.nan
    return out.reshape(LAT_GRID.shape)


# ---------------------------------------------------------------------------
# CHANGED: run_job now regrids each granule, groups by day, writes one stack
# ---------------------------------------------------------------------------
def run_job(job):
    input_dir = job["input_dir"]
    output_file = job["output_file"]

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".nc"))
    if not files:
        print(f"No files found in {input_dir}.")
        return

    print(f"--- Starting stacking for '{input_dir}' ({len(files)} files) ---")

    daily = defaultdict(list)    # UTC date -> list of (40, 50) arrays
    skipped = 0

    for i, filename in enumerate(files, start=1):
        src = None
        try:
            src = nc.Dataset(os.path.join(input_dir, filename), "r")
            granule_time = TIME_PARSERS[job["time_format"]](filename)

            lon, lat, vals = read_swath(src, job)
            if vals.size < 3:
                skipped += 1
                continue

            grid = regrid_interp(lon, lat, vals, job)
            if np.all(np.isnan(grid)):
                skipped += 1
                continue

            daily[granule_time.date()].append(grid)
        except Exception as e:
            print(f"Error at file: {filename}: {e}")
        finally:
            if src is not None:
                src.close()

        if i % 25 == 0 or i == len(files):
            print(f"Progress: [{i}/{len(files)}] files processed.")

    days = sorted(daily)
    print(f"{len(days)} days with data; {skipped} granules had no usable pixels in the box.")

    if os.path.exists(output_file):
        os.remove(output_file)

    out = nc.Dataset(output_file, "w", format="NETCDF4")
    out.createDimension("day", None)
    out.createDimension("lat", len(LAT_CENTERS))
    out.createDimension("lon", len(LON_CENTERS))

    v_lat = out.createVariable("lat", "f8", ("lat",))
    v_lat.units = "degrees_north"
    v_lat[:] = LAT_CENTERS
    v_lon = out.createVariable("lon", "f8", ("lon",))
    v_lon.units = "degrees_east"
    v_lon[:] = LON_CENTERS

    v_time = out.createVariable("time", "i4", ("day",))
    v_time.units = "days since 1970-01-01 00:00:00"
    v_time.calendar = "standard"

    v_data = out.createVariable(job["primary_variable"], "f4",
                                ("day", "lat", "lon"), fill_value=np.nan)
    v_data.units = job["units"]
    v_n = out.createVariable("n_obs", "i2", ("day", "lat", "lon"))
    v_n.long_name = "number of granules contributing to each cell-day"

    out.setncattr("created_by", "datacombination.py shared-grid regridder")
    out.setncattr("regrid_method", "scipy LinearNDInterpolator + KD-tree distance mask")
    out.setncattr("daily_combine", "nanmean" + (" in log10 space" if job["log"] else ""))

    for k, day in enumerate(days):
        stack = np.stack(daily[day])                       # (n_granules, 40, 50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN cells
            mean = np.nanmean(stack, axis=0)
        if job["log"]:
            mean = 10.0 ** mean                            # back to mg m^-3
        v_time[k] = (day - date(1970, 1, 1)).days
        v_data[k, :, :] = mean
        v_n[k, :, :] = np.sum(~np.isnan(stack), axis=0)

    out.close()
    print(f"Wrote {output_file}: {len(days)} days on a "
          f"{len(LAT_CENTERS)}x{len(LON_CENTERS)} grid.\n")


for job in JOBS:
    run_job(job)
    print(f"--- {job['input_dir']} completed ---")

print("All jobs finished.")

# Proof that all three files share one grid
grids = []
for job in JOBS:
    if os.path.exists(job["output_file"]):
        with nc.Dataset(job["output_file"]) as ds:
            grids.append((ds["lat"][:].copy(), ds["lon"][:].copy()))
same = all(np.array_equal(grids[0][0], g[0]) and np.array_equal(grids[0][1], g[1])
           for g in grids[1:])
print("Identical lat/lon across all output files:", same)