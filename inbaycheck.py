import xarray as xr
import glob, os

nc_dir = "./data/wind2_nc"

f = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))[0]
ds = xr.open_dataset(f)
print(ds)

in_bay = ((ds.owiLon > -122.2) & (ds.owiLon < -121.7) &
          (ds.owiLat > 36.5) & (ds.owiLat < 37.1))
ok = in_bay & (ds.owiMask == 0)
print("valid wind cells in bay:", int(ok.sum()))
print(ds.owiWindSpeed.where(ok).mean().item(), "m/s")