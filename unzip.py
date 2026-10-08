import glob, os, zipfile

zip_dir = "./data/wind2"
nc_dir = "./data/wind2_nc"
os.makedirs(nc_dir, exist_ok=True)

for zp in sorted(glob.glob(os.path.join(zip_dir, "*.zip"))):
    with zipfile.ZipFile(zp) as z:
        for name in z.namelist():
            if "/measurement/" in name and name.endswith(".nc"):
                out = os.path.join(nc_dir, os.path.basename(name))
                if not os.path.exists(out):
                    with z.open(name) as src, open(out, "wb") as dst:
                        dst.write(src.read())

