import earthaccess
earthaccess.login()

for short_name in ("SENTINEL-1C_OCN", "SENTINEL-1D_OCN"):
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=("2025-05-01", "2026-10-02"),
        bounding_box=(-122.2, 36.5, -121.7, 37.1),
    )
    print(short_name, len(results))
    earthaccess.download(results, local_path="./data/wind2")