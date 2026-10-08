import xarray as xr
import earthaccess

auth = earthaccess.login()

#PRODUCT DETAILS FOR DATA COLLECTION
###################################################
# Wind product: ASCATC-L2-Coastal                 # #Trying SENTINEL-1A_OCN
# Sea Surface Temperature Product: MODISA_L2_SST  #
# Chlorophyll product: PACE_OCI_L2_BGC            #
###################################################
results = earthaccess.search_data(
    short_name='SENTINEL-1A_OCN',
    temporal=('2025-05-01','2026-09-12'),
    bounding_box=(-122.2, 36.5, -121.7, 37.1)
)

print(len(results))

files = earthaccess.download(results, local_path='./data/wind2')


