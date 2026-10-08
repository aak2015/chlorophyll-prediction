import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

path = "data/wind/ascat_20250501_052700_metopc_33631_eps_o_coa_3301_ovw.l2.nc"

chlor_data = xr.open_dataset(path, group='geophysical_data')
nav_data = xr.open_dataset(path, group='navigation_data')

dataset = chlor_data.assign_coords(
    latitude=(('number_of_lines','pixels_per_line'), nav_data['latitude'].data),
    longitude=(('number_of_lines', 'pixels_per_line'), nav_data['longitude'].data)
)

chlorophyll = dataset['chlor_a'].where(dataset['chlor_a'] > 0)

vmin, vmax = np.nanpercentile(chlorophyll, [5,95])

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_extent([-123.0,-121.5,36.0,37.5], crs=ccrs.PlateCarree())

graph = ax.pcolormesh(dataset['longitude'], dataset['latitude'], chlorophyll, transform=ccrs.PlateCarree(), norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap='viridis', shading='auto')

ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)

plt.colorbar(graph, ax=ax, label='chlor_a [mg m$^{-3}$')

plt.title("Chlorophyll Data in Monterey Bay")

plt.show()