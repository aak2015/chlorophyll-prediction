import netCDF4 as nc

ds = nc.Dataset("data/wind/ascat_20250501_052700_metopc_33631_eps_o_coa_3301_ovw.l2.nc")

print("Top-level groups:", list(ds.groups.keys()))
print("Top-level variables:", list(ds.variables.keys()))

def walk(group, indent=0):
    for name, subgroup in group.groups.items():
        print(" " * indent + f"[group] {name}")
        print(" " * indent + f"  variables: {list(subgroup.variables.keys())}")
        walk(subgroup, indent + 4)

walk(ds)
ds.close()