
import os
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from func_niger import *
import numpy as np


'''
Read RTI elevation files and convert it to GMS dataset

Notes: 
    - Specify a new output folder if needed (odir)
    - Change dataset_name (top_m or bot_m)
To run the script: 
    - Go to the scripts folder
    - Type: python convert_RTI_ele_to_GMS_dataset.py
    - Make sure you place the input/output folders, 
      at the same level with scripts folder

'''

# cd c:\Users\hpham\Documents\P32_Niger_2019\NWbud\scripts\

# Define some input parameters
grid_size = 2000*2000  # m2
fsize = 12
nrows, ncols, nlays = 537, 669,  4
ncells = nrows*ncols*nlays

dataset_name = 'bot_m'  # 'top_m' or 'bot_m'

opt_plot_ele = True
# Define some plot options
cmap = plt.get_cmap('jet')  # RdYlBu, gist_rainbow, bwr, jet, BuGn_r,
levels = np.linspace(-3000, 1000, 21)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
fsize = 12

# get name of geologic units
# Old data from Dennis
# geo_units_th, geo_units_ele, geo_units_real_name = get_geo_units()  # old data from Dennis
# New ISOPACH from AG, March 2020
geo_units_th, geo_units_ele, geo_units_real_name = get_geo_units_new()

# Open RTI elevation files


# Read input data =============================================================
#idir = r'../input/RTI_thickness_v3_April2020/'

idir = r'../input/RTI_thickness_v2_Mar2020/'

arr_dim = (nrows, ncols, nlays+1)
# x, y, zall = read_data(idir, geo_units_th, arr_dim)  # Old data Dennis

# Creat a new folder to save figures
odir = '../output/top_ele_15KM_resampled2KM/'
gen_outdir(odir)

# NEW data from AG, format exported from arcmap table
scale_unit = 1e6  # Convert (UTM_m) to (UTM_m x 1e6)
x, y, zall_tmp = read_data2(idir, geo_units_ele, arr_dim, scale_unit)
#zall[zall < 0.1] = np.nan
fig_id = ['a', 'b', 'c', 'd']

# Get elevation
z0 = zall_tmp[:, :, 0]
th = zall_tmp[:, :, 1:]
th[th < 0] = 0
zall_ele = zall_tmp.copy()

z1 = z0 - th[:, :, 0]
z2 = z1 - th[:, :, 1]
z3 = z2 - th[:, :, 2]
z4 = z3 - th[:, :, 3]

zall_ele[:, :, 0] = z0
zall_ele[:, :, 1] = z1
zall_ele[:, :, 2] = z2
zall_ele[:, :, 3] = z3
zall_ele[:, :, 4] = z4

'''
for i in range(1,6,1):
    z_tmp = zall_ele[:, :, i]
'''

'''
#zall_ele[:, :, 1] = th[:, :, 1] - z0
for i in range(1, 1, nlays+1):
    print(f'i={i}')
    zall_ele[:, :, i] = zall_ele[:, :, i-1] - th[:, :, i]
    print(f'{zall_ele[250, 250, i]}')
'''

if dataset_name == 'top_m':
    geo_units = geo_units_ele.copy()
    geo_units.remove('ISOPACH_Ci')
    zall = zall_ele[:, :, :-1]
elif dataset_name == 'bot_m':
    geo_units = geo_units_ele.copy()
    geo_units.remove('Land_surface')
    zall = zall_ele[:, :, 1:]

#
ofile = '../output/dataset_' + dataset_name + '.dat'
fid = open(ofile, 'w')
fid.write('DATASET\n')
fid.write('OBJTYPE "grid3d"\n')
fid.write('BEGSCL\n')
fid.write(f'ND {ncells}\n')
fid.write(f'NC {ncells}\n')
fid.write(f'NAME "{dataset_name}"\n')
fid.write('TS 1 0\n')

#
#cell_flag = np.empty(shape=(ncells, 1))
ele = np.empty(shape=(ncells, 1))
for i, geon in enumerate(geo_units):
    z = zall[:, :, i]
    # z = np.flipud(z)
    z2 = np.reshape(np.flipud(z), [nrows*ncols, 1])  # flipup for GMS dataset
    #print(f'k={i}, z={z[389-1, 736-1]}, {geon}')

    if opt_plot_ele:

        zmin, zmax = round(np.nanmin(z), -2), round(np.nanmax(z), -2)
        print(f'i={i}, zmin={zmin}, zmax={zmax}')
        levels = np.linspace(-300, 1100, 15)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4.6),
                                sharey=False)
        grid_plot(geon, i, x, y, z, cmap, norm,
                  levels, fsize, odir, fig, axs)
        # Save file
        ofile = odir + dataset_name + ' ' + geon + '.png'
        fig.savefig(ofile, dpi=150, transparent=False, bbox_inches='tight')
        print(f'Saved {ofile}')
        #
    count_start = i*nrows*ncols
    count_stop = (i+1)*nrows*ncols
    #print(f'{count_start}, {count_stop}\n')
    ele[count_start:count_stop] = z2
    #cell_flag[count_start:count_stop, 1] = z2

    # z = np.flipud(z)

    #grid_plot(geo_name, i, x, y, z, cmap, norm, levels, fsize, odir)

    # fid.write('\n')
# Assign activate/inactivate cells
cell_flag = np.copy(ele)
cell_flag[cell_flag != -99999] = 1
cell_flag[cell_flag == -99999] = 0


#
np.savetxt(fid, cell_flag, fmt='%d')
ele[ele == -99999] = -99
#ele[ele < 0] = 0
np.savetxt(fid, ele, fmt='%7.3f')

fid.write('ENDDS\n')
fid.close()


# Save ibound
dataset_name = 'ibound'
ofile = '../output/dataset_' + dataset_name + '.dat'
fid = open(ofile, 'w')
fid.write('DATASET\n')
fid.write('OBJTYPE "grid3d"\n')
fid.write('BEGSCL\n')
fid.write(f'ND {ncells}\n')
fid.write(f'NC {ncells}\n')
fid.write(f'NAME "{dataset_name}"\n')
fid.write('TS 1 0\n')
np.savetxt(fid, cell_flag, fmt='%d')
np.savetxt(fid, cell_flag, fmt='%d')
fid.write('ENDDS\n')
fid.close()

print(f'Saved output file at {ofile}\n')
