import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from func_niger import *

# import matplotlib.patches as patches

# cd c:\Users\hpham\Documents\P32_Niger_2019\NWbud\scripts\

plot_top_ele = False  # one fig one unit

# Thickness
read_thickness = False  # Main
plot_thickness = False  # sub
# Estimate_volume = False  # sub
# get_vol_dri_domain = False

# Plot elevations
plot_top_ele_in_one_fig = False  # Subplots: all units in one fig

# Plot DIFF = top_ele - bot_ele of one unit
plot_th_DIFF = False

# Plot a single unit
plot_th_one_unit = False

# get name of geologic units
geo_units_th, geo_units_ele, geo_units_real_name = get_geo_units()
# Create some output directory
odir = '../output/check_top_ele'
gen_outdir(odir)

# Define some input parameters
grid_size = 2000*2000  # m2
fsize = 12
nrows, ncols, nlays = 673, 771, 6
cmap = plt.get_cmap('jet')  # RdYlBu, gist_rainbow, bwr, jet, BuGn_r,
levels = [0, 25, 50, 100, 150, 200, 250, 500, 750, 1000, 1500]
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

if read_thickness:
    # Read input data
    idir = r'/Users/hpham/Documents/P32_Niger_2019/04_GMS/thickness_xyz/'
    arr_dim = (nrows, ncols, nlays)
    x, y, zall = read_data(idir, geo_units_th, arr_dim)
    zall[zall < 0.1] = np.nan

    fig_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    # Calculate volume
    df_vol = pd.DataFrame(columns=['Name', 'Vol_RTI', 'Vol_DRI'])
    vol = []
    vol_dri = []
    for i in range(len(geo_units_th)):
        z = zall[:, :, i]
        # if Estimate_volume:
        volume = np.nansum(z*grid_size/1e9)  # m3/1e9 -> km3
        vol.append(volume)

        # if get_vol_dri_domain:
        start_row = 272
        end_row = 572
        start_column = 184
        end_column = 559
        z_dri = z[start_row:end_row, start_column:end_column]
        volume_dri = np.nansum(z_dri*grid_size/1e9)  # m3/1e9 -> km3
        vol_dri.append(volume_dri)
        # print('Blanked cell thickness less than 0.1 m \n')
        #
        #
    df_vol['Name'] = geo_units_th
    df_vol['Vol_RTI'] = vol
    df_vol['Vol_DRI'] = vol_dri
    df_vol.to_csv('volume.csv', index=False)

    # Plot a subplot for all units
    if plot_thickness:
        #
        geo_units_real_name_th = geo_units_real_name.copy()
        geo_units_real_name_th.remove('Land surface')
        geo_units_real_name_th.remove('Carbonifere and Basement')

        #
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(7, 10),
                                sharey=False)
        for i, ax in enumerate(fig.axes):
            z = zall[:, :, i]
            zmin = round(np.nanmin(z), -2)
            zmax = round(np.nanmax(z), -2)
            print(f'i={i}, zmin={zmin}, zmax={zmax}\n')

            stitle = '(' + fig_id[i] + ') ' + geo_units_real_name_th[i]
            if (i == 0 or i == 1 or i == 2 or i == 4):
                levels = np.linspace(0, 500, 11)
            elif i == 3:
                levels = np.linspace(0, 5000, 11)
            elif i == 5:
                levels = np.linspace(0, 2000, 9)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            grid_plot(stitle, i, x, y, z, cmap,
                      norm, levels, fsize, odir, fig, ax)
        #
        # Save file
        ofile = odir + '/all_thickness.png'
        fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')
        print(f'Saved {ofile}')


if plot_top_ele:
    # define some parameters
    idir = r'c:/Users/hpham/Documents/P32_Niger_2019/04_GMS/layer_ele/'
    '''
    geo_units_ele = ['MNT_Skua_2m_grd_zm',
                 'Cont_Term_Map_XS_2m_grd_zm',
                 'Paleo_Map_XS_2m_grd_zm',
                 'Cret_Sup_Map_XS_2m_grd_zm',
                 'Cont_Ham_Map_XS_2m_grd_zm',
                 'Cont_Int_Map_XS_2m_grd_zm',
                 'Juras_Map_XS_2m_grd_zm',
                 'Trias_Map_XS_2m_grd_zm',
                 'Socle_Skua_Map_XS_2m_grd_zm']
    '''
    fsize = 12

    cmap = plt.get_cmap('jet')  # RdYlBu, gist_rainbow, bwr, jet, BuGn_r,
    #levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 60, 80, 100, 200, 300]  #
    levels = np.linspace(-3000, 1000, 21)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    odir = 'top_ele'
    if not os.path.exists(odir):  # Make a new directory if not exist
        os.makedirs(odir)
        print(f'\nCreated directory {odir}\n')

    for i, geon in enumerate(geo_units_ele):
        geo_name = geon
        ifile = idir + geo_name + '.txt'
        data = np.loadtxt(ifile, delimiter=',')

        # Plot ========================================================================
        x = np.unique(data[:, 0])
    #    x = np.sort(x)
        y = np.unique(data[:, 1])
    #    y = np.sort(y)
        nrow = len(y)
        ncol = len(x)

        z = np.reshape(data[:, 2], [ncol, nrow])
        z = np.transpose(z)
        # z = np.flipud(z)
        z[z == -99999] = np.nan
        grid_plot(geo_name, i, x, y, z, cmap, norm, levels, fsize, odir)

if plot_top_ele_in_one_fig:
    # define some parameters
    idir = r'c:/Users/hpham/Documents/P32_Niger_2019/04_GMS/layer_ele/'
    fig_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    cmap = plt.get_cmap('jet')  # RdYlBu, gist_rainbow, bwr, jet, BuGn_r,
    arr_dim = (nrows, ncols, nlays+2)
    x, y, zall = read_data(idir, geo_units_ele, arr_dim)
    # zall[zall == -99999] = np.nan
    #
    # Plot a subplot for all units
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9.5, 10),
                            sharey=False)
    for i, ax in enumerate(fig.axes):
        if i <= nlays-1:  # Only plot 8 figures
            z = zall[:, :, i]
            zmin = round(np.nanmin(z), -2)
            zmax = round(np.nanmax(z), -2)
            print(f'i={i}, zmin={zmin}, zmax={zmax}\n')
            if i <= 5:
                levels = np.linspace(-300, 1000, 14)
            else:
                levels = np.linspace(-4000, 1000, 11)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            stitle = '(' + fig_id[i] + ') ' + geo_units_real_name[i]
            grid_plot(stitle, i, x, y, z, cmap, norm,
                      levels, fsize, odir, fig, ax)

    # Save file
    ofile = odir + '/all_elevations.png'
    fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')
    print(f'Saved {ofile}')


if plot_th_DIFF:
    # define some parameters
    idir = r'c:/Users/hpham/Documents/P32_Niger_2019/04_GMS/layer_ele/'
    fig_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    cmap = plt.get_cmap('jet')  # RdYlBu, gist_rainbow, bwr, jet, BuGn_r,
    arr_dim = (nrows, ncols, nlays+1)
    #
    geo_units_ele = geo_units_ele.copy()
    geo_units_ele.remove('MNT_Skua_2m_grd_zm')
    #
    x, y, zall = read_data(idir, geo_units_ele, arr_dim)
    # zall[zall == -99999] = np.nan
    #
    # Plot a subplot for all units
    geo_units_real_name_th = geo_units_real_name.copy()
    geo_units_real_name_th.remove('Land surface')
    geo_units_real_name_th.remove('Carbonifere and Basement')
    #
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(7, 10),
                            sharey=False)
    for i, ax in enumerate(fig.axes):
        # for i in range(len(geo_units_real_name_th)):
        #    ax = axs[i]
        ztop = zall[:, :, i]
        zbot = zall[:, :, i+1]
        z = ztop-zbot
        zmin = round(np.nanmin(z), -2)
        zmax = round(np.nanmax(z), -2)
        print(f'i={i}, zmin={zmin}, zmax={zmax}')
        if (i == 0 or i == 1 or i == 2 or i == 4):
            #levels = np.linspace(-400, 600, 11)
            levels = [-500, 0, 100, 200, 300, 400, 500, 600]
        elif i == 3:
            #levels = np.linspace(-500, 5000, 11)
            levels = [-500, 0, 1000, 2000, 3000, 4000, 5000]
        elif i == 5:
            #levels = np.linspace(-2000, 2500, 9)
            levels = [-2000, 0, 500, 1000, 1500, 2000, 2500]
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        stitle = '(' + fig_id[i] + ') ' + geo_units_real_name_th[i]
        grid_plot(stitle, i, x, y, z, cmap, norm,
                  levels, fsize, odir, fig, ax)

    # Save file
    ofile = odir + '/all_ele_diff.png'
    fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')
    print(f'Saved {ofile}')

if plot_th_one_unit:
    # define some parameters
    idir = r'c:/Users/hpham/Documents/P32_Niger_2019/04_GMS/layer_ele/'
    fig_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    cmap = plt.get_cmap('jet')  # RdYlBu, gist_rainbow, bwr, jet, BuGn_r,
    arr_dim = (nrows, ncols, nlays+1)
    #
    geo_units_ele = geo_units_ele.copy()
    geo_units_ele.remove('MNT_Skua_2m_grd_zm')
    #
    x, y, zall = read_data(idir, geo_units_ele, arr_dim)
    # zall[zall == -99999] = np.nan
    #
    # Plot a subplot for all units
    geo_units_real_name_th = geo_units_real_name.copy()
    geo_units_real_name_th.remove('Land surface')
    geo_units_real_name_th.remove('Carbonifere and Basement')
    #

    for i in range(len(geo_units_real_name_th)):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4.6),
                                sharey=False)
        ax = axs
        ztop, zbot = zall[:, :, i],  zall[:, :, i+1]
        z = ztop-zbot
        zmin, zmax = round(np.nanmin(z), -2), round(np.nanmax(z), -2)
        print(f'i={i}, zmin={zmin}, zmax={zmax}')
        if (i == 0 or i == 1 or i == 2 or i == 4):
            #levels = np.linspace(-400, 600, 11)
            levels = [-500, 0, 100, 200, 300, 400, 500, 600]
        elif i == 3:
            #levels = np.linspace(-500, 5000, 11)
            levels = [-500, 0, 1000, 2000, 3000, 4000, 5000]
        elif i == 5:
            #levels = np.linspace(-2000, 2500, 9)
            levels = [-2000, 0, 500, 1000, 1500, 2000, 2500]
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        stitle = '(' + fig_id[i] + ') ' + geo_units_real_name_th[i]
        grid_plot(stitle, i, x, y, z, cmap, norm,
                  levels, fsize, odir, fig, ax)

        # Save file
        ofile = odir + '/Ele DIFF ' + geo_units_real_name_th[i] + '.png'
        fig.savefig(ofile, dpi=150, transparent=False, bbox_inches='tight')
        print(f'Saved {ofile}')

# References:
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/pcolormesh_levels.html#sphx-glr-gallery-images-contours-and-fields-pcolormesh-levels-py
# https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
