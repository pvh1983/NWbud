import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import datetime as dt
from metpy.interpolate import (interpolate_to_grid, remove_nan_observations,
                               remove_repeat_coordinates)
from mpl_toolkits.basemap import Basemap
import os


def choosing_dataset(dataset_in):
    if dataset_in == 'Ministry of Hydraulics':
        #print('Testing \n')
        ifile_raw = r'../input/gwlevel/CaracteristiquesPEM.csv'  # data set 1
    elif dataset_in == 'Alan Fryar':
        ifile_raw = r'../input/gwlevel/Wells data REGION DE MARADI.csv'  # data set 2
    elif dataset_in == 'PEUEMOABERIAF':
        ifile_raw = r'../input/gwlevel/Export_Output_PEUEMOABERIAF_with_altitude.csv'  # data set 3
    elif dataset_in == 'Michel Wakil':
        ifile_raw = r'../input/gwlevel/Michel Wakil Donnees Puits Copie de PEM FALMEY DOSSO GAYA hp.csv'  # data set 4
    elif dataset_in == 'Ministry_of_Hydraulics_new_altitude':
        # Dataset 5, same as Dataset 1 but added altitudes from SRTM.
        ifile_raw = r'../input/gwlevel/CaracteristiquesPEM_added_altitude.csv'
    elif dataset_in == 'Alan Fryar with altitude':
        # Dataset 6, same as Dataset 2 but added altitudes from SRTM.
        ifile_raw = r'../input/gwlevel/df_filtered Alan Fryar All wells with xycoors added altitudes final.csv'

    print(f'Reading file {ifile_raw} \n')
    return ifile_raw


def get_par_4well_depth(well_type):
    if well_type == 'Shallow':
        thres_depth_min, thres_depth_max = -100, 50  # meter.
        # levels = [0,5,10,15,20,25,30,35,40,50]  # For gwlevels
    #    levels = [0,100,200,300,400,500,600]  # For gwlevels
        levels = range(100, 500, 25)
    elif well_type == 'Deep':
        thres_depth_min, thres_depth_max = 50, 1e3  # meter.
        # levels = [0,5,10,15,20,25,30,35,40,50,60,80,100,150,200,250]  # For gwlevels
    #    levels = [0,100,200,300,400,500,600]  # For gwlevels
        levels = range(100, 500, 25)
    elif well_type == 'All':
        thres_depth_min, thres_depth_max = -100, 1e3  # meter.
        # levels = [0,5,10,15,20,25,30,35,40,60,80,100,150,200]  # For gwlevels
        levels = range(100, 500, 25)
    return thres_depth_min, thres_depth_max, levels


def proceess_raw_data(df, dataset_in):
    # Define some input parameters
    col_keep = ['IRH PEM', 'Name PEM', 'XCoor', 'YCoor', 'Altitude', 'Static level', 'Flow',
                'Start of Foration', 'End of Foration', 'Depth Drilled', 'Equipped depth']
    df = df[col_keep]

    # Merge two date columns
    df['Date'] = df['End of Foration']
    for i in range(df.shape[0]):
        if isinstance(df['Date'].iloc[i], str) == True:  # If is string (date string)
            pass
        # If empty, take value of 'Start of Foration'
        elif math.isnan(df['Date'].iloc[i]) == True:
            df['Date'].iloc[i] = df['Start of Foration'].iloc[i]
    ofile = '../output/gwlevel/' + dataset_in + \
        '/' + dataset_in + ' dataset clean.csv'
    df.to_csv(ofile, index=None,  encoding="ISO-8859-1")
    print(f'New data was saved at {ofile} \n')


def func_cleanup_data(ifile, ifile_out, thres_depth_min, thres_depth_max, well_type, odir, cutoff_year, dataset_in):
    # Read input file
    #ifile = r'../input/gwlevel/CaracteristiquesPEM_clean.csv'
    df = pd.read_csv(ifile, encoding="ISO-8859-1")

    print(f'Reading input file {ifile}\n')

    f = open(ifile_out, 'w')

    print(f'Original Dataframe size: {df.shape}', file=f)

    df = df.dropna(subset=['XCoor', 'YCoor'])
    print(
        f'Data AFTER removing NaN rows at columns [XCoor] and [YCoor] is: {df.shape}', file=f)

    ofile = odir + 'df_filtered ' + dataset_in + \
        ' ' + well_type + ' wells with xycoors.csv'
    df.to_csv(ofile, index=False, encoding="ISO-8859-1")
    print(f'Filted gwlevels were saved at {ofile} \n')

    # Make sure no space in empty cell in this column
    df = df.dropna(subset=['Date'])
    print(
        f'Data AFTER removing NaN rows at column [Date] is: {df.shape}', file=f)

    df = df.dropna(subset=['Depth Drilled'])
    print(
        f'Data AFTER removing NaN rows at column [Depth Drilled] is: {df.shape}', file=f)

    df = df.dropna(subset=['Static level'])
    # Depth to water 3000 m (Del all large than 3000 m b/c type)
    df = df[df['Static level'] < 3000]
    print(
        f'Data AFTER removing NaN rows at column [Static level] is: {df.shape}', file=f)

    df = df.dropna(subset=['Altitude'])
    print(
        f'Data AFTER removing NaN rows at column [Altitude] is: {df.shape}', file=f)

    df = df.dropna(subset=['Name PEM'])
    print(
        f'Data AFTER removing NaN rows at column [Name PEM] is: {df.shape}', file=f)

    df = df[df['Depth Drilled'] > thres_depth_min]
    df = df[df['Depth Drilled'] < thres_depth_max]

    print(
        f'Dataframe AFTER apply threshold of {well_type} wells: {df.shape}', file=f)
    df = df.reset_index()

    # print(df['Date'])
    # df.to_csv('../output/gwlevel/testing.csv')

    # Make sure no space in empty cell in this column
    date = pd.to_datetime(df['Date'])
    print(
        f'Date min/max of the current dataframe: {date.min().year}, {date.max().year} \n', file=f)

    # Delete observations after 2020 (probably, due to typo)
    df['Date'] = pd.to_datetime(df['Date'])
    check_date = pd.Timestamp(dt.date(cutoff_year, 1, 1))
    df = df[df['Date'] < check_date]
    print(
        f'Dataframe AFTER removing data measured after {check_date}: {df.shape}', file=f)
    # print(
    #    f'(Considering the data before this year {check_date} as the pre-development condition)', file=f)
    date = pd.to_datetime(df['Date'])
    print(f'Date min/max = {date.min().year}, {date.max().year} \n', file=f)

    date = pd.to_datetime(df['Date'])
    print(
        f'Date min/max of the current dataframe: {date.min().year}, {date.max().year} \n', file=f)

    #check_gwdepth_thre = 500
    #df = df[df['Static level'] < check_gwdepth_thre]
    #print(f'Dataframe AFTER removing typo gwdepth > {check_gwdepth_thre} (m): {df.shape}', file=f)
    df = df.reset_index()

    # write the final dataframe to a csv file
    ofile = odir + 'df_filtered ' + dataset_in + ' ' + well_type + ' wells.csv'
    df.to_csv(ofile, index=False, encoding="ISO-8859-1")
    print(f'Filted gwlevels were saved at {ofile} \n')
    f.close()
    return df


def func_plot_gwlevel_vs_year(df, odir, well_type, dataset_in):
    # [] Plot groundwater levels vs years
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6.5)
    plt.grid(color='#e6e6e6', linestyle='-', linewidth=0.5, axis='both')
    x = pd.to_datetime(df['Date'])
    y = df['Altitude'] - df['Static level']
    #df.plot(x='Date', y='Static level', style='o')
    # plt.plot(x,y,'o')
    #plt.scatter(x,y,s=120, marker ='o', linewidths =0.25, facecolors=None)
    plt.plot(x, y, 'o', color='gray',
             markersize=6, linewidth=0.25,
             markerfacecolor='white',
             markeredgecolor='k',
             markeredgewidth=0.25, alpha=0.9)
    plt.ylabel('Groundwater level (m)', fontsize=14)

    ofile = odir + 'plot by year dataset ' + \
        dataset_in + ' for ' + well_type + ' wells.png'
    fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')
    print(f'\nThe figure was saved at {ofile} \n')
    plt.show()


def func_map_gwlevel_in2D(df, domain, levels, opt_contour, show_well_loc, odir, well_type, dataset_in):
    # Plot
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6.5)

    x = df.XCoor
    y = df.YCoor
    #val = df.WLE
    val = df['Altitude'] - df['Static level']
    print(f'Depth min/max (meter): {val.min()}, {val.max()} \n')

    xmin, xmax, ymin, ymax = domain
#    map = Basemap(llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax,
#                  resolution='h',
#                  projection='lcc', lat_1=10, lat_2=10, lon_0=-95)  # lat_1=17, lat_2=10,

    map = Basemap(llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax,
                  projection='lcc', resolution='l',  lat_1=10, lat_2=10, lon_0=-95, epsg=4269)

    #map.shadedrelief()    #
    # map.etopo()
    map.drawcountries()
    map.drawstates()
    # map.drawmapboundary(fill_color='#46bcec')

    # Plot BACKGROUND =========================================================
    showbg = False
    if showbg == True:
        #        m.arcgisimage(service='ESRI_Imagery_World_2D',
        #                      xpixels=1500, ypixel=None, dpi=300, verbose=True, alpha=1)  # Background
        map.arcgisimage(service='ESRI_Imagery_World_2D',
                        xpixels=1500, dpi=300, verbose=True, alpha=0.25)  # Background

    cmap = plt.get_cmap('jet')  # RdYlBu, gist_rainbow, bwr, jet, BuGn_r,
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    x1, y1 = map(x.values, y.values)

    # Delete bad locations
    # bad_locx = np.where(x1 > 1e9)  # delete bad records
    # bad_locy = np.where(y1 > 1e9) # delete
    #x1 = np.delete(x1, bad_locx)
    #y1 = np.delete(y1, bad_locx)
    #print(x1, y1)

    val = val.to_numpy()
    #val = np.delete(val, bad_locx)
    print('Shape of x1, y1 and v1')
    print(x1.shape, y1.shape, val.shape)

    gx, gy, img1 = interpolate_to_grid(
        x1, y1, val, interp_type='barnes', hres=5000, search_radius=40000)  # work: 10000/40000

    img1 = np.ma.masked_where(np.isnan(img1), img1)
    print(gx.shape, gy.shape, img1.shape)

#    gx, gy, img1 = interpolate_to_grid(
#        x1, y1, val, interp_type='linear', hres=20000) #

    #img1 = np.ma.masked_where(np.isnan(img1), img1)
    if opt_contour == '_contour_':
        # RdYlBu, gist_rainbow, bwr, jet
        #print(gx.shape, gy.shape)
        s = ax.pcolormesh(gx, gy, img1,
                          cmap=cmap, norm=norm, alpha=1)
        #
        # Interpolate
        #rbf = scipy.interpolate.Rbf(gx, gy, val, function='linear')
        #zi = rbf(gx, gy)
        # s = plt.imshow(val, vmin=val.min(), vmax=val.max(), origin='lower',
        #        extent=[gx.min(), gx.max(), gy.min(), gy.max()])
        #CS = plt.contour(gx, gy, gaussian_filter(img1, 5.), 4, colors='k',interpolation='none')

        fig.colorbar(s, shrink=.4, pad=0.01, boundaries=levels)
        show_well_loc = False
        # map.drawmapscale(-1.0, 10, 4.1, 14, 500)  # Why not working now?

    # Show well locations
    if show_well_loc:
        # ax.scatter(x1,y1,s=1,c='k',marker='o')
        plt.plot(x1, y1, 'o', color='gray',
                 markersize=3, linewidth=0.5,
                 markerfacecolor='white',
                 markeredgecolor='k',
                 markeredgewidth=0.5, alpha=0.9)
    # drawmapscale(lon, lat, lon0, lat0, length, barstyle=’simple’, units=’km’,
    # fontsize=9, yoffset=None, labelstyle=’simple’, fontcolor=’k’, fillcolor1=’w’,
    # fillcolor2=’k’, ax=None, format=’%d’, zorder=None)
    ofile = odir + 'map gwdepth ' + dataset_in + ' for ' + \
        well_type + ' wells ' + opt_contour + '.png'
    fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')
    print(f'\nThe figure was saved at {ofile} \n')
    plt.show()


def func_get_mean_gwlevel_given_year(df, min_nobs, odir_):

    odir = odir_ + '/time series at least ' + \
        str(min_nobs) + ' meas/'
    if not os.path.exists(odir):  # Make a new directory if not exist
        os.makedirs(odir)
    print(f'\nCreated directory {odir}\n')

    uwname = df['Name PEM'].unique()
    dfout = pd.DataFrame(columns=['Name', 'Lat', 'Long', 'gwlevel'])
    Name_tmp = []
    Lat_tmp = []
    Long_tmp = []
    gwlevel_tmp = []
    depth_tmp = []
    count = 0

    for i, wn in enumerate(uwname):
        dfi = df[df['Name PEM'] == wn]
        print(f'i={i+1}, nobs = {dfi.shape[0]}, well_name = {wn}')
        if dfi.shape[0] >= min_nobs:
            ofile_dfi = odir + wn + '.csv'
            dfi.to_csv(ofile_dfi, encoding="ISO-8859-1")
            count += 1
        mean_gw = dfi['Static level'].mean(skipna=True)
        Name_tmp.append(wn)
        Lat_tmp.append(dfi['YCoor'].iloc[0])
        Long_tmp.append(dfi['XCoor'].iloc[0])
        gwlevel_tmp.append(mean_gw)
        depth_tmp.append(dfi['Depth Drilled'].iloc[0])
        # print(mean_gw)
    dfout['Name'] = Name_tmp
    dfout['Long'] = Long_tmp
    dfout['Lat'] = Lat_tmp
    dfout['Depth'] = depth_tmp
    dfout['gwlevel'] = gwlevel_tmp
    print(f'Found {count} well locations that have more than {min_nobs} \n')
    return dfout

# More background map
# https://kbkb-wx-python.blogspot.com/2016/04/python-basemap-background-image-from.html


'''
map_list = [
'ESRI_Imagery_World_2D',    # 0
'ESRI_StreetMap_World_2D',  # 1
'NatGeo_World_Map',         # 2
'NGS_Topo_US_2D',           # 3
#'Ocean_Basemap',            # 4
'USA_Topo_Maps',            # 5
'World_Imagery',            # 6
'World_Physical_Map',       # 7     Still blurry
'World_Shaded_Relief',      # 8
'World_Street_Map',         # 9
'World_Terrain_Base',       # 10
'World_Topo_Map'            # 11
]
'''
