
from scipy.ndimage.filters import gaussian_filter
import datetime as dt
import pandas as pd
import geopandas as gpd
import os
import sys
import math
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import scipy.interpolate
from metpy.cbook import get_test_data
from metpy.interpolate import (interpolate_to_grid, remove_nan_observations,
                               remove_repeat_coordinates)
from metpy.plots import add_metpy_logo

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh


# [1] Choose options to run:
# [1.1] Raw data (to combine two date columns), run once to get CaracteristiquesPEM_clean.csv
process_date = False

opt_plot = True
opt_get_ts = False  # Get time series for each well [not done yet]
show_well_loc = True
opt_contour = ''  # '_contour_' or leave it blank

cutoff_year = 2020

# Use conda env irp
# Last updated: 12/22/2019
# Windows:  cd c:\Users\hpham\Dropbox\Study_2019_Dropbox\P32 Niger\00_codes\
#           cd c:\Users\hpham\Documents\P32_Niger_2019\NWbud\scripts\
# MacOS:    cd Dropbox/Study_2019_Dropbox/P32\ Niger/


#df = pd.read_csv('../01_gwlevel/gwdepth_Niger.csv', encoding="ISO-8859-1")
# df = pd.read_csv('../01_gwlevel/process_gwlevel.csv')  # Processed data by WB

# Processed data by WB with depth
#df = pd.read_csv('../01_gwlevel/PEM_WL_WRB.csv')

ifile = r'../input/gwlevel/CaracteristiquesPEM_clean.csv'
df = pd.read_csv(ifile)

well_type = 'All_wells'  # Shallow vs. Deep vs. All_wells
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
elif well_type == 'All_wells':
    thres_depth_min, thres_depth_max = -100, 1e3  # meter.
    # levels = [0,5,10,15,20,25,30,35,40,60,80,100,150,200]  # For gwlevels
    levels = range(100, 500, 25)

# Open a file to print out results
odir = '../output/gwlevel/'
if not os.path.exists(odir):  # Make a new directory if not exist
    os.makedirs(odir)
    print(f'\nCreated directory {odir}\n')
ifile_out = odir + 'logs_gwlevels_' + well_type + '_wells.txt'
f = open(ifile_out, 'w')

#
if process_date:
    # raw data
    #ifile = r'../_gwlevel/CaracteristiquesPEM.csv'
    #df = pd.read_csv(ifile, encoding="ISO-8859-1")

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

    df.to_csv('CaracteristiquesPEM_clean.csv', index=None)

print(f'Original Dataframe size: {df.shape}', file=f)

df = df.dropna(subset=['XCoor', 'YCoor'])
print(
    f'Dataframe AFTER removing NaN of columns [XCoor, YCoor] is: {df.shape}', file=f)

df = df.dropna(subset=['Date'])
print(
    f'Dataframe AFTER removing NaN rows of column [Date] is: {df.shape}', file=f)

df = df.dropna(subset=['Depth Drilled'])
print(f'Dataframe AFTER removing depth NaN is: {df.shape}', file=f)

df = df.dropna(subset=['Static level'])
print(
    f'Dataframe AFTER removing NaN rows of [Static level] colu  is: {df.shape}', file=f)

df = df.dropna(subset=['Altitude'])
print(
    f'Dataframe AFTER removing NaN rows of [Altitude] is: {df.shape}', file=f)

df = df.dropna(subset=['Name PEM'])
print(
    f'Dataframe AFTER removing NaN rows of column [Name PEM] is: {df.shape}', file=f)


df = df[df['Depth Drilled'] > thres_depth_min]
df = df[df['Depth Drilled'] < thres_depth_max]
print(
    f'Dataframe AFTER apply threshold of {well_type} wells: {df.shape}', file=f)
df = df.reset_index()

date = pd.to_datetime(df['Date'])
print(
    f'Date min/max of the current dataframe: {date.min().year}, {date.max().year} \n', file=f)

ofile = odir + 'df_filtered_' + well_type + '_wells.csv'
df.to_csv(ofile, index=False)
print(f'Filted gwlevels were saved at {ofile} \n', file=f)

# Delete observations after 2020 (probably, due to typo)
df['Date'] = pd.to_datetime(df['Date'])
check_date = pd.Timestamp(dt.date(cutoff_year, 1, 1))
df = df[df['Date'] < check_date]
print(
    f'Dataframe AFTER removing data measured after > {check_date}: {df.shape}', file=f)
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
uwname = df['Name PEM'].unique()
dfout = pd.DataFrame(columns=['Name', 'Lat', 'Long', 'gwlevel'])
Name_tmp = []
Lat_tmp = []
Long_tmp = []
gwlevel_tmp = []
depth_tmp = []
for i, wn in enumerate(uwname):
    dfi = df[df['Name PEM'] == wn]
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

# Plot
fig, ax = plt.subplots()
fig.set_size_inches(8, 6.5)
x = df['Date']
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

ofile = odir + 'gwplot_by_year_' + well_type + '.png'
fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')


# Group by well names

if opt_get_ts:
    wname = df['Name PEM'].unique()


if opt_plot:
    # Plot
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6.5)

    x = df.XCoor
    y = df.YCoor
    #val = df.WLE
    val = df['Altitude'] - df['Static level']
    print(f'Depth min/max (meter): {val.min()}, {val.max()} \n', file=f)

    xmin, xmax, ymin, ymax = [-4, 14, 10, 16]
    map = Basemap(llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax,
                  resolution='h',
                  projection='lcc', lat_1=10, lat_2=10, lon_0=-95)  # lat_1=17, lat_2=10,
    map.shadedrelief()    #
    # map.etopo()
    map.drawcountries()
    # map.drawstates()
    # map.drawmapboundary(fill_color='#46bcec')

    cmap = plt.get_cmap('jet')  # RdYlBu, gist_rainbow, bwr, jet, BuGn_r,
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    x1, y1 = map(x.values, y.values)
    bad_locx = np.where(x1 > 1e9)  # delete bad records
    # bad_locy = np.where(y1 > 1e9) # delete
    x1 = np.delete(x1, bad_locx)
    y1 = np.delete(y1, bad_locx)
    val = val.to_numpy()
    val = np.delete(val, bad_locx)

    gx, gy, img1 = interpolate_to_grid(
        x1, y1, val, interp_type='barnes', hres=10000, search_radius=40000)
    img1 = np.ma.masked_where(np.isnan(img1), img1)

#    gx, gy, img1 = interpolate_to_grid(
#        x1, y1, val, interp_type='linear', hres=20000) #

    #img1 = np.ma.masked_where(np.isnan(img1), img1)
    if opt_contour == '_contour_':
        # RdYlBu, gist_rainbow, bwr, jet
        s = plt.pcolormesh(gx, gy, img1, cmap=cmap, norm=norm, alpha=1)
        #
        # Interpolate
        #rbf = scipy.interpolate.Rbf(gx, gy, val, function='linear')
        #zi = rbf(gx, gy)
        # s = plt.imshow(val, vmin=val.min(), vmax=val.max(), origin='lower',
        #        extent=[gx.min(), gx.max(), gy.min(), gy.max()])
        #CS = plt.contour(gx, gy, gaussian_filter(img1, 5.), 4, colors='k',interpolation='none')

        fig.colorbar(s, shrink=.4, pad=0.01, boundaries=levels)
        show_well_loc = False
        map.drawmapscale(-1.0, 10, 4.1, 14, 500)

    # Show locations

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
    ofile = odir + 'gwdepth_' + well_type + '_wells' + opt_contour + '.png'
    fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')

print('Done!')
f.close()
plt.show()

# References
# https://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap
# https://github.com/matplotlib/basemap/blob/master/examples/customticks.py
