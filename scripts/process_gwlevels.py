
from func4gwlevel import *
import pandas as pd
import os
#import sys


#import geopandas as gpd
#from scipy.ndimage.filters import gaussian_filter
#import scipy.interpolate
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#from mpl_toolkits.basemap import Basemap

#from metpy.cbook import get_test_data
# from metpy.interpolate import (interpolate_to_grid, remove_nan_observations,
#                               remove_repeat_coordinates)
#from metpy.plots import add_metpy_logo

#from matplotlib.axes import Axes
#from cartopy.mpl.geoaxes import GeoAxes
#GeoAxes._pcolormesh_patched = Axes.pcolormesh

# Use conda env irp
# Last updated: 01/30/2020
# Windows:  cd c:\Users\hpham\Documents\P32_Niger_2019\NWbud\scripts\
# MacOS:    cd Dropbox/Study_2019_Dropbox/P32\ Niger/

# [1] Choose options to run:
# [1.1] Raw data (to combine two date columns),
# run once to get ../input/CaracteristiquesPEM_clean.csv
opt_process_raw_data = False

# [1.2] Clean up the data (after 1.1)
opt_cleanup_data = False
opt_plot_gwlevel_vs_year = True
opt_get_mean_gwlevel_given_year = False
opt_map_gwlevel_in2D = False
opt_get_ts = False  # Get time series for each well [not done yet]
show_well_loc = False
opt_contour = ''  # '_contour_' or leave it blank
cutoff_year = 2020


well_type = 'All_wells'  # Shallow vs. Deep vs. All_wells
thres_depth_min, thres_depth_max, levels = get_par_4well_depth(well_type)

# Open a file to print out results
odir = '../output/gwlevel/'
if not os.path.exists(odir):  # Make a new directory if not exist
    os.makedirs(odir)
    print(f'\nCreated directory {odir}\n')


# Process raw data CaracteristiquesPEM.xls (Ministry of Hydraulics)
if opt_process_raw_data:
    ifile_raw = r'../_gwlevel/CaracteristiquesPEM.csv'
    df = pd.read_csv(ifile_raw, encoding="ISO-8859-1")
    proceess_raw_data(df)

if opt_cleanup_data:
    print('\nRunning opt_cleanup_data ... \n')
    ifile = r'../input/gwlevel/CaracteristiquesPEM_clean.csv'
    ifile_log_out = odir + 'logs_gwlevels_' + well_type + '_wells.txt'
    df = func_cleanup_data(ifile, ifile_log_out,
                           thres_depth_min, thres_depth_max, well_type, odir, cutoff_year)

# Get mean of gwlevels at the same location but different time
if opt_get_mean_gwlevel_given_year:
    ifile_gwlevel_clean = odir + 'df_filtered_' + well_type + '_wells.csv'
    df = pd.read_csv(ifile_gwlevel_clean)
    df_mean = func_get_mean_gwlevel_given_year(df)

# [] Plot groundwater levels vs years
if opt_plot_gwlevel_vs_year:
    ifile_gwlevel_clean = odir + 'df_filtered_' + well_type + '_wells.csv'
    df = pd.read_csv(ifile_gwlevel_clean)
    func_plot_gwlevel_vs_year(df, odir, well_type)
    

# Group by well names
if opt_get_ts:
    wname = df['Name PEM'].unique()


if opt_map_gwlevel_in2D:
    func_map_gwlevel_in2D()


print('Done all!')

# plt.show()

# References
# https://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap
# https://github.com/matplotlib/basemap/blob/master/examples/customticks.py
