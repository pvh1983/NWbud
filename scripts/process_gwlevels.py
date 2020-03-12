
from func4gwlevel import *
import pandas as pd
import os


# Use conda env irp
# Last updated: 01/30/2020
# Windows:  cd c:\Users\hpham\Documents\P32_Niger_2019\NWbud\scripts\
# MacOS:    cd Dropbox/Study_2019_Dropbox/P32\ Niger/

# [1] Choose options to run:
# [1.1] Raw data (to combine two date columns),
# run once to get ../input/CaracteristiquesPEM_clean.csv

'''
LIST OF DAT SETS
1. CaracteristiquesPEM.xls (Ministry of Hydraulic2s)
2. Wells data REGION DE MARADI.xlsx (Alan Fryar)
3. PEUEMOABERIAF Wells.csv (received 2020-01-24)
4. Données Puits Copie de PEM FALMEY, DOSSO, GAYA (5).xls (received from Michel Wakil 2020-01-28)
[See email KP Jan 30, 2020]
Link to dataset: 
https://dri0-my.sharepoint.com/:f:/g/personal/karl_pohlmann_dri_edu/Eqiq6bElA6NNpWUZEFCoGnkBYEGscPu3cAnS-_qOCaASbA?e=jYIkR0

'''

# [Step 1] ====================================================================
opt_process_raw_data = False
# Choose one of four availale raw files
# 1 'Ministry of Hydraulics', 2 'Alan Fryar', 3 'PEUEMOABERIAF', 4 'Michel Wakil',
# 5 Ministry_of_Hydraulics_new_altitude, 6: Alan Fryar with altitude
dataset_in = 'Alan Fryar with altitude'
ifile_raw = choosing_dataset(dataset_in)


# [Step 2] Clean up the data ==================================================
opt_cleanup_data = False
opt_plot_gwlevel_vs_year = False

# [Step 3] Mapping in 2D ======================================================
opt_map_gwlevel_in2D = True
show_well_loc = True
opt_contour = '_contour_'  # '_contour_' or leave it blank

# [Step 4] Get mean and time series ===========================================
opt_get_mean_gwlevel_given_year = False
cutoff_year = 2020  # to eleminate typos in year.
min_nobs = 2  # To find wells with long-term observations

# Some options ================================================================
well_type = 'All'  # Shallow vs. Deep vs. All
thres_depth_min, thres_depth_max, levels = get_par_4well_depth(well_type)

# Create a new directory to save outputs
odir = '../output/gwlevel/' + dataset_in + '/'
if not os.path.exists(odir):  # Make a new directory if not exist
    os.makedirs(odir)
    print(f'\nCreated directory {odir}\n')

# Process raw data CaracteristiquesPEM.xls (Ministry of Hydraulics)
if opt_process_raw_data:
    print(f'\nReading raw data file {ifile_raw}\n')
    df = pd.read_csv(ifile_raw, encoding="ISO-8859-1")
    proceess_raw_data(df, dataset_in)

if opt_cleanup_data:
    print('\nRunning Step 2: opt_cleanup_data ... \n')
    #ifile = r'../input/gwlevel/CaracteristiquesPEM_clean.csv'
    ifile = odir + dataset_in + ' dataset clean.csv'
    ifile_log_out = odir + 'logs ' + dataset_in + ' ' + well_type + ' wells.txt'
    df = func_cleanup_data(ifile, ifile_log_out,
                           thres_depth_min, thres_depth_max, well_type, odir, cutoff_year, dataset_in)

# Plot groundwater levels vs years
if opt_plot_gwlevel_vs_year:
    ifile_gwlevel_clean = odir + 'df_filtered ' + \
        dataset_in + ' ' + well_type + ' wells.csv'
    df = pd.read_csv(ifile_gwlevel_clean, encoding="ISO-8859-1")
    func_plot_gwlevel_vs_year(df, odir, well_type, dataset_in)

# Map the location of measurements and generate gwlevel contours
if opt_map_gwlevel_in2D:
    # [latmin, latmax, longmin, longmax]
    xmin, xmax, ymin, ymax = [0, 16, 10, 24]
    domain = [xmin, xmax, ymin, ymax]
    ifile_gwlevel_clean = odir + 'df_filtered ' + \
        dataset_in + ' ' + well_type + ' wells.csv'
    df = pd.read_csv(ifile_gwlevel_clean, encoding="ISO-8859-1")
    func_map_gwlevel_in2D(df, domain, levels, opt_contour,
                          show_well_loc, odir, well_type, dataset_in)


# [] Get mean of gwlevels at the same location but different time
if opt_get_mean_gwlevel_given_year:
    ifile_gwlevel_clean = odir + 'df_filtered ' + \
        dataset_in + ' ' + well_type + ' wells.csv'
    print(f'\nReading input file {ifile_gwlevel_clean}\n')
    df = pd.read_csv(ifile_gwlevel_clean)
    df_mean = func_get_mean_gwlevel_given_year(df, min_nobs, odir)


print('Done all!')

# plt.show()

# References
# https://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap
# https://github.com/matplotlib/basemap/blob/master/examples/customticks.py
