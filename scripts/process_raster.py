# import rasterio
# import rasterio.plot
# import pyproj
# import numpy as np
# import matplotlib
import geopandas as gpd
import georaster as gr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
import pandas as pd
#from shapely.geometry import Polygon
#import cartopy.crs as ccrs

# Last updated: Jan 07, 2020
# cd c:\Users\hpham\Documents\P32_Niger_2019\NWbud\scripts\

'''
OTHER NOTES
# make sure georaster NOT georasterS
# Use conda env irp
# Make sure you choose a correct value for masking
# Run all dataset to get *.csv file FIRSTLY.
'''


# [01] Get maps of EVT for each year and time series at given points
get_spatio_temp_data = True
show_loc = True  # Show locations where data are extracted.
show_country_bou = True  # Show country boundaries

# [02] Plot and compare time series data
#      run get_spatio_temp_data to get two input files
plt_time_series_at_given_points = True


# [03] Choosing a dataset to analyze
dataset = 'FAO_WaPOR'  # 'FAO_WaPOR' vs. 'USGS_MODIS' vs. CHIRPS_5KM
# dataset = 'USGS_MODIS'  # 'FAO_WaPOR' vs. 'USGS_MODIS'
# dataset = 'FAO_WaPOR vs. USGS MODIS'
dataset = 'CHIRPS_5KM'  # 'FAO_WaPOR' vs. 'USGS_MODIS'


# [2] Read file to get the list of locations to extract time series
loc_type = 'ground'  # 'borehole' or 'ground' or 'climate_station'
if loc_type == 'borehole':
    ifile_loc = r'c:/Users/hpham/Documents/P32_Niger_2019/NWbud/input/borehole_location.csv'
    nrows, ncols = 9, 5
elif loc_type == 'ground':
    ifile_loc = r'c:/Users/hpham/Documents/P32_Niger_2019/NWbud/input/ET_station.csv'
    nrows, ncols = 3, 2
elif loc_type == 'climate_station':
    ifile_loc = r'c:/Users/hpham/Documents/P32_Niger_2019/NWbud/input/Niger_weather_station.csv'
    nrows, ncols = 5, 3

dfloc = pd.read_csv(ifile_loc)

# List of station's names
sname = ['Birni N Konni', 'Magaria', 'Maradi Aero',
         'Niamey Aero', 'Tahoua Aero', 'Zinder Aero']
#sname = dfloc.Name.to_list()

if get_spatio_temp_data:
    # Path to data directory
    data_path = r'c:/Users/hpham/Documents/P32_Niger_2019/NWbud/input/'

    if (dataset == 'FAO_WaPOR' or dataset == 'CHIRPS_5KM'):
        star_year, stop_year = 2009, 2019  # 2009, 2019
    elif dataset == 'USGS_MODIS':
        star_year, stop_year = 2003, 2018  # 2003, 2018

    # [5] Create a new folder to save the figures
    odir = '../output/EVT/' + dataset + '/'
    if not os.path.exists(odir):  # Make a new directory if not exist
        os.makedirs(odir)
        print(f'\nCreated directory {odir}\n')

    extent = (0, 16, 10, 24)
    xmin, xmax, ymin, ymax = extent
    levels = range(0, 1700, 50)
    # levels = [0, 50, 300, 300, 400, 500, 700, 850, 1000, 3286]
    cmap = plt.get_cmap('jet')  # RdYlBu, gist_rainbow, bwr, jet, BuGn_r,
    ETdate = range(star_year, stop_year + 1, 1)
    cols = ['Date'] + ['ET_vol'] + list(dfloc.Name)
    dfts = pd.DataFrame(columns=cols)
    dfts['Date'] = ETdate

    for i, year in enumerate(ETdate):  # FAO

        # Plotting ...
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6.5)

        if show_country_bou:
            # ifile_Nbou = r'c:/Users/hpham/Documents/P32_Niger_2019/GIS_Niger/Niger_bou.shp'
            # ifile_Nbou = r'c:/Users/hpham/Documents/P32_Niger_2019/GIS_Niger/Niger_bou_converted.shp'
            ifile_Nbou = r'c:/Users/hpham/Documents/P32_Niger_2019/GIS_Niger/world_country_proj_new31.shp'
            nbou = gpd.GeoDataFrame.from_file(ifile_Nbou)
            nbou.plot(ax=ax, linewidth=1., color='None',
                      edgecolor='#D5DCF9', alpha=0.8)

            # Plot the survey area of 260,000 km2
            ifile_study_area = r'c:/Users/hpham/Documents/P32_Niger_2019/GIS_Niger/Survey_area_260000_deg.shp'
            study_area = gpd.GeoDataFrame.from_file(ifile_study_area)
            study_area.plot(ax=ax, linewidth=1.,
                            color='None', edgecolor='#ff3844', alpha=0.8)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        if dataset == 'FAO_WaPOR':
            #            ifile = data_path + 'EVT/FAO Actual ET and Interception ETIa/' + \
            #                         'L2_NER_AETI_' + str(year)[-2:] + '.tif'  # FAO only Niger
            ifile = data_path + 'EVT/FAO_WaPor_Africa_250m/' + \
                'L1_AETI_' + str(year)[-2:] + \
                '.tif'  # FAO Africa (continantal)

            # ifile = 'FAO_WaPOR2.tif'
            cof = 0.1  # the pixel value in the downloaded data must be multiplied by 0.1
            nodata_val = 0
            xres, yres = 250, 250  # m
            ptitle1 = 'Actual EvapoTranspiration and Interception (mm)'
        elif dataset == 'USGS_MODIS':
            # Original data
            ifile = data_path + 'EVT/Annual Actual ET SSEBop ver4/y' + \
                str(year) + '_modisSSEBopETv4_actual_mm.tif'
            # Clipped data
#            ifile = data_path + 'EVT/USGS_MODIS Clipped/y' + \
#                str(year) + '_modisSSEBopETv4_Clip1.tif'

            cof = 1
            nodata_val = 32767
            xres, yres = 1000, 1000  # m
            ptitle1 = 'Annual Actual Evapotranspiration (mm)'
        elif dataset == 'FAO_WaPOR vs. USGS MODIS':
            ifile = 'Difference1.tif'
            levels = [-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]
            # RdYlBu, gist_rainbow, bwr, jet, BuGn_r,
            cmap = plt.get_cmap('bwr')
            cof = 1
            nodata_val = -99999
        elif dataset == 'CHIRPS_5KM':
            ifile = data_path + 'PRECIP/CHIRPS_5KM/' + \
                'L1_PCP_' + str(year)[-2:] + \
                '.tif'  # FAO Africa (continantal)
            #levels = range(0, 3700, 100)
            cof = 0.1  # the pixel value in the downloaded data must be multiplied by 0.1
            nodata_val = -9999
            xres, yres = 5000, 5000  # m
            ptitle1 = 'CHIRPS Annual Precipitation'

        print(f'Reading file {ifile} \n')

        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # To use WGS84 coordinates:
        my_image = gr.SingleBandRaster(ifile, load_data=extent, latlon=True)

        #my_image = gr.SingleBandRaster(ifile, latlon=True)
        # my_image2 = gr.SingleBandRaster(ifile, load_data=extent)
        print(my_image.extent)
        print(my_image.nx, my_image.ny)
        #xres, yres = my_image.xres, my_image.yres
        # m = Basemap(width=12000000,height=9000000,projection='lcc',
        #            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
        # draw coastlines.
        # m.drawcoastlines()

        v = my_image.r
        v = v.astype('float')

        # Testing a point
        '''
        xt, yt, = 1.315733, 14.270843  # Long, Lat @ river
        val = my_image.value_at_coords(xt, yt, window=1)
        print(f'v={val} at the test point x={xt}, y={yt}')
        '''

        if dataset == 'FAO_WaPOR':
            v[v < nodata_val] = 'nan'
        else:
            v[v == nodata_val] = 'nan'

        cmap_ = 'jet'
        # t = f'{ptitle1} in {year}. Source: ' + dataset
        t = f'{ptitle1} in {year}'
        if dataset == 'FAO_WaPOR vs. USGS MODIS':
            t = 'ET_Diff (mm) = [USGS_MODIS] - [FAO_WaPOR]'
            cmap_ = 'bwr'
        plt.title(t)

        plt.imshow(v*cof, vmin=min(levels), vmax=max(levels), cmap=cmap_,  # cmap='viridis_r'
                   norm=norm, alpha=1, extent=extent)
        sum_EVT = np.nansum(v)*xres*yres/1000  # from milimeter to meter
        total_vol_EVT = cof*sum_EVT/1e9  # km3
        dfts['ET_vol'].iloc[i] = total_vol_EVT
        print(f'Total volume of ET = {total_vol_EVT} (km3) \n')
        # axs[2].contour(Z, levels, colors='k', origin='lower', extent=extent)

        plt.colorbar(shrink=0.5)

        # plt.xlabel('Long (degree)')
        # plt.ylabel('Lat (degree)')

        #map.shadedrelief()    #
        # map.etopo()
        # map.drawcountries()
        # map.drawstates()

        # s = plt.pcolormesh(gx, gy, img1, vmin=val.min(), vmax=val.max(),
        #                    cmap='jet', norm=norm, alpha=0.65)  # RdYlBu, gist_rainbow, bwr, jet
        # fig.colorbar(s, shrink=.4, pad=0.01, boundaries=levels)

        # xt, yt, = 13.6141, 12.3855  # Long, Lat @ an egde

        # xt, yt, = 1.315733, 14.270843  # Long, Lat @ river
        for j, pname in enumerate(dfloc['Name']):
            xt = dfloc['Longitude'].iloc[j]
            yt = dfloc['Latitude'].iloc[j]
            icol, irow = my_image.coord_to_px(xt, yt)
            val = cof*my_image.r[int(irow), int(icol)]
            #val = cof*my_image.value_at_coords(xt, yt, window=1)
            dfts[pname].iloc[i] = val
            print(f'v={val} at x={xt}, y={yt}')
            if show_loc:
                plt.scatter(xt, yt, s=12, marker='o', c='#ffff00',
                            edgecolors='k', linewidths=0.25)

        # Save figures (one for each year)
        ofile = odir + 'ET ' + str(year) + ' ' + dataset + '.png'
        fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')
        print(f'Figures were saved: {ofile}')

    ofile = '../output/EVT ' + dataset + '_' + loc_type + '.csv'
    dfts.to_csv(ofile, index=False)


if plt_time_series_at_given_points:
    ifile = r'c:/Users/hpham/Documents/P32_Niger_2019/NWbud/output/EVT FAO_WaPOR.csv'
    df_fao = pd.read_csv(ifile)
    ifile = r'c:/Users/hpham/Documents/P32_Niger_2019/NWbud/output/EVT USGS_MODIS.csv'
    df_usgs = pd.read_csv(ifile)
    ifile = r'c:/Users/hpham/Documents/P32_Niger_2019/NWbud/input/ET_obs.csv'
    df_etobs = pd.read_csv(ifile)
    ifile = r'c:/Users/hpham/Documents/P32_Niger_2019/NWbud/output/EVT CHIRPS_5KM_ground.csv'
    df_prep = pd.read_csv(ifile)

    fig1, axes1 = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=(10, 10))

    for i, ax1 in enumerate(axes1.flatten()):
        # Only plot 44 subplots (because npoints=44)
        #        if i <= df_fao.shape[1]-2:
        # print(i)
        ax1.set_title(sname[i])

        x1 = df_fao['Date']
        y1 = df_fao[sname[i]]
        sc1 = ax1.plot(x1, y1, ':s')

        x2 = df_usgs['Date']
        y2 = df_usgs[sname[i]]
        sc2 = ax1.plot(x2, y2, ':^')

        if (i == 0 or i == 2 or i == 4):
            ax1.set_ylabel('Evapotranspiration (mm)')

        ax1.set_ylim([0, 4000])
        ax1.tick_params(axis='x', rotation=90)

        if loc_type == 'ground':
            x3 = df_etobs['Date']
            y3 = df_etobs[sname[i]]
            sc3 = ax1.plot(x3, y3, '--o')

        if (i == 0 and loc_type == 'ground'):
            ax1.legend(['FAO_WaPOR', 'USGS MODIS',
                        'Penman Monteith'], fontsize=8, loc='upper left')
        elif (i == 0 and loc_type == 'borehole'):
            ax1.legend(['FAO_WaPOR', 'USGS MODIS'], fontsize=8)

        ax1.grid(linewidth=0.1, color='c')
        ax1.xaxis.set_ticks(range(2001, 2019+1, 1))

        # Plot CHIRPS annual precipitation
        ax2 = ax1.twinx()

        x4 = df_prep['Date']
        y4 = df_prep[sname[i]]
        sc4 = ax2.bar(x4, y4, width=0.6, alpha=0.6)
        ax2.set_ylim([0, 3200])
        ax2.invert_yaxis()
        ax2.set_ylabel('Annual Precipitation (mm)', color='blue')
        # ax2.set_
        ax2.xaxis.set_ticks(range(2001, 2019+1, 1))
        # fig1.colorbar(sc1, ax=ax1)
    ofile = '../output/EVT_time_series_comparison_' + loc_type + '.png'
    plt.savefig(ofile, dpi=300, transparent=False)  # bbox_inches='tight'

    # Plot 2: Total volume of ET
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

    x1 = df_fao['Date']
    y1 = df_fao['ET_vol']
    sc1 = ax2.plot(x1, y1, ':s')

    x2 = df_usgs['Date']
    y2 = df_usgs['ET_vol']
    sc2 = ax2.plot(x2, y2, ':^')

    ax3 = ax2.twinx()

    x3 = df_prep['Date']
    y3 = df_prep['ET_vol']
    sc4 = ax3.bar(x3, y3, width=0.5, alpha=0.8, align='center')
    #sc4 = ax3.plot(x3, y3, '-x')
    ax3.set_ylim([0, 3200])
    ax3.invert_yaxis()

    ax2.set_ylim([0, 2000])
    ax2.tick_params(axis='x', rotation=90)
    ax2.legend(['FAO_WaPOR', 'USGS MODIS'], loc='lower left')
    ax2.set_ylabel('Total Volume of Evapotranspiration (km$^3$)')
    ax3.set_ylabel('Total Volume of Precipitation (km$^3$)', color='blue')
    ax2.grid(linewidth=0.1, color='c')
    ax2.xaxis.set_ticks(range(2001, 2019+1, 1))
    ax3.xaxis.set_ticks(range(2001, 2019+1, 1))
    ofile = '../output/Total_EVT_volume_time_series_comparison_' + loc_type + '.png'
    plt.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')


plt.show()


# REFERENCES
# To create a GIF file: https://ezgif.com/maker
#
