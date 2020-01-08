# import rasterio
# import rasterio.plot
# import pyproj
# import numpy as np
# import matplotlib
# import geopandas as gpd
import georaster as gr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
#from mpl_toolkits.basemap import Basemap
import numpy as np
import os

# Last updated: Jan 07, 2020
# cd c:\Users\hpham\Documents\P32_Niger_2019\NWbud\scripts\

'''
OTHER NOTES
# make sure georaster NOT georasterS
# Use conda env irp
# Make sure you choose a correct value for masking
'''

# Path to data directory
data_path = r'c:/Users/hpham/Documents/P32_Niger_2019/NWbud/input/'

dataset = 'FAO WaPOR'  # 'FAO WaPOR' vs. 'USGS MODIS ETa'
# dataset = 'USGS MODIS ETa'  # 'FAO WaPOR' vs. 'USGS MODIS ETa'
#dataset = 'FAO WaPOR vs. USGS MODIS'

if dataset == 'FAO WaPOR':
    star_year, stop_year = 2009, 2019
elif dataset == 'USGS MODIS ETa':
    star_year, stop_year = 2003, 2018

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

for year in range(star_year, stop_year + 1, 1):  # FAO
    if dataset == 'FAO WaPOR':
        ifile = data_path + 'EVT/FAO Actual ET and Interception ETIa/' + \
            'L2_NER_AETI_' + str(year)[-2:] + '.tif'  # FAO only Niger
        #ifile = 'FAO WaPOR2.tif'
        cof = 0.1  # the pixel value in the downloaded data must be multiplied by 0.1
        nodata_val = 0
    elif dataset == 'USGS MODIS ETa':
        # ifile = 'y2018_modisSSEBopETv4_actual_mm.tif' # Org file, worldwide
        # ifile = 'y2003_modisSSEBopETv4_Clip1.tif'  # Clipped data for Niger
        #ifile = 'USGS MODIS2.tif'
        ifile = data_path + 'EVT/USGS MODIS ETa Clipped/y' + \
            str(year) + '_modisSSEBopETv4_Clip1.tif'
        cof = 1
        nodata_val = 32767
    elif dataset == 'FAO WaPOR vs. USGS MODIS':
        ifile = 'Difference1.tif'
        levels = [-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]
        cmap = plt.get_cmap('bwr')  # RdYlBu, gist_rainbow, bwr, jet, BuGn_r,
        cof = 1
        nodata_val = -99999

    print(f'Reading file {ifile} \n')

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # To use WGS84 coordinates:
    my_image = gr.SingleBandRaster(ifile, load_data=extent, latlon=True)
    #my_image = gr.SingleBandRaster(ifile)
    # my_image = gr.SingleBandRaster(ifile, load_data=extent)
    print(my_image.extent)
    print(my_image.nx, my_image.ny)

    # Plotting ...

    fig, ax = plt.subplots()

    fig.set_size_inches(8, 6.5)

    # m = Basemap(width=12000000,height=9000000,projection='lcc',
    #            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
    # draw coastlines.
    # m.drawcoastlines()

    # map = Basemap(llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax,
    #             projection='lcc', lat_1=none, lat_2=none, lon_0=8)  # lat_1=17, lat_2=10,

    # map = Basemap(projection='lcc', lon_0=8, lat_0=18, height=2000000,
    #              width=2000000)

    # map = Basemap(llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax,
    #              projection='lcc', lat_1=33, lat_2=45, lon_0=-95)

    v = my_image.r
    v = v.astype('float')

    if dataset == 'FAO WaPOR':
        v[v < nodata_val] = 'nan'
    else:
        v[v == nodata_val] = 'nan'

    cmap_ = 'jet'
    t = f'Annual Actual ET (mm) in {year}. Source: ' + dataset
    if dataset == 'FAO WaPOR vs. USGS MODIS':
        t = 'ET_Diff (mm) = [USGS MODIS ETa] - [FAO WaPOR]'
        cmap_ = 'bwr'
    plt.title(t)

    plt.imshow(v*cof, vmin=min(levels), vmax=max(levels), cmap=cmap_,  # cmap='viridis_r'
               norm=norm, alpha=1, extent=extent)
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
    xt, yt, = 1.315733, 14.270843  # Long, Lat @ river
    val = cof*my_image.value_at_coords(xt, yt, window=1)
    print(f'v={val} at x={xt}, y={yt}')

    plt.scatter(xt, yt, s=20, marker='o', c='k')

    ofile = odir + 'ET ' + str(year) + ' ' + dataset + '.png'
    fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')
    print(f'Outputs saved: {ofile}')

plt.show()
