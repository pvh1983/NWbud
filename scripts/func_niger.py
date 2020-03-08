import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def read_data(idir, ifiles, arr_dim):  # Read ifiles and save to a numpy arrya
    nrows, ncols, nlays = arr_dim
    zall = np.empty(shape=arr_dim)
    for i, geon in enumerate(ifiles):
        geo_name = geon
        ifile = idir + geo_name + '.txt'
        data = np.loadtxt(ifile, delimiter=',')

        #
        data = data[data[:, 0] <= 827871.400]
        data = data[data[:, 1] <= 2493481.000]
        
        #
        z = np.reshape(data[:, 2], [ncols, nrows])
        x = np.unique(data[:, 0])/1e6
    #    x = np.sort(x)
        y = np.unique(data[:, 1])/1e6
    #    y = np.sort(y)
        #z = np.reshape(data[:, 2], [ncols, nrows])
        z = np.transpose(z)
        # z = np.flipud(z)
        z[z == -99999] = np.nan
        #z[z < 0.1] = np.nan
        zall[:, :, i] = z
    return x, y, zall


def fig_font_size():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


def gen_outdir(odir):
    # odir = '../output/check_top_ele'
    if not os.path.exists(odir):  # Make a new directory if not exist
        os.makedirs(odir)
        print(f'\nCreated directory {odir}\n')


def get_geo_units():
    geo_units_th = ['Cont_Term_Map_XS_TVT_2m_grd_zm',
                    # 'Paleo_Map_XS_TVT_2m_grd_zm',
                    'Cret_Sup_Map_XS_TVT_2m_grd_zm',
                    'Cont_Ham_Map_XS_TVT_2m_grd_zm',
                    'Cont_Int_Map_XS_TVT_2m_grd_zm',
                    'Juras_Map_XS_TVT_2m_grd_zm',
                    'Trias_Map_XS_TVT_2m_grd_zm']
    geo_units_ele = ['MNT_Skua_2m_grd_zm',
                     'Cont_Term_Map_XS_2m_grd_zm',
                     # 'Paleo_Map_XS_2m_grd_zm',  # this unit was merged with CT
                     'Cret_Sup_Map_XS_2m_grd_zm',
                     'Cont_Ham_Map_XS_2m_grd_zm',
                     'Cont_Int_Map_XS_2m_grd_zm',
                     'Juras_Map_XS_2m_grd_zm',
                     'Trias_Map_XS_2m_grd_zm',
                     'Socle_Skua_Map_XS_2m_grd_zm']
    #
    geo_units_real_name = ['Land surface',
                           'Continental Terminal (CT)',
                           'Crétacé Supérieur (Upper Cretaceous)',
                           'Continental Hamadien (CH)',
                           'Continental Intercalaire (Ci)',
                           'Jurassic',
                           'Triassic',
                           'Carbonifere and Basement']
    return geo_units_th, geo_units_ele, geo_units_real_name


def grid_plot(geo_name, i, x, y, z, cmap, norm, levels, fsize, odir, fig, ax):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # font = {'size': fsize}
    # plt.rc('font', **font)
    fig_font_size()
    #
    s = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm,  # rasterized=True,
                      alpha=1)  # np.arange(0, 1.01, 0.01)
    # s = ax.imshow(x, y, z)
    # Color bar inside
    # cbax = inset_axes(ax, width="10%", height="1%", loc=3)
    fig.colorbar(s, ax=ax, boundaries=levels, shrink=0.9,  # pad=0.01,
                 orientation='horizontal')
    # fig.colorbar(s, ax=cbax, boundaries=levels,  # shrink=0.9, pad=0.01,
    #             orientation='horizontal')  # shrink=.4, pad=0.01,

    # plt.colorbar(cax=cbaxes, ticks=[0.,1], orientation='horizontal')
    # contourf

    # plt.plot(x, y, 'k.')
    # plt.xlabel('UTM_X', fontsize=fsize)
    # plt.ylabel('UTM_Y', fontsize=fsize)
    ax.set_title(geo_name)  # fontsize=fsize
    plt.xticks(rotation=30)

    '''
    # Add DRI model domail
    # Create a Rectangle patch
    # DRI domain (smaller)
    xtopleft = -345000/1e6
    ytopleft = 1350000/1e6
    rwidth = 750000/1e6
    rheight = 600000/1e6
    # rect = patches.Rectangle((xtopleft, ytopleft), rwidth,
    #                         rheight, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    # ax.add_patch(rect)
    '''
    ax.grid(False)

    # Save file
    # ofile = odir + '/' + str(i+1) + '_ele_' + geo_name + '.png'
    # fig.savefig(ofile, dpi=150, transparent=False, bbox_inches='tight')
    # print(f'Saved {ofile}')
    # plt.close(fig)
    # return ax, fig
    # if (i == 7 or i == 8):
    #    subplot.clf()  # which clears data and axes
