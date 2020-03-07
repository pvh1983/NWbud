
import numpy as np


'''
Read RTI files and convert it to GMS dataset

'''

# cd c:\Users\hpham\Documents\P32_Niger_2019\NWbud\scripts\

# Define some input parameters
nrows, ncols, nlays = 673, 771, 8
ncells = nrows*ncols*nlays
dataset_name = 'bot_m'  # 'top_m' or 'bot_m'


# Open RTI elevation files
if dataset_name == 'top_m':
    geo_units = ['MNT_Skua_2m_grd_zm',
                 'Cont_Term_Map_XS_2m_grd_zm',
                 'Paleo_Map_XS_2m_grd_zm',
                 'Cret_Sup_Map_XS_2m_grd_zm',
                 'Cont_Ham_Map_XS_2m_grd_zm',
                 'Cont_Int_Map_XS_2m_grd_zm',
                 'Juras_Map_XS_2m_grd_zm',
                 'Trias_Map_XS_2m_grd_zm']  # 'Socle_Skua_Map_XS_2m_grd_zm'
elif dataset_name == 'bot_m':
    geo_units = ['Cont_Term_Map_XS_2m_grd_zm',
                 'Paleo_Map_XS_2m_grd_zm',
                 'Cret_Sup_Map_XS_2m_grd_zm',
                 'Cont_Ham_Map_XS_2m_grd_zm',
                 'Cont_Int_Map_XS_2m_grd_zm',
                 'Juras_Map_XS_2m_grd_zm',
                 'Trias_Map_XS_2m_grd_zm',
                 'Socle_Skua_Map_XS_2m_grd_zm']

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
    geo_name = geon
    ifile = '../input/RTI_layer_ele/' + geo_name + '.txt'
    data = np.loadtxt(ifile, delimiter=',')
    if i == 0:  # Process 'MNT_Skua_2m_grd_zm' because it has one more row/col
        data2 = data[data[:, 0] <= 827871.400]
        data3 = data2[data2[:, 1] <= 2493481.000]
        z = np.reshape(data3[:, 2], [ncols, nrows])
        print(f'i={i}, size={data3.shape}\n')
    else:
        z = np.reshape(data[:, 2], [ncols, nrows])

    #
    # Continue processing data ...
    z = np.transpose(z)
    z2 = np.reshape(z, [nrows*ncols, 1])
    #
    count_start = i*nrows*ncols
    count_stop = (i+1)*nrows*ncols
    ele[count_start:count_stop] = z2
    #cell_flag[count_start:count_stop, 1] = z2

    # z = np.flipud(z)

    #grid_plot(geo_name, i, x, y, z, cmap, norm, levels, fsize, odir)

    # fid.write('\n')
# Assign activate/inactivate cells
cell_flag = np.copy(ele)
cell_flag[cell_flag == -99999] = 0
cell_flag[cell_flag != -99999] = 1

#
np.savetxt(fid, cell_flag, fmt='%d')
ele[ele == -99999] = -999
np.savetxt(fid, ele, fmt='%7.3f')

fid.write('ENDDS\n')
fid.close()

print(f'Saved output file at {ofile}\n')
