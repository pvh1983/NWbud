import numpy as np
import matplotlib.pyplot as plt
import os


# Run this code to get the coordinates of all grid cell centers
# cd c:\Users\hpham\Documents\P32_Niger_2019\NWbud\scripts\

cell_size = 2000.  # meters

Top = 2216000-cell_size/2.
Bot = 1142000+cell_size/2.
Left = -550000+cell_size/2.
Right = 788000-cell_size/2.

nx = int((Right-Left)/cell_size)
ny = int((Top-Bot)/cell_size)
nlay = 4

print(f'nx={nx}, ny={ny}\n')

x = np.linspace(Left, Right, nx+1)
y = np.linspace(Top, Bot, ny+1)
ncells_each_lay = (nx+1)*(ny+1)
ncells = (nx+1)*(ny+1)*nlay

print(f'size x,y (col/row) ={x.shape, y.shape} \n')
x.shape
y.shape

xv, yv = np.meshgrid(x, y)

print(f'size xv,yv ={xv.shape, yv.shape} \n')

x1 = np.reshape(xv, ((nx+1)*(ny+1)))
y1 = np.reshape(yv, ((nx+1)*(ny+1)))

# out = np.concatenate((x1, y1), axis=1)


# Get cell ijk
cid = np.empty((ncells_each_lay, 2))
ct = 0
for i in range(1, ny+2, 1):
    for j in range(1, nx+2, 1):
        cid[ct, 0] = j  # col (x)
        cid[ct, 1] = i  # row (y)
        ct += 1
out = np.vstack((y1, x1, cid[:, 1], cid[:, 0]))
out2 = np.transpose(out)
print(f'Number of rows of x and y : {out2.shape}\n')
np.savetxt('../output/grid_coor.csv', out2, fmt='%9.3f', delimiter=',')
# plt.plot(x1, y1)
# plt.show()

# Write top/bot elevations to file
# ===========================================================================


def print_ofile(ifile, ncells, ncells_each_lay, nlay, odir, x1, y1, cid):
    ipath = '../input/GMS_datasets/exported_from_GMS/' + ifile
    nrows_skip = ncells + 7
    data = np.loadtxt(ipath, skiprows=nrows_skip)
    print(f'\nReading file {ipath}\n')
    print(data)
    print(f'Skipped {nrows_skip} rows\n')
    print(f'Number of rows READ: {data.shape}\n')

    data_each_lay = np.reshape(data, (nlay, ncells_each_lay))
    data_each_lay = np.transpose(data_each_lay)
    #print(f'Dimension of data_each_lay: {data_each_lay.shape}\n')
    #np.savetxt('test.csv',  data_each_lay[:, 0], fmt='%9.3f', delimiter=',')

    for i in range(nlay):
        ofile = odir + ifile[:3] + '_ele_' + '_layer_' + str(i+1) + '.csv'
        fid = open(ofile, 'w')
        fid.write('Lat_m,Long_m, i, j, Ele_m\n')
        v1 = data_each_lay[:, i]
        # print(v1[109945-1])
        # print(v1)
        out = np.vstack((y1, x1, cid[:, 1], cid[:, 0], v1))
        out_xyz = np.transpose(out)
        np.savetxt(fid, out_xyz, fmt='%9.3f', delimiter=',')
        fid.close()
        print(f'Saved {ofile}\n')


#
# Creat a new folder to save figures
odir = '../output/layer_ele_from_GMS/'
if not os.path.exists(odir):  # Make a new directory if not exist
    os.makedirs(odir)
    print(f'\nCreated directory {odir}\n')

# NOTES: Manually open dataset files and delete the final row ENDDS
ifile = ['top_final.dat', 'bot_final.dat']  #
for f in ifile:
    print_ofile(f, ncells, ncells_each_lay, nlay, odir, x1, y1, cid)

print('\nDone all!')
# plt.plot(x1, y1)
# plt.show()


# out = np.concatenate((x1, y1), axis=1)
