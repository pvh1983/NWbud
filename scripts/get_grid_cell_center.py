import numpy as np
import matplotlib.pyplot as plt

cell_size = 2000.  # meters

Top = 2216000-cell_size/2.
Bot = 1142000+cell_size/2.
Left = -550000+cell_size/2.
Right = 788000-cell_size/2.

nx = int((Right-Left)/cell_size)
ny = int((Top-Bot)/cell_size)
print(f'nx={nx}, ny={ny}\n')

x = np.linspace(Left, Right, nx+1)
y = np.linspace(Top, Bot, ny+1)
ncells = (nx+1)*(ny+1)

print(f'size x,y ={x.shape, y.shape} \n')
x.shape
y.shape

xv, yv = np.meshgrid(x, y)

print(f'size xv,yv ={xv.shape, yv.shape} \n')

x1 = np.reshape(xv, ((nx+1)*(ny+1), 1))
y1 = np.reshape(yv, ((nx+1)*(ny+1), 1))

out = np.concatenate((x1, y1), axis=1)

np.savetxt('../output/grid_coor.csv', out, delimiter=',')
#plt.plot(x1, y1)
# plt.show()
