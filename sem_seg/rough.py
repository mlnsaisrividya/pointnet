import numpy as np
import open3d
from matplotlib import pyplot as plt
import mayavi
import h5py
def load_h5(h5_filename):
    f = h5py.File(h5_filename,"r")
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


h5_filename = 'indoor3d_sem_seg_hdf5_data/ply_data_all_0.h5'
data_batch, label_batch = load_h5(h5_filename)
dps = data_batch[0:80,:,0:6] # area1 conference1
lbs = label_batch[0:80,:]
#dps = data_batch[606:661,:,0:6]
#lbs = label_batch[606:661,:]
print(dps.shape)
print(lbs.shape)

#all points in one place for visualisation purpose 
new = dps.reshape((dps.shape[0]*dps.shape[1]),dps.shape[2])
print(new.shape)
# row manipulation
#number_of_rows = new.shape[0]
#random_indices = np.random.choice(number_of_rows, 
#                                  size=11250, 
#                                  replace=False) #taking exactly 5% of points
  
# display random rows
#print("\nRandom row:")
#ne = new[random_indices, :]
#print(ne.shape)
#ne = np.random.Generator.choice(new, axis = 0)
#ne = new.select('RandomSampling', 5);
# new = dps[0,:,:]
# for i in range(1,3):
#     np.concatenate((new, dps[i,:,:]), axis=1)
# print(new.shape)
# for i in range(4):
#open3d.visualization.draw_geometries([newarr])


#%matplotlib notebook
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
#scatter = ax.scatter(kset[50,:,0] , kset[50,:,1] , kset[50,:,2])
#for i in range(80):
scatter = ax.scatter(new[:, 0], new[:, 1], new[:, 2] ,c= new[:,3:6])
#scatter = ax.scatter(new[25000:26000, 0], new[25000:26000, 1], new[25000:26000, 2], c= new[25000:26000,3:6])
plt.show()
