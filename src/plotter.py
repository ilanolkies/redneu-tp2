import numpy as np
from matplotlib import pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D

def plot(y, labels, s = 5):
    c = labels[:,0]

    fig = mpl.figure()

    xyz = fig.add_subplot(131, projection='3d')
    xyz.scatter3D(y[:,0], y[:,1], y[:,2], 'b.', c=c, s=s)

    xyz2 = fig.add_subplot(132, projection='3d')
    xyz2.scatter3D(y[:,3], y[:,4], y[:,5], 'b.', c=c, s=s)

    xyz3 = fig.add_subplot(133, projection='3d')
    xyz3.scatter3D(y[:,6], y[:,7], y[:,8], 'b.', c=c, s=s)

    mpl.show()
