import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(y, labels, s = 5):
    fig = plt.figure()

    xyz = fig.add_subplot(131, projection='3d')
    xyz.scatter3D(y[:,0], y[:,1], y[:,2], 'b.', c=labels[:,0], s=s)
    xyz.set_xlabel('$Y_1$')
    xyz.set_ylabel('$Y_2$')
    xyz.set_zlabel('$Y_3$')

    xyz2 = fig.add_subplot(132, projection='3d')
    xyz2.scatter3D(y[:,3], y[:,4], y[:,5], 'b.', c=labels[:,0], s=s)
    xyz2.set_xlabel('$Y_4$')
    xyz2.set_ylabel('$Y_5$')
    xyz2.set_zlabel('$Y_6$')

    xyz3 = fig.add_subplot(133, projection='3d')
    xyz3.scatter3D(y[:,6], y[:,7], y[:,8], 'b.', c=labels[:,0], s=s)
    xyz3.set_xlabel('$Y_7$')
    xyz3.set_ylabel('$Y_8$')
    xyz3.set_zlabel('$Y_9$')

    plt.show()

def plot_error(errors):
    plt.title('Errores')
    plt.xlabel('Epochs')
    plt.ylabel('Ortogonalidad')
    plt.plot(errors)
    plt.show()
