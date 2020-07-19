from dataset import load_dataset, normalize
import numpy as np
from matplotlib import pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D

def ortogonalidad(W, M):
    return np.sum(np.abs(np.dot(W.T,W) - np.identity(M)))/2

def regla_oja(X, W, h, lr):
    y = np.dot(X[h], W)
    z = np.dot(y, W.T)
    return lr * np.outer(X[h] - z, y)

def regla_sanger(X, W, M, h, lr):
    y = np.dot(X[h], W)
    d = np.triu(np.ones((M, M)))
    z = np.dot( W, np.array([y]).T*d)
    return lr * (np.array([X[h]]).T - z) * y


class Hebbiano:
    def __init__(self, dataset_dir):
        data, self.labels = load_dataset(dataset_dir)
        X = normalize(data)

        self.X = X
        self.P, self.N = X.shape

    def train(self, alg, M, lr, min_ort, max_epoch, trace = 0):
        W = np.random.normal( 0, 0.1, (self.N, M))

        t = 0
        o = ortogonalidad(W, M)
        while(o > min_ort and t <= max_epoch):
            t+=1
            o = ortogonalidad(W, M)
            if (trace != 0 and t % trace == 0): print(o)

            #sanger lr adaptative.
            if(alg == 'sanger'):
                if(o < 4):
                    if t % 30 == 0:
                        lr=0.001/t

            for h in range(self.P):
                if alg == 'oja':
                    W += regla_oja(self.X, W, h, lr)
                elif alg == 'sanger':
                    W += regla_sanger(self.X, W, M, h, lr)

        self.W = W

    def plot(self):
        #Activacion
        y = np.dot(self.X, self.W)
        #Agrego la categoria a las instancias
        y = np.append(y, self.labels, axis=1)

        #Grafico
        fig = mpl.figure()
        xyz = fig.add_subplot(131, projection='3d')
        xyz.scatter3D(y[:,0], y[:,1], y[:,2], 'b.', c=y[:,9], s=7)

        xyz2 = fig.add_subplot(132, projection='3d')
        xyz2.scatter3D(y[:,3], y[:,4], y[:,5], 'b.', c=y[:,9], s=7)

        xyz3 = fig.add_subplot(133, projection='3d')
        xyz3.scatter3D(y[:,6], y[:,7], y[:,8], 'b.', c=y[:,9], s=7)
        mpl.show()
