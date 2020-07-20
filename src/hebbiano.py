import numpy as np
import pickle

def normalizar(X):
    return (X - X.mean(axis=0)) / X.std()

def initW(N, M):
    return np.random.normal(0, 0.1, (N, M))

def ortogonalidad(W, M):
    return np.sum(np.abs(np.dot(W.T, W) - np.identity(M))) / 2

def adapt_lr(o, t, lr, r):
    return lr/10 if o < r[0] and o > r[1] and t % 100 == 0 else lr
    #return 0.001 / t if o < 4 and t % 30 == 0 else lr

def regla_oja(X, W, h, lr):
    y = np.dot(X[h], W)
    z = np.dot(y, W.T)
    return lr * np.outer(X[h] - z, y)

def regla_sanger(X, W, M, h, lr):
    y = np.dot(X[h], W)
    d = np.triu(np.ones((M, M)))
    z = np.dot( W, np.array([y]).T*d)
    return lr * (np.array([X[h]]).T - z) * y

def activacion(X, W):
    return np.dot(X, W)

class Hebbiano:
    def __init__(self, X = np.zeros((1, 1))):
        if X.shape != (1, 1):
            self.X = normalizar(X)
            self.P, self.N = X.shape

    def train(self, alg, M, lr, min_ort, max_epoch, trace = 0):
        W = initW(self.N, M)

        O = []

        t = 0
        o = ortogonalidad(W, M)
        while(o > min_ort and t <= max_epoch):
            t+=1
            o = ortogonalidad(W, M)
            O += [o]
            if (trace != 0 and t % trace == 0): print(o)

            # adaptacion de lr para sanger
            if alg == 'oja':
                lr = adapt_lr(o, t, lr, (0.017, 0.008))
            elif alg == 'sanger':
                lr = adapt_lr(o, t, lr, (0.038, 0.0379))

            for h in range(self.P):
                if alg == 'oja':
                    W += regla_oja(self.X, W, h, lr)
                elif alg == 'sanger':
                    W += regla_sanger(self.X, W, M, h, lr)

        self.W = W
        return O

    def test(self, X):
        return activacion(X, self.W)

    def save(self, path):
        pickle.dump(self.W, open(path, 'wb'))

    def load(self, path):
        self.W = pickle.load(open(path, 'rb'))
