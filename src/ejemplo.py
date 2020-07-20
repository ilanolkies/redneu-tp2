from dataset import load_dataset
from hebbiano import Hebbiano
from plotter import plot
from matplotlib import pyplot as plt

data, labels = load_dataset('./dataset/tp2_training_dataset.csv')

data_train = data[:700]
labels_train = labels[:700]

data_test = data[700:]
labels_test = labels[700:]

modelo = Hebbiano(data_train)

errors = modelo.train('oja', 9, 0.001, 0.001, 50, 10)

plt.plot(errors)
plt.show()

y = modelo.test(data_test)

plot(y, labels_test)
