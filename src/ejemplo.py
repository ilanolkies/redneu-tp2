from dataset import load_dataset
from hebbiano import Hebbiano
from plotter import plot, plot_error
from matplotlib import pyplot as plt

data, labels = load_dataset('./dataset/tp2_training_dataset.csv')

data_train = data[:700]
labels_train = labels[:700]

data_test = data[700:]
labels_test = labels[700:]

modelo = Hebbiano(data_train)

errors = modelo.train('oja', 9, 0.00001, 0.001, 1500, 100)

plot_error(errors)
plot(modelo.test(data_train), labels_train)
plot(modelo.test(data_test), labels_test)
