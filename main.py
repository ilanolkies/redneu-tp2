import sys
sys.path.insert(1, './src')
from os import path
from dataset import load_dataset
from hebbiano import Hebbiano
from plotter import plot, plot_error

nombre_modelo = sys.argv[1]
dataPath = sys.argv[2]

if not path.exists(nombre_modelo + '.p'):
  data, labels = load_dataset(dataPath)
  modelo = Hebbiano(data)

  alg = sys.argv[3]
  if alg == 'oja':
    errors = modelo.train('oja', 9, 0.00001, 0.0001, 1500, 100) 
  elif alg == 'sanger':
    errors = modelo.train('sanger', 9, 0.00001, 0.001, 2000, 100) #0.04134266836522473

  modelo.save(nombre_modelo + '.p')
  
  plot_error(errors)
  plot(modelo.test(data), labels)
  
else:
  modelo = Hebbiano()
  modelo.load(nombre_modelo + '.p')

  data, labels = load_dataset(dataPath)

  y = modelo.test(data)

  plot(y, labels)
