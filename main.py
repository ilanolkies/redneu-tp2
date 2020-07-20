import sys
sys.path.insert(1, './src')
from os import path
from dataset import load_dataset
from hebbiano import Hebbiano
from plotter import plot

nombre_modelo = sys.argv[1]
dataPath = sys.argv[2]

if not path.exists(nombre_modelo + '.p'):
  data, labels = load_dataset(dataPath)
  modelo = Hebbiano(data)

  alg = sys.argv[3]
  if alg == 'oja':
    modelo.train('oja', 9, 0.001, 0.001, 600, 100) # TBD
  elif alg == 'sanger':
    modelo.train('oja', 9, 0.001, 0.001, 600, 100) # TBD

  modelo.save(nombre_modelo + '.p')
else:
  modelo = Hebbiano()
  modelo.load(nombre_modelo + '.p')

  data, labels = load_dataset(dataPath)

  y = modelo.test(data)

  plot(y, labels)