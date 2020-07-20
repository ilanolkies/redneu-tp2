# Redes neuronales artificiales - Trabajo práctico 2

Implementación de redes de parendizaje no supervisado


## Aprendizaje Hebbiano no supervisado

Mediante las reglas de aprendizaje de **Oja** y de **Sanger**

<!-- ## Aprendizaje Competitivo -->

## Modelos entrenados

Modelo entrenado con Oja

```
python main.py oja_model PATH_DATASET
```

Modelo entrenado con Sanger

```
python main.py sanger_model PATH_DATASET
```

## Entrenar modelo

```
python main.py NOMBRE_MODELO PATH_DATASET ALG
```

Entrenar con Oja:

```
python main.py NOMBRE_MODELO PATH_DATASET oja
```

Entrenar con Sanger:

```
python main.py NOMBRE_MODELO PATH_DATASET sanger
```

## Implementación

La implementación de los algoritmos esta en `src/`

El modelo Hebbiano se puede usar con la clase `Hebbiano` exportada en `hebbiano.py`

```python
from hebbiano import Hebbiano

hebbiano = Hebbiano('./dataset/tp2_training_dataset.csv')

# alg, M, lr, min_ort, max_epoch, trace (= 0)
hebbiano.train('oja', 9, 0.001, 0.00001, 600, 1)
hebbiano.plot()
```

## Experimentación

Los experimentos se pueden ejecutar con

```
jupyeter lab
```

Los gráficos del informe fueron generados con ejemplo_oja.py y ejemplo_sanger.py

## Setup

Usar Python: v3.6.5

Para instalar version de Python, la primera vez:

1. Instalar `pyenv` - https://github.com/pyenv/pyenv#installation

  En mac:

  ```
  brew update
  brew install pyenv
  ```

2. Inicializar `pyenv`

  ```
  eval "$(pyenv init -)"
  ```

3. Instalar v3.6.5

  ```
  pyenv install 3.6.5
  ```

4. Activar v3.6.5 en terminal

  ```
  pyenv global 3.6.5
  ```

5. Verificar version

  ```
  python --version
  > Python 3.6.5
  ```

6. Instalar dependencias

  ```
  pip install -r requirements.txt
  ```

Para volver a activarlo en una nueva terminal - setup de primera vez ya terminado

```
eval "$(pyenv init -)"
pyenv global 3.6.5
```
