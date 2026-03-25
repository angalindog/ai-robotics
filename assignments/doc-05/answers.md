# Respuestas

## Respuesta 1

Partimos del código original en un [Notebook de python](/assignments/doc-05/Clase_NNA3.ipynb), para generar un [archivo python](/assignments/doc-05/1_Clase_NNA3.py) donde podremos depurar y trabajar en la solución del problema

Generamos un nuevo arreglo `X` para los datos de entrenamiento:

```python
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
```

### Compuerta NAND

| X1  | X2  | Y   |
| --- | --- | --- |
| 0   | 0   | 1   |
| 0   | 1   | 1   |
| 1   | 0   | 1   |
| 1   | 1   | 0   |

La salida deseada para la compuerta NAND es:

```python
Yr = np.array([[1, 1, 1, 0]])
```

### Compuerta XOR

| X1  | X2  | Y   |
| --- | --- | --- |
| 0   | 0   | 0   |
| 0   | 1   | 1   |
| 1   | 0   | 1   |
| 1   | 1   | 0   |

La salida deseada para la compuerta XOR es:

```python
Yr = np.array([[0, 1, 1, 0]])
```

### Solución

Para ambas compuertas se debe hacer la misma modificación en el código de entrenamiento para ajustarse al tamaño de la matriz de entrada.

```python
while i < 50000:
    for j in range(4):

        Xo = X[:,j].reshape(2,1)
        Yd = Yr[:,j].reshape(1,1)

        A2, Z2, A1, Z1 = propaga(Xo)

        dEdW2, dEdb2, dEdW1, dEdb1 = backpropagation(Xo, Yd, A2, Z2, A1, Z1)

        W2 = W2 - eta * dEdW2
        b2 = b2 - eta * dEdb2
        W1 = W1 - eta * dEdW1
        b1 = b1 - eta * dEdb1
```

- Compuerta NAND:
    ```
    [0 0] [[0.99963465]]
    [0 1] [[0.99025791]]
    [1 0] [[0.99025683]]
    [1 1] [[0.01753176]]
    ```

- Compuerta XOR
    ```
    [0 0] [[0.02040771]]
    [0 1] [[0.98233969]]
    [1 0] [[0.98234536]]
    [1 1] [[0.01831342]]
    ```

## Respuesta 2

En el dataset MNIST se tienen 10 clases distintas de ropa:

1. Abrigo
2. Pullover
3. Camiseta
4. Camisa
5. Pantalón
6. Vestido
7. Sandalia
8. Zapatilla
9.  Bota
10. Bolso

El código se estructura en su respectivo [archivo python](/assignments/doc-05/2_Fashion.py), y se documenta en un [Notebook propio](/assignments/doc-05/fashion.ipynb)

Se obtiene `Precision: 0.8780999779701233` de los códigos realizados, es decir una precisión del $87.8%$. Y como resultado de la comparación de los valores obtenidos:

```
Prediccion: 9
Real: 9
```

Con la normalización de los datos resulta un mejor estandar para el modelo desarrollado, a pesar de las similitudes que tienen algunas de las categorías, las cuales pueden ser las mayores causantes de reducción de la precisión del modelo.

## Respuesta 3

<!-- Consiga un data set de cualquier tipo, puede ser de [Kaggle](https://www.kaggle.com/datasets), estudie sus características (features), y su rótulo. Diseñe una red neuronal y haga ejemplos con base en los pesos aprendidos. -->

**Dataset seleccionado:** Grossing movies worldwide from Box Office Mojo - Recaudación de peliculas a nivel mundial por Box Office Mojo

- [Enlace del dataset](https://www.kaggle.com/datasets/muhammadaqeelkabir/grossing-movies-worldwide-from-box-office-mojo/data)
- [Archivo del dataset](/assignments/doc-05/box_office_dataset.csv)
- Dataset con API

```py
import kagglehub

# Download latest version
path = kagglehub.dataset_download("muhammadaqeelkabir/grossing-movies-worldwide-from-box-office-mojo")

print("Path to dataset files:", path)
```

### Features / Características (X)

Elementos de alimentación o aprendizaje para la red neuronal.

- Domestic lifetime gross
- Domestic percentage
- Foreign lifetime gross
- Foreign percentage
- Worldwide lifetime gross
- Year

### Rótulo (Y)

Elemento a ser predecido por la red neuronal, basado en las características.

- Movie success / Éxito de la película
  - Dometic success
  - Foreign success

Inicialmente, con base en la recaudación a nivel mundial, se puede establecer una cantidad específica, la cuál indique que si se supera dicha cantidad, la película es un éxito, y se puede llegar a profundizar en el análisis, donde con base en los porcentajes locales, y extranjeros, se podría determinar si el éxito fue mayor a nivel local, extranjero, o mundial.

El archivo python de la red neuronal, se encuentra en [redNeuronal.py](/assignments/doc-05/redNeuronal.py)