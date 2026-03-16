# Respuestas

## Respuesta 1

Partimos del código original en un [Notebook de python](/assignments/doc-05/Clase_NNA3.ipynb), para generar un [archivo python](/assignments/doc-05/Clase_NNA3.py) donde podremos depurar y trabajar en la solución del problema

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