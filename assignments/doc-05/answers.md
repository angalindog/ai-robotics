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
9. Bota
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

# Respuestas V2

## Marco teórico y conceptos fundamentales

### Redes Neuronales Artificiales (RNA)

> **Curso:** Informática Evolutiva / Machine Learning  
> **Autor base:** José J. Martínez P.  
> **Elaborado con:** Ejemplos, analogías y ejercicios para Google Colab

---

## Tabla de Contenidos

1. [¿Qué es una RNA?](#1-qué-es-una-rna)
2. [Neurona Biológica vs Neurona Artificial](#2-neurona-biológica-vs-neurona-artificial)
3. [Funciones de Activación](#3-funciones-de-activación)
4. [Ejemplos de Clasificación con una Neurona](#4-ejemplos-de-clasificación-con-una-neurona)
5. [Redes Multicapa y Propagación Hacia Adelante](#5-redes-multicapa-y-propagación-hacia-adelante)
6. [Aprendizaje Supervisado](#6-aprendizaje-supervisado)
7. [Backpropagation (Retropropagación)](#7-backpropagation-retropropagación)
8. [Deep Learning](#8-deep-learning)
9. [Redes Convolucionales y Pooling](#9-redes-convolucionales-y-pooling)
10. [Ejercicio 1: NAND y XOR con 2 capas ocultas](#10-ejercicio-1-nand-y-xor-con-2-capas-ocultas)
11. [Ejercicio 2: Fashion MNIST con TensorFlow](#11-ejercicio-2-fashion-mnist-con-tensorflow)

---

## 1. ¿Qué es una RNA?

Una **Red Neuronal Artificial (RNA)** es un modelo computacional inspirado en el cerebro biológico. Su objetivo es aprender patrones a partir de datos, igual que los animales aprenden a reconocer presas o predadores a partir de la experiencia. Es como los pajaros en la naturaleza, filtran las iamgenes de todo lo que les rodea y reconoce colores, olores, formas, etc, y a partir de alli decide si es comestible o no, en caso de que se equivoque, puede reajustar su modelo mental.
Los RNA funcionan de manera similar: Filtrar datos --> encuentra patrones --> decide --> aprende del error. Pero el modelo de RNA mas popular y de mayor exito que ha dado fruto en la actualidad de la IA, es el Transformer, cuyo modelo es empleado por los grandes modelos de lenguaje generativo como ChatGPT, Claude, Gemini, etc.

## 2. Neurona Biológica vs Neurona Artificial

Es muy similar y aproximados los dos modelos como se vera en la siguiente descripcion: Dendritas (entradas) -> Soma (núcleo, suma señales) -> Axón(Salida) (Soma)<- Sinapsis (pesos de conexión).
Para que exista una señal clara la suma de señales debe superar un umbral.

### Neurona Artificial (Perceptrón)

**Analogía:**

- Dendritas → entradas `x₁, x₂, ..., xₙ`
- Sinapsis → pesos `w₁, w₂, ..., wₙ` (cuánto importa cada entrada)
- Umbral → sesgo `b` (bias)
- Axón → salida `z` → función de activación → `g(z)`

## 3. Funciones de Activación

La función de activación g(z) es la que introduce la no linealidad a la red, si no estuviera toda la red seria una combinación lineal sin importar el numero de capas con las que cuente. Es como el umbral neuronal en una neurona natural, ya que decide si la neurona se activa o dispara y con que intensidad lo hace.

### Función de paso unitario (Heaviside)

```
g(z) = 1  si z ≥ 0
g(z) = 0  si z < 0
```

### Función Sigmoide (Logística)

```
g(z) = 1 / (1 + e^(-z))
```

- Salida siempre entre **0 y 1**
- Derivada: `g'(z) = g(z) · (1 - g(z))` ← muy útil para Backprop
- Interpreta la salida como una **probabilidad**

### Función ReLU (Rectified Linear Unit)

```
g(z) = max(0, z)
```

- **Más eficiente** para redes profundas (no sufre vanishing gradient)
- Derivada simple: 1 si z>0, 0 si z≤0

## 4. Ejemplos de Clasificación con una Neurona

Una sola neurona con función sigmoide puede clasificar funciones lógicas **linealmente separables**.

### Compuerta AND

```
Pesos: b = -30, w₁ = 20, w₂ = 20

x₁  x₂  z = -30 + 20x₁ + 20x₂   g(z)
 0   0        -30                  ≈ 0
 0   1        -10                  ≈ 0
 1   0        -10                  ≈ 0
 1   1         10                  ≈ 1  ✅ Solo 1 AND 1 = 1
```

### Compuerta OR

```
Pesos: b = -10, w₁ = 20, w₂ = 20

x₁  x₂  z = -10 + 20x₁ + 20x₂   g(z)
 0   0        -10                  ≈ 0
 0   1         10                  ≈ 1
 1   0         10                  ≈ 1
 1   1         30                  ≈ 1  ✅
```

### Compuerta NOT

```
Pesos: b = 10, w₁ = -20

x₁  z = 10 - 20x₁   g(z)
 0       10           ≈ 1  ✅
 1      -10           ≈ 0  ✅
```

**Conclusión clave:** Solo cambiando los pesos `W`, la misma arquitectura puede aprender funciones completamente diferentes.

### ¿Por qué XOR necesita más de una capa?

```
XOR no es linealmente separable:
  (0,0)→0   (0,1)→1   (1,0)→1   (1,1)→0

No se puede trazar UNA sola línea recta que separe los 1s de los 0s.
Se necesitan AL MENOS 2 capas ocultas.
```

## 5. Redes Multicapa y Propagación Hacia Adelante

Una red nauronal multicapa es un conjunto de "neuronas" organizadas en niveles, donde cada una de las neuronas actua como una pequeña calculadora, que recibe información, la procesa y la pasa a la siguiente capa.
Se divide en 3 partes, la primera es una capa de entrada que recibe los datos, como color, peso, tamaño, etc; las siguientes son capas ocultas, dichas capas intermedias es donde la red reconoce patrones complejos que no son obvios a simple vista; y por ultimo se encuentra la capa de salida que entrega la respuesta.

### Propagación hacia adelante (Forward Porpagation)

Se puede entender como una cascada de decisiones donde la información viaja desde la entrada hacia la salida para generar la predicción. Es facil entender con un ejemplo, imagina que quieres comprar un carros y tienes 3 amigos (capas) que te aconsejaran en un punto en particular, el Amigo 1 te aconseja sobre el color, el Amigo 2, te aconseja sobre el tipo de combustible, y el Amigo 3 sobre la cantidad de puestos, asi que tu le crees a todos por igual pero le dan importancia a cada opinion dependiendo de tus intereses, si te preocupa mas el color, le dara un mayor peso a esa opinion, y si le da igual el tipo de combustible le daras un valor muy bajo a ese componente. Por lo que forward propagation multiplica cada opcion por su importancia y se suma; un detalle importante es que en este modelo no existe la retroalimentación.

Para que una neurona funcione, realiza tres operaciones matemáticas simples:

#### A. Suma Ponderada ($z$)

Cada entrada ($x$) se multiplica por un peso ($w$) y se le suma un sesgo ($b$, que es como una "preferencia" inicial).

$$z = (x_1 \cdot w_1) + (x_2 \cdot w_2) + \dots + b$$

#### B. Función de Activación ($a$)

El resultado $z$ pasa por un filtro llamado **función de activación** (como la función Sigmoide o ReLU). Esto sirve para decidir si la neurona se "dispara" o no. Es lo que permite que la red aprenda patrones complejos que no son simples líneas rectas.

$$a = \sigma(z)$$

#### C. El proceso en la red completa

Si tenemos una capa con varias neuronas, usamos **matrices** para calcular todo el nivel de la red de un solo golpe:

$$A = \sigma(W \cdot X + B)$$

- **$W$**: Matriz de todos los pesos de las conexiones.
- **$X$**: Vector de las entradas (inputs).
- **$B$**: Vector de los sesgos (bias).

## 6. Aprendizaje Supervisado

El aprendizaje supervisado es como aprender a conducir con un experto, cada vez que el alumno se equivoca el instructor lo corrige hasta que el alumno pueda conducir solo.
La red recibe X= Caracteristicas (features) de entrada, y Y = rótulos correctos (labels)
Por lo que la red aprende ajustando los pesos W, para minimizar el error entre su salida ÿ y los rotulos Y, es como si se aprendiera con una examen que tiene las respuestas, se resuelve el ejercicio (forward pass), se compara la respuesta que dio con lo que hay en el libro o examen (cálculo del error), luego se identifica en que paso se equivoco (backpropagation), se corrige el método (actualización de pesos), y se repiten los ejercicios n-esimas veces (momentos de entrenamiento).

El aprendizaje supervisado consiste en entrenar un modelo mediante pares de **entrada ($X$)** y **salida correcta ($Y$)**.

### División del Dataset

- **Entrenamiento (80%):** Ajuste de pesos $W$.
- **Validación (10%):** Monitoreo durante el entrenamiento.
- **Prueba (10%):** Evaluación final con datos desconocidos.

### Matemáticas del Error

La **Función de Costo ($J(W)$)** mide la precisión global del modelo:

$$J(W) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^i - Y^i)^2$$

El objetivo es minimizar esta función mediante algoritmos como el **Descenso del Gradiente**.

## 7. Backpropagation (Retropropagación)

Este modelo parte de la regla de la cadena del calculo diferencial, ya que solo se conoce el error de la capa de salida, la retropropagación, propaga ese error hacia atras, capa por capa.

El error de las redes neuronales es que se sabe que su salida o respuesta es incorrecta, pero se requiere saber en que capa estuvo la culpa, o si se le dio el peso inadecuado a cierta capa.

### Derivaciones por Capa

**Definición del delta (señal de error):**

```
δ³ = diag(σ'(Z³)) · e            ← error de salida × derivada de activación

δ² = diag(σ'(Z²)) · W³ᵀ · δ³   ← propaga δ³ hacia atrás

δ¹ = diag(σ'(Z¹)) · W²ᵀ · δ²   ← propaga δ² hacia atrás
```

**Gradientes de los pesos:**

```
∂E/∂B^k = -δ^k
∂E/∂W^k = -δ^k · A^(k-1)ᵀ
```

**Actualización de pesos con descenso de gradiente:**

```
ΔW^(k) = -η·δ^k - η·δ^k·A^(k-1)ᵀ
W^(k)  = W^(k) + ΔW^(k)
```

Donde `η` (eta) es la **tasa de aprendizaje** (learning rate).

### La Tasa de Aprendizaje `η`

```
η muy grande:  los pesos saltan demasiado, nunca converge
η muy pequeña: aprende muy lento, puede quedar atrapado
η típico:      0.001 a 0.1
```

### Generalización para cualquier capa interna `k`

```
δ^k = diag(σ'(Z^k)) · W^(k+1)ᵀ · δ^(k+1)
ΔW^(k+1) = -η·δ^k - η·δ^k·A^(k-1)ᵀ
W^(k) = W^(k) + ΔW^(k+1)
```

## 8. Deep Learning

A diferencia de las redes neuronales simples o tradicionales que pueden contener entre 3 y 5 capas, en deep learning se puede llegar a tener 10, 50 o 100 capas ocultas, ya que cada capa actua como un filtro extra de limpieza y se puede tener un resultado o respuesta mas precisa debido al nivel de complejidad que adquiere.

En estos modelos existe un concepto importante llamado Transfer Learning que es para evitar reinventar la rueda, si una red de 7 capas ya seabe identificar objetos, es porque las capas anteriores ya han aprendido a idenficiar bordes, colores, etc, por ejemplo si se sitene una que identifica perros y gatos, simplemente se cambia la ultima capa y de las otras 6 se les congela los pesos, y la ultima capa se reemplaza segun el objetivo del estudio, por lo que se requiere menos datos.

## 9. Redes Convolucionales y Pooling

Tienen un objetivo especial y son para procesar datos que tienen una estrcutura de rejilla como las imagenes. Si una imagen de 520x520 tiene mas de 260.000 pixeles una sola neurona tendria que manejar un gran volumen computacional, por lo que las CNN solucionan esto mediante la invariancia espacial, pues no importa donde se ubiquen las caratericias de los objetos dentro de una imagen los rasgos siempren seran los mismos.

Las CNN están optimizadas para procesar imágenes mediante el uso de filtros locales en lugar de conexiones globales.

### Operaciones Principales

1. **Convolución:** Aplicación de filtros (kernels) para extraer mapas de características (bordes, texturas, formas).
2. **Pooling (Max Pooling):** Reducción de la dimensionalidad espacial. Ayuda a que el modelo sea robusto ante pequeñas traslaciones de la imagen.

### Estructura de Salida

Al final de las capas convolucionales, se utiliza una operación de **Flatten** para conectar con capas densas (**Fully Connected**) que realizan la clasificación final.

$$Dimension_{out} = (n - k + 1) \times (n - k + 1)$$

# Respuesta #1

# Respuesta #2

# Respuesta #3
