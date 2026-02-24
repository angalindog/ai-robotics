## Ejercicios y Problemas

> **1. Maximizar la función $f(x) = x \operatorname{sen}(10 \pi x) + 1$, con $x \in [0,1]$.**

2. Suponga que usted es el jefe de gobierno y está interesado en que pasen los proyectos de su programa político. Sin embargo, en el congreso conformado por 5 partidos, no es fácil su tránsito, por lo que debe repartir el poder, conformado por ministerios y otras agencias del gobierno, con base en la representación de cada partido. Cada entidad estatal tiene un peso de poder, que es el que se debe distribuir. Suponga que hay 50 curules, distribuya aleatoriamente, con una distribución no uniforme entre los 5 partidos esas curules. Defina una lista de 50 entidades y asígneles aleatoriamente un peso político de 1 a 100 puntos. Cree una matriz de poder para repartir ese poder, usando AGs.

3. Una empresa proveedora de energía eléctrica dispone de cuatro plantas de generación para satisfacer la demanda diaria de energía eléctrica en Cali, Bogotá, Medellín y Barranquilla. Cada una puede generar 3, 6, 5 y 4 GW al día respectivamente. Las necesidades de Cali, Bogotá, Medellín y Barranquilla son de 4, 3, 5 y 3 GW al día respectivamente. Los costos por el transporte de energía por cada GW entre plantas y ciudades se dan en la siguiente tabla:

| Planta   | Cali | Bogotá | Medellín | Barranq. |
| :------- | :--: | :----: | :------: | :------: |
| Planta C |  1   |   4    |    3     |    6     |
| Planta B |  4   |   1    |    4     |    5     |
| Planta M |  3   |   4    |    1     |    4     |
| Planta B |  6   |   5    |    4     |    1     |

**Costos del KW-H por generador:**

| Generador | $KW-H |
| :-------- | :---- |
| Planta C  | 680   |
| Planta B  | 720   |
| Planta M  | 660   |
| Planta B  | 750   |

Encontrar usando AGs el mejor despacho de energía minimizando los costos de transporte y generación.

> **4. Genere aleatoriamente una población de 50 matrices de 120 por 180, con números de 0 a 255, preséntelas como una gráfica RGB. La función de aptitud es una imagen cualquiera. Evolucione la población inicial hasta llegar a la imagen.**

> **5. Genere aleatoriamente una población de 50 palabras, que se escuche por el parlante del computador. Tomando como función de aptitud una palabra suya, usando AGs, con base en las palabras generadas aleatoriamente llegue a la palabra que usó como función de aptitud.**

6. Tome el algoritmo de la dieta y ahora incluya costos. Ahora encuentre una dieta que trate de satisfacer la dieta pero con un costo mínimo. Este es un ejemplo de AG multi-objetivo con dos funciones objetivo.

---

**Nota:** Se desarrollaron los ejercicios **1, 4 y 5** utilizando librerías de Python para computación evolutiva.
