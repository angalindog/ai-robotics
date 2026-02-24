# Se importan las librerias
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Se establecen parametros de operación del AG
L = 22 # bits (longitud) por cromosoma - resolución de 1/(2^22) ≈ 2.4e-7 en [0,1]
K = 70 # tamaño de la población
M = 200 # número de generaciones
P_MUT = 0.3 # probabilidad de mutación por bit
XI, XF = 0.0, 1.0 # dominio

# Función objetivo
def f(x):
    return x*math.sin(10*math.pi*x)+1 # xE[0,1]

# Codificación y decodificación
def decodifica(crom): #Convierte cadena binaria a valor real en [XI, XF]
    val = sum(crom[i]*2**(L-i-1) for i in range(L))
    return XI + (XF-XI)* val / (2**L -1)

# Operadores genéticos
def genera_poblacion():
    return [list(np.random.randint(0, 2, L)) for _ in range(K)]

def evalua_poblacion(pob): # Devuelve (aptitudes, probabilidades de selección)
    apts = []
    for crom in pob:
        x = decodifica(crom)
        apt = f(x) # f(x) ≥ 0 en todo [0,1] porque mínimo ≈ 0
        apts.append(max(apt, 1e-9)) # evitar división por cero
    total = sum(apts)
    probs = [a / total for a in apts]
    return apts, probs

def seleccion(pob, probs): #Selección por ruleta con reemplazo
    indices = list(range(K))
    elegidos = random.choices(indices, weights=probs, k=K)
    return [list(pob[i]) for i in elegidos]

def cruce(pob): # Cruce en un punto para cada par consecutivo
    hijos = []
    for i in range(0, K, 2):
        p1, p2 = list(pob[i]), list(pob[i+1])
        pt = random.randint(1, L - 1)
        h1 = p1[:pt] + p2[pt:]
        h2 = p2[:pt] + p1[pt:]
        hijos += [h1, h2]
    return hijos

def mutacion(pob): # Mutación bit a bit con probabilidad P_MUT
    for crom in pob:
        for j in range(L):
            if random.random() < P_MUT:
                crom[j] = 1 - crom[j]
    return pob

#Ciclo de procesamiento
pob = genera_poblacion()
historial_mejor = []
historial_promedio = []

for gen in range(M):
    apts, probs = evalua_poblacion(pob)

    mejor_apt = max(apts)
    mejor_x = decodifica(pob[apts.index(mejor_apt)])
    prom_apt = sum(apts) / K

    historial_mejor.append(mejor_apt)
    historial_promedio.append(prom_apt)

    # Elitismo - Conservar al mejor
    elite = list(pob[apts.index(mejor_apt)])

    padres = seleccion(pob, probs)
    hijos = cruce(padres)
    pob = mutacion(hijos)

    # Se reintroduce el élite en la primera posicion
    pob[0] = elite

# Resultados
apts, _ = evalua_poblacion(pob)
idx_mejor = apts.index(max(apts))
x_opt = decodifica(pob[idx_mejor])
f_opt = f(x_opt)


print(f"  x optimo encontrado : {x_opt:.8f}")
print(f"  f(x) maximo         : {f_opt:.8f}")
print(f"  (Teorico esperado   : x ~ 0.8521, f ~ 1.8504)")


# Gráficas 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Ejercicio 1 — Maximizar f(x) = x·sin(10πx) + 1", fontsize=14, fontweight='bold')

# Función y punto óptimo
xs = np.linspace(0, 1, 1000)
ys = [f(xi) for xi in xs]
axes[0].plot(xs, ys, 'b-', linewidth=1.5, label='f(x)')
axes[0].axvline(x_opt, color='red', linestyle='--', label=f'x óptimo = {x_opt:.4f}')
axes[0].scatter([x_opt], [f_opt], color='red', s=80, zorder=5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('Función y máximo encontrado')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Convergencia
axes[1].plot(historial_mejor,   'r-',  linewidth=1.5, label='Mejor aptitud')
axes[1].plot(historial_promedio,'b--', linewidth=1.2, label='Aptitud promedio')
axes[1].set_xlabel('Generación')
axes[1].set_ylabel('Aptitud')
axes[1].set_title('Convergencia del AG')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exp3.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  Gráfica guardada.")