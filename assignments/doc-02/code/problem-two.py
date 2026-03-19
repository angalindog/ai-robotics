import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# ─────────────────────────────────────────
# Parámetros del modelo
# ─────────────────────────────────────────
SIZE       = 50     # Tamaño de la grilla (SIZE x SIZE celdas)
DENSIDAD   = 0.65   # Fracción de celdas con árbol sano al inicio (0.0 a 1.0)
NUM_FOCOS  = 3      # Cantidad de focos iniciales de incendio
P_IGNICION = 0.4    # Probabilidad de que un árbol se enciende por cada vecino ardiendo
T_QUEMA    = 4      # Pasos que tarda un árbol en convertirse en carbón
T_CARBON   = 6      # Pasos que tarda el carbón en quedar vacío
PASOS      = 120    # Cantidad de iteraciones de la simulación

# ─────────────────────────────────────────
# Códigos de estado (igual que el robot)
# ─────────────────────────────────────────
VACIO  = 0   # celda sin árbol  → negro
SANO   = 1   # árbol sano       → verde
LLAMAS = 2   # árbol ardiendo   → naranja/rojo
CARBON = 3   # ceniza           → gris

# ─────────────────────────────────────────
# Inicializar el mapa (configuración C0)
# ─────────────────────────────────────────
def crear_mapa(size, densidad, num_focos):
    # Cada celda es SANO con probabilidad = densidad, VACIO si no
    mapa  = np.zeros((size, size), dtype=int)
    timer = np.zeros((size, size), dtype=int)  # cuenta cuántos pasos lleva cada celda en su estado

    for i in range(size):
        for j in range(size):
            if random.random() < densidad:
                mapa[i, j] = SANO

    # Colocar focos iniciales en posiciones aleatorias con árbol sano
    sanos = [(i, j) for i in range(size) for j in range(size) if mapa[i, j] == SANO]
    focos = random.sample(sanos, min(num_focos, len(sanos)))
    for fx, fy in focos:
        mapa[fx, fy] = LLAMAS
        timer[fx, fy] = 1

    return mapa, timer

# ─────────────────────────────────────────
# Contar vecinos ardiendo (vecindad de Moore: 8 vecinos)
# ─────────────────────────────────────────
def vecinos_ardiendo(mapa, x, y):
    count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue                          # la celda no es su propio vecino
            nx, ny = x + dx, y + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE: # verificar que esté dentro del mapa
                if mapa[nx, ny] == LLAMAS:
                    count += 1
    return count

# ─────────────────────────────────────────
# Un paso de tiempo del autómata celular
# Reglas probabilísticas aplicadas en paralelo:
#   R1. SANO   → LLAMAS : prob = 1-(1-P_IGNICION)^n_vecinos_ardiendo
#   R2. LLAMAS → CARBON : después de T_QUEMA pasos
#   R3. CARBON → VACIO  : después de T_CARBON pasos
# ─────────────────────────────────────────
def paso_tiempo(mapa, timer):
    nuevo_mapa  = mapa.copy()   # calcular todo sobre el estado actual
    nuevo_timer = timer.copy()  # antes de aplicar cualquier cambio (paralelismo)

    for i in range(SIZE):
        for j in range(SIZE):

            if mapa[i, j] == SANO:
                # R1: ignición por vecinos ardiendo
                n = vecinos_ardiendo(mapa, i, j)
                prob = 1 - (1 - P_IGNICION) ** n   # más vecinos ardiendo → más probable
                if random.random() < prob:
                    nuevo_mapa[i, j]  = LLAMAS
                    nuevo_timer[i, j] = 1

            elif mapa[i, j] == LLAMAS:
                # R2: sigue ardiendo o se convierte en carbón
                if timer[i, j] >= T_QUEMA:
                    nuevo_mapa[i, j]  = CARBON
                    nuevo_timer[i, j] = 1
                else:
                    nuevo_timer[i, j] = timer[i, j] + 1

            elif mapa[i, j] == CARBON:
                # R3: sigue como carbón o queda vacío
                if timer[i, j] >= T_CARBON:
                    nuevo_mapa[i, j]  = VACIO
                    nuevo_timer[i, j] = 0
                else:
                    nuevo_timer[i, j] = timer[i, j] + 1

    return nuevo_mapa, nuevo_timer

# ─────────────────────────────────────────
# Simulación principal
# ─────────────────────────────────────────
def simular_incendio():
    mapa, timer = crear_mapa(SIZE, DENSIDAD, NUM_FOCOS)

    # Colores: VACIO=negro, SANO=verde, LLAMAS=naranja, CARBON=gris
    cmap = ListedColormap(['#1a1a1a', '#2d8a2d', '#ff6600', '#888888'])

    plt.figure(figsize=(6, 6))

    for paso in range(PASOS):
        mapa, timer = paso_tiempo(mapa, timer)

        # Visualización
        plt.clf()
        plt.imshow(mapa, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
        llamas_activas = np.sum(mapa == LLAMAS)
        plt.title(f"Incendio Forestal — Paso {paso + 1}  |  En llamas: {llamas_activas}")
        plt.axis('off')
        plt.pause(0.15)

        # Detener si el fuego se extinguió
        if llamas_activas == 0:
            plt.title(f"Fuego extinguido en el paso {paso + 1}")
            print(f"Fuego extinguido en el paso {paso + 1}")
            print(f"Árboles sanos restantes : {np.sum(mapa == SANO)}")
            print(f"Celdas vacías (quemadas): {np.sum(mapa == VACIO)}")
            break

    plt.show()

# Ejecutar simulación
simular_incendio()