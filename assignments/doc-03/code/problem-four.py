# IMPORTACIÓN DE LIBRERÍAS
import numpy as np # Manejo de matrices (las imágenes son matrices de píxeles)
import matplotlib.pyplot as plt # Creación de gráficos estáticos
import matplotlib.animation as animation # Herramientas para animar figuras
from matplotlib.gridspec import GridSpec # Control avanzado de la posición de los gráficos
from PIL import Image # Procesamiento de imágenes y creación de GIFs
import io # Manejo de datos en memoria (evita guardar archivos temporales)
import random # Generación de números aleatorios para la evolución

# Fijamos semillas para que los resultados sean iguales cada vez que se ejecute el código
np.random.seed(42) 
random.seed(42)

# PARÁMETROS
H, W   = 30, 40 # Alto (Height) y Ancho (Width) de la imagen en píxeles
K      = 100 # Tamaño de la población (cuántas imágenes compiten a la vez)
M      = 2000 # Número de generaciones (cuántas veces van a evolucionar)
P_MUT  = 0.001 # Probabilidad de que un píxel cambie al azar (0.1%)

# OBJETIVO
def crea_imagen_objetivo(): #Crea la imagen que el algoritmo debe intentar imitar.
    # Se crea el fondo con un degradado de colores
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            img[i, j] = [int(255*i/H), int(255*j/W), 128]
    cx, cy, r = H//2, W//2, min(H, W)//4
    for i in range(H):
        for j in range(W):
            # RGB: El rojo cambia con la fila, el verde con la columna
            if (i-cx)**2 + (j-cy)**2 <= r**2:
                img[i, j] = [255, 255, 255]
    return img
objetivo = crea_imagen_objetivo()

# OPERADORES DEL ALGORITMO GENÉTICO
def genera_cromosoma(): # Crea una imagen aleatoria (un individuo de la población)
    # Genera una matriz de HxWx3 con valores de color aleatorios (0 a 255)
    return np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

def aptitud(crom): # Mide qué tan parecida es una imagen a la objetivo (Error Cuadrático Medio)
    diff = crom.astype(float) - objetivo.astype(float) # Diferencia de color píxel a píxel
    mse  = np.mean(diff**2) # Promedio de las diferencias 
    return 1.0 / (1.0 + mse) # A menor error, mayor aptitud

def seleccion_torneo(pob, apts): # Elige a los mejores individuos para ser padres mediante combates de a 3.
    nueva = []
    for _ in range(K):
        # Elige 3 al azar y el que tenga mejor aptitud gana el derecho a reproducirse
        cands   = random.sample(range(K), 3)
        ganador = max(cands, key=lambda i: apts[i])
        nueva.append(pob[ganador].copy())
    return nueva

def cruce(pob): #Mezcla dos imágenes padres cortándolas horizontalmente
    hijos = []
    for i in range(0, K, 2): # Punto de corte aleatorio (fila)
        pt = random.randint(1, H - 1) 
        # El hijo 1 recibe la parte superior del padre A y la inferior del padre B
        h1 = np.vstack([pob[i][:pt],   pob[i+1][pt:]])
        # El hijo 2 recibe la superior del padre B y la inferior del padre A
        h2 = np.vstack([pob[i+1][:pt], pob[i][pt:]])
        hijos += [h1, h2]
    return hijos

def mutacion(pob): # Cambia colores de píxeles al azar para mantener la diversidad
    for crom in pob:
        mascara      = np.random.random((H, W, 3)) < P_MUT
        ruido        = np.random.randint(0, 256, (H, W, 3))
        crom[mascara] = ruido.astype(np.uint8)[mascara]
    return pob

# CICLO PRINCIPAL DE EVOLUCIÓN
pob        = [genera_cromosoma() for _ in range(K)] # Generación 0 (puro ruido)
historial  = [] # Guarda la mejor aptitud de cada generación
mejor_crom = pob[0].copy()
mejor_apt  = aptitud(pob[0])
snapshots  = {}          # Fotos en momentos clave (0, 100, 400...)
frames_gif = []          # Todos los mejores cromosomas, uno por generación
hist_gif   = []          # Almacena las mejores imágenes para el video final

for gen in range(M):
    apts = [aptitud(c) for c in pob] # Evaluamos a todos
    idx  = np.argmax(apts) # Buscamos al mejor de esta generación

    # Si encontramos a alguien mejor que el mejor histórico, lo guardamos
    if apts[idx] > mejor_apt:
        mejor_apt  = apts[idx]
        mejor_crom = pob[idx].copy()

    # Guardamos fotos en hitos específicos
    historial.append(mejor_apt)
    frames_gif.append(mejor_crom.copy())
    hist_gif.append(list(historial))   # copia acumulada

    if gen in [0, 100, 400, M - 1]:
        snapshots[gen] = mejor_crom.copy()
    
    # Aplicación de la Selección Natural
    elite  = pob[idx].copy() # Elitismo: guardamos al mejor sin cambios
    padres = seleccion_torneo(pob, apts) # Selección
    hijos  = cruce(padres) # Reproducción
    pob    = mutacion(hijos) # Variación
    pob[0] = elite # Aseguramos que el mejor no se pierda

print(f"Evolucion de Imagen RGB")
print(f"  Resolucion : {H}x{W} px | Poblacion: {K} | Generaciones: {M}")
print(f"  Aptitud inicial : {historial[0]:.6f}")
print(f"  Aptitud final   : {historial[-1]:.6f}")
print(f"  Mejora          : {(historial[-1]/historial[0]-1)*100:.1f}%")


# GENERACIÓN DE RESULTADOS VISUALES (ESTÁTICOS)
# Crea una fila de imágenes mostrando el progreso
fig = plt.figure(figsize=(15, 4))
fig.suptitle("Ejercicio 4 — Evolución de Imagen con AG", fontsize=13, fontweight='bold')

gens_mostrar = sorted(snapshots.keys())
for k, gen in enumerate(gens_mostrar):
    ax = fig.add_subplot(1, len(gens_mostrar) + 2, k + 1)
    ax.imshow(snapshots[gen])
    ax.set_title(f'Gen {gen}')
    ax.axis('off')

# Muestra la imagen objetivo al final para comparar
ax_obj = fig.add_subplot(1, len(gens_mostrar) + 2, len(gens_mostrar) + 1)
ax_obj.imshow(objetivo)
ax_obj.set_title('Objetivo')
ax_obj.axis('off')

# Gráfico de línea que muestra cómo subió la aptitud
ax_conv = fig.add_subplot(1, len(gens_mostrar) + 2, len(gens_mostrar) + 2)
ax_conv.plot(historial, 'purple', linewidth=1.5)
ax_conv.set_xlabel('Generación')
ax_conv.set_ylabel('Aptitud')
ax_conv.set_title('Convergencia')
ax_conv.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ejercicio.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n  Grafica estatica guardada.")

# GENERACIÓN DEL GIF ANIMADO
print("  Generando GIF de evolucion (esto tarda unos segundos)...")

gif_frames = []

# Seleccionar 100 frames representativos para un GIF fluido pero liviano
indices_gif = list(range(0, M, max(1, M // 100)))
if M - 1 not in indices_gif:
    indices_gif.append(M - 1)

for gen_idx in indices_gif:
    img_evol  = frames_gif[gen_idx]
    apt_actual = historial[gen_idx]
    hist_hasta = list(historial[:gen_idx + 1])

    # Layout del frame: imagen izquierda | objetivo centro | grafica derecha
    fig_gif = plt.figure(figsize=(9, 3.2), facecolor='#1a1a2e')
    gs      = GridSpec(1, 3, figure=fig_gif, wspace=0.35,
                       left=0.05, right=0.97, top=0.82, bottom=0.18)

    # Panel izquierdo — imagen evolucionando
    ax1 = fig_gif.add_subplot(gs[0])
    ax1.imshow(img_evol)
    ax1.set_title(f'Generacion {gen_idx}', color='white', fontsize=9, pad=4)
    ax1.axis('off')
    # Borde de color que va de rojo → verde según aptitud
    pct   = apt_actual / max(historial) if max(historial) > 0 else 0
    color_borde = (1 - pct, pct, 0.2)
    for spine in ax1.spines.values():
        spine.set_edgecolor(color_borde)
        spine.set_linewidth(3)
        spine.set_visible(True)

    # Panel central — objetivo (fijo)
    ax2 = fig_gif.add_subplot(gs[1])
    ax2.imshow(objetivo)
    ax2.set_title('Objetivo', color='white', fontsize=9, pad=4)
    ax2.axis('off')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#00d4ff')
        spine.set_linewidth(2)
        spine.set_visible(True)

    # Panel derecho — curva de convergencia hasta este frame
    ax3 = fig_gif.add_subplot(gs[2])
    ax3.set_facecolor('#0d0d1a')
    ax3.plot(hist_hasta, color='#a855f7', linewidth=1.8)
    ax3.fill_between(range(len(hist_hasta)), hist_hasta,
                     alpha=0.25, color='#a855f7')
    ax3.axhline(max(historial), color='#00d4ff', linewidth=0.8,
                linestyle='--', alpha=0.6)
    ax3.set_xlim(0, M)
    ax3.set_ylim(historial[0] * 0.95, max(historial) * 1.05)
    ax3.set_xlabel('Generacion', color='#cccccc', fontsize=7)
    ax3.set_ylabel('Aptitud', color='#cccccc', fontsize=7)
    ax3.set_title('Convergencia', color='white', fontsize=9, pad=4)
    ax3.tick_params(colors='#aaaaaa', labelsize=6)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#444444')
    ax3.scatter([gen_idx], [apt_actual], color='#facc15',
                s=40, zorder=5)
    ax3.grid(True, alpha=0.15, color='#555555')

    # Título general del frame
    fig_gif.suptitle(
        f'AG — Evolucion de imagen   |   Aptitud: {apt_actual:.5f}   ({pct*100:.1f}% del maximo)',
        color='white', fontsize=9, fontweight='bold', y=0.97
    )
    fig_gif.patch.set_facecolor('#1a1a2e')

    # Convertir figura a imagen PIL
    buf = io.BytesIO()
    fig_gif.savefig(buf, format='png', dpi=90,
                    facecolor='#1a1a2e', bbox_inches='tight')
    buf.seek(0)
    gif_frames.append(Image.open(buf).copy())
    plt.close(fig_gif)
    buf.close()

# Guardar GIF — últimos frames más lentos para apreciar el resultado final
durations = [80] * len(gif_frames)
for i in range(-5, 0):
    durations[i] = 400   # últimos 5 frames más lentos

gif_frames[0].save(
    'ejercicio.gif',
    save_all    = True,
    append_images = gif_frames[1:],
    duration    = durations,
    loop        = 0,
    optimize    = True
)

print("  GIF guardado: ejercicio.gif")
print(f"  Frames en el GIF: {len(gif_frames)}")