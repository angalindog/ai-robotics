import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import Voronoi
from pathlib import Path

# Configuración de las tres capas

CAPAS = {
    "droguerias": {
        "json":          "droguerias_villavicencio.json",
        "titulo":        "Zonas de influencia — Droguerías / Farmacias",
        "color_punto":   "#D62828",   # rojo
        "color_region":  "#FADADD",
        "color_borde":   "#8B1A1A",
        "label_leyenda": "Droguería / Farmacia",
        "salida":        "voronoi_droguerias.png",
    },
    "salud": {
        "json":          "salud_villavicencio.json",
        "titulo":        "Zonas de influencia — Centros de Atención en Salud",
        "color_punto":   "#1A6BB5",   # azul
        "color_region":  "#D6E8FB",
        "color_borde":   "#0D3D6E",
        "label_leyenda": "Centro de salud / Hospital / Clínica",
        "salida":        "voronoi_salud.png",
    },
    "colegios": {
        "json":          "colegios_villavicencio.json",
        "titulo":        "Zonas de influencia — Colegios e Instituciones Educativas",
        "color_punto":   "#2E8B57",   # verde
        "color_region":  "#D4EDDA",
        "color_borde":   "#145A32",
        "label_leyenda": "Colegio / Institución educativa",
        "salida":        "voronoi_colegios.png",
    },
}

# Área urbana de Villavicencio (bounding box)

LAT_MIN, LAT_MAX = 4.07, 4.20
LON_MIN, LON_MAX = -73.68, -73.57

# Constantes de conversión grados → km (proyección local plana)
LAT_C_REF = (LAT_MIN + LAT_MAX) / 2
KM_LAT    = 111.0
KM_LON    = 111.0 * np.cos(np.radians(LAT_C_REF))

# Funciones

def cargar_puntos(json_path):
    """Lee el JSON y devuelve (nombres, lats, lons)."""
    if not Path(json_path).exists():
        raise FileNotFoundError(
            f"No se encontró '{json_path}'.\n"
            "Ejecuta primero: python obtener_datos.py"
        )
    with open(json_path, encoding="utf-8") as f:
        datos = json.load(f)
    if len(datos) < 4:
        raise ValueError(
            f"'{json_path}' tiene solo {len(datos)} punto(s). "
            "Se necesitan al menos 4 para Voronoi. "
            "Agrega puntos manualmente al JSON."
        )
    nombres = [d["nombre"] for d in datos]
    lats    = np.array([d["lat"] for d in datos])
    lons    = np.array([d["lon"] for d in datos])
    return nombres, lats, lons


def a_km(lats, lons, lat_c, lon_c):
    """Convierte lat/lon a coordenadas planas en km."""
    x = (lons - lon_c) * KM_LON
    y = (lats - lat_c) * KM_LAT
    return x, y


def bbox_km(lat_c, lon_c):
    x_min = (LON_MIN - lon_c) * KM_LON
    x_max = (LON_MAX - lon_c) * KM_LON
    y_min = (LAT_MIN - lat_c) * KM_LAT
    y_max = (LAT_MAX - lat_c) * KM_LAT
    return x_min, x_max, y_min, y_max


def calcular_voronoi(x, y, x_min, x_max, y_min, y_max, margen=5.0):
    """Calcula Voronoi con puntos fantasma en los bordes."""
    puntos_borde = np.array([
        [x_min - margen, y_min - margen],
        [x_max + margen, y_min - margen],
        [x_min - margen, y_max + margen],
        [x_max + margen, y_max + margen],
        [(x_min + x_max) / 2, y_min - margen],
        [(x_min + x_max) / 2, y_max + margen],
        [x_min - margen, (y_min + y_max) / 2],
        [x_max + margen, (y_min + y_max) / 2],
    ])
    puntos_reales = np.column_stack([x, y])
    puntos_ext    = np.vstack([puntos_reales, puntos_borde])
    return Voronoi(puntos_ext), puntos_reales


def generar_mapa(capa_id, config):
    print(f"\nGenerando: {config['salida']} ...", end=" ", flush=True)

    nombres, lats, lons = cargar_puntos(config["json"])
    lat_c = np.mean(lats)
    lon_c = np.mean(lons)

    x, y                         = a_km(lats, lons, lat_c, lon_c)
    x_min, x_max, y_min, y_max   = bbox_km(lat_c, lon_c)
    vor, puntos_reales            = calcular_voronoi(x, y, x_min, x_max, y_min, y_max)

    # ── Figura ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 15))
    fig.patch.set_facecolor('#F5F0E8')
    ax.set_facecolor('#EDF4FB')

    # Paleta de colores para regiones (variantes suaves del color base)
    base_rgb = mcolors_to_rgb(config["color_region"])
    colores  = generar_paleta(base_rgb, len(puntos_reales))
    np.random.seed(42)
    np.random.shuffle(colores)

    # — Regiones de Voronoi coloreadas —
    for i, region_idx in enumerate(vor.point_region[:len(puntos_reales)]):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        vertices = vor.vertices[region]
        ax.add_patch(plt.Polygon(
            vertices, closed=True,
            facecolor=colores[i],
            edgecolor=config["color_borde"],
            linewidth=0.8, linestyle='--', zorder=1
        ))

    # — Líneas de Voronoi —
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            p1, p2 = vor.vertices[simplex[0]], vor.vertices[simplex[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=config["color_borde"], linewidth=0.9,
                    alpha=0.6, zorder=2)

    # — Puntos —
    ax.scatter(x, y, s=120, c=config["color_punto"], marker='o',
               edgecolors='white', linewidths=1.5, zorder=5)

    # — Etiquetas —
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    for xi, yi, nombre in zip(x, y, nombres):
        dx = 0.15 if xi >= cx else -0.15
        dy = 0.10 if yi >= cy else -0.10
        nombre_corto = nombre[:28] + ("…" if len(nombre) > 28 else "")
        ax.annotate(
            nombre_corto,
            xy=(xi, yi), xytext=(xi + dx, yi + dy),
            fontsize=5.5, color='#1a1a1a',
            ha='left' if dx > 0 else 'right', va='center',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                      edgecolor='#cccccc', alpha=0.82, linewidth=0.5),
            arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5),
            zorder=6
        )

    # — Límites y rejilla —
    ax.set_xlim(x_min - 0.3, x_max + 0.3)
    ax.set_ylim(y_min - 0.3, y_max + 0.3)
    ax.grid(True, linestyle=':', color='#aaaaaa', alpha=0.4, zorder=0)

    # — Títulos y ejes —
    ax.set_title(
        f'Diagrama de Voronoi — {config["titulo"]}\n'
        'Villavicencio, Meta, Colombia  (datos: OpenStreetMap)',
        fontsize=13, fontweight='bold', pad=18, color='#1a1a1a'
    )
    ax.set_xlabel('Distancia este-oeste (km)', fontsize=10, color='#333')
    ax.set_ylabel('Distancia norte-sur (km)',  fontsize=10, color='#333')

    # Ejes secundarios con coordenadas geográficas
    def x_to_lon(v): return v / KM_LON + lon_c
    def y_to_lat(v): return v / KM_LAT + lat_c

    xticks = np.arange(np.ceil(x_min), np.floor(x_max) + 1, 1)
    yticks = np.arange(np.ceil(y_min), np.floor(y_max) + 1, 1)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(xticks[::2])
    ax2.set_xticklabels([f'{x_to_lon(v):.3f}°O' for v in xticks[::2]], fontsize=7)
    ax2.set_xlabel('Longitud', fontsize=8, color='#555')

    ax3 = ax.twinx()
    ax3.set_ylim(ax.get_ylim())
    ax3.set_yticks(yticks[::2])
    ax3.set_yticklabels([f'{y_to_lat(v):.3f}°N' for v in yticks[::2]], fontsize=7)
    ax3.set_ylabel('Latitud', fontsize=8, color='#555')

    # — Leyenda —
    ax.legend(
        handles=[
            mpatches.Patch(color=config["color_punto"],
                           label=f'{config["label_leyenda"]} (n={len(puntos_reales)})'),
            mpatches.Patch(facecolor=config["color_region"],
                           edgecolor=config["color_borde"], linestyle='--',
                           label='Región de Voronoi (zona de influencia)'),
            plt.Line2D([0], [0], color=config["color_borde"], linewidth=1,
                       linestyle='--', label='Frontera de Voronoi'),
        ],
        loc='lower right', fontsize=8.5, framealpha=0.92, edgecolor='#ccc'
    )

    # — Estadísticas —
    area_km2 = (x_max - x_min) * (y_max - y_min)
    ax.text(0.01, 0.99,
            f'Puntos mapeados:  {len(puntos_reales)}\n'
            f'Área cubierta:    ~{area_km2:.1f} km²\n'
            f'Densidad media:   {len(puntos_reales)/area_km2:.2f} / km²',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='#aaa', alpha=0.88, linewidth=0.8))

    # — Fuente —
    ax.text(0.01, 0.01,
            'Fuente: OpenStreetMap contributors (ODbL) | Ejercicio AC sección 2.10',
            transform=ax.transAxes, fontsize=6.5, color='#777', va='bottom')

    plt.tight_layout()
    plt.savefig(config["salida"], dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓")


# Utilidades de color

def mcolors_to_rgb(hex_color):
    """Convierte hex a (r, g, b) en rango 0-1."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def generar_paleta(base_rgb, n):
    """Genera n colores con variación de brillo alrededor del color base."""
    r, g, b = base_rgb
    paleta = []
    for i in range(n):
        factor = 0.75 + 0.50 * (i % 5) / 4   # varía entre 0.75 y 1.25
        rc = min(1.0, r * factor)
        gc = min(1.0, g * factor)
        bc = min(1.0, b * factor)
        paleta.append((rc, gc, bc, 0.40))      # alpha 0.40
    return paleta

# Main

if __name__ == "__main__":
    errores = []
    for capa_id, config in CAPAS.items():
        try:
            generar_mapa(capa_id, config)
        except FileNotFoundError as e:
            print(f"\n⚠ {e}")
            errores.append(capa_id)
        except ValueError as e:
            print(f"\n⚠ {e}")
            errores.append(capa_id)

    print(f"\n{'═'*50}")
    print("  ARCHIVOS GENERADOS")
    print(f"{'═'*50}")
    for capa_id, config in CAPAS.items():
        if capa_id not in errores:
            print(f"  ✓ {config['salida']}")
        else:
            print(f"  ✗ {config['salida']}  (sin datos)")
    print(f"{'═'*50}")