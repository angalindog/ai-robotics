"""
Extracción de tres capas desde OpenStreetMap — Villavicencio, Meta
    · Droguerías / farmacias
    · Centros de atención en salud
    · Colegios / instituciones educativas

Genera por cada capa:
    {capa}_villavicencio.json
    {capa}_villavicencio.csv

Dependencias:
    pip install requests
"""

import requests
import json
import csv
import time

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# ─────────────────────────────────────────────────────────────
# NOTA: geocodeArea fue eliminado — solo funciona en Overpass
# Turbo (web), no en la API directa. Se usan dos métodos:
#   1. area_id: ID de la relación OSM de Villavicencio
#      (relation 835030 → area ID = 3600835030)
#   2. bbox: bounding box geográfico, siempre funciona
# ─────────────────────────────────────────────────────────────

CAPAS = {

    "droguerias": {
        "nombre_humano": "Droguerías / Farmacias",
        "queries": [
            ("area_id",
             """
[out:json][timeout:90];
area(3600835030)->.searchArea;
(
  nwr["amenity"="pharmacy"](area.searchArea);
  nwr["shop"="pharmacy"](area.searchArea);
  nwr["shop"="chemist"](area.searchArea);
  nwr["healthcare"="pharmacy"](area.searchArea);
);
out center;
"""),
            ("bbox",
             """
[out:json][timeout:90];
(
  nwr["amenity"="pharmacy"](4.07,-73.68,4.20,-73.57);
  nwr["shop"="pharmacy"](4.07,-73.68,4.20,-73.57);
  nwr["shop"="chemist"](4.07,-73.68,4.20,-73.57);
  nwr["healthcare"="pharmacy"](4.07,-73.68,4.20,-73.57);
);
out center;
"""),
        ],
    },

    "salud": {
        "nombre_humano": "Centros de atención en salud",
        "queries": [
            ("area_id",
             """
[out:json][timeout:90];
area(3600835030)->.searchArea;
(
  nwr["amenity"="hospital"](area.searchArea);
  nwr["amenity"="clinic"](area.searchArea);
  nwr["amenity"="doctors"](area.searchArea);
  nwr["amenity"="health_post"](area.searchArea);
  nwr["healthcare"="hospital"](area.searchArea);
  nwr["healthcare"="clinic"](area.searchArea);
  nwr["healthcare"="centre"](area.searchArea);
  nwr["healthcare"="doctor"](area.searchArea);
  nwr["healthcare"="health_post"](area.searchArea);
);
out center;
"""),
            ("bbox",
             """
[out:json][timeout:90];
(
  nwr["amenity"="hospital"](4.07,-73.68,4.20,-73.57);
  nwr["amenity"="clinic"](4.07,-73.68,4.20,-73.57);
  nwr["amenity"="doctors"](4.07,-73.68,4.20,-73.57);
  nwr["amenity"="health_post"](4.07,-73.68,4.20,-73.57);
  nwr["healthcare"="hospital"](4.07,-73.68,4.20,-73.57);
  nwr["healthcare"="clinic"](4.07,-73.68,4.20,-73.57);
  nwr["healthcare"="centre"](4.07,-73.68,4.20,-73.57);
  nwr["healthcare"="doctor"](4.07,-73.68,4.20,-73.57);
  nwr["healthcare"="health_post"](4.07,-73.68,4.20,-73.57);
);
out center;
"""),
        ],
    },

    "colegios": {
        "nombre_humano": "Colegios / Instituciones educativas",
        "queries": [
            ("area_id",
             """
[out:json][timeout:90];
area(3600835030)->.searchArea;
(
  nwr["amenity"="school"](area.searchArea);
  nwr["amenity"="college"](area.searchArea);
  nwr["amenity"="university"](area.searchArea);
  nwr["amenity"="kindergarten"](area.searchArea);
);
out center;
"""),
            ("bbox",
             """
[out:json][timeout:90];
(
  nwr["amenity"="school"](4.07,-73.68,4.20,-73.57);
  nwr["amenity"="college"](4.07,-73.68,4.20,-73.57);
  nwr["amenity"="university"](4.07,-73.68,4.20,-73.57);
  nwr["amenity"="kindergarten"](4.07,-73.68,4.20,-73.57);
);
out center;
"""),
        ],
    },
}

PAUSA_ENTRE_INTENTOS = 12   # segundos entre requests fallidos (evita 429)
PAUSA_ENTRE_CAPAS    = 15   # segundos entre capas exitosas


# ─────────────────────────────────────────────────────────────
# Funciones
# ─────────────────────────────────────────────────────────────

def extraer_punto(elemento):
    if elemento["type"] == "node":
        return elemento["lat"], elemento["lon"]
    if "center" in elemento:
        return elemento["center"]["lat"], elemento["center"]["lon"]
    return None, None


def extraer_nombre(tags):
    for campo in ("name", "brand", "operator"):
        if tags.get(campo):
            return tags[campo]
    return "Sin nombre"


def ejecutar_query(nombre_intento, query):
    print(f"   · Intento '{nombre_intento}'...", end=" ", flush=True)
    try:
        resp = requests.get(
            OVERPASS_URL,
            params={"data": query},
            timeout=120,
            headers={"User-Agent": "VoronoiAcademico/1.0"}
        )
        resp.raise_for_status()
        elementos = resp.json().get("elements", [])
        print(f"{len(elementos)} elementos")
        return elementos
    except requests.exceptions.ConnectionError:
        print("ERROR: sin conexión")
        return "SIN_CONEXION"
    except requests.exceptions.Timeout:
        print("ERROR: timeout")
        return None
    except requests.exceptions.HTTPError as e:
        codigo = e.response.status_code if e.response else "?"
        if codigo == 429:
            print(f"ERROR 429: demasiadas solicitudes")
        elif codigo == 400:
            print(f"ERROR 400: query inválida")
        else:
            print(f"ERROR HTTP {codigo}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def procesar(elementos, capa_id):
    vistos, resultado = set(), []
    for e in elementos:
        tags     = e.get("tags", {})
        lat, lon = extraer_punto(e)
        if lat is None:
            continue
        clave = (round(lat, 5), round(lon, 5))
        if clave in vistos:
            continue
        vistos.add(clave)
        resultado.append({
            "osm_id":     e.get("id", ""),
            "osm_type":   e.get("type", ""),
            "capa":       capa_id,
            "nombre":     extraer_nombre(tags),
            "lat":        lat,
            "lon":        lon,
            "amenity":    tags.get("amenity", ""),
            "healthcare": tags.get("healthcare", ""),
            "shop":       tags.get("shop", ""),
            "brand":      tags.get("brand", ""),
            "direccion":  tags.get("addr:street", ""),
            "barrio":     tags.get("addr:suburb", ""),
        })
    return resultado


def guardar(resultado, capa_id):
    json_path = f"{capa_id}_villavicencio.json"
    csv_path  = f"{capa_id}_villavicencio.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    if resultado:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(resultado[0].keys()))
            w.writeheader()
            w.writerows(resultado)
    print(f"   ✓ {json_path}  ({len(resultado)} registros)")
    print(f"   ✓ {csv_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Servidor: {OVERPASS_URL}\n")
    resumen = {}

    for capa_id, config in CAPAS.items():
        print(f"\n{'═'*55}")
        print(f"  {config['nombre_humano'].upper()}")
        print(f"{'═'*55}")

        resultado    = []
        sin_conexion = False

        for idx, (nombre_intento, query) in enumerate(config["queries"]):
            # Pausa antes de cada intento excepto el primero de cada capa
            if idx > 0:
                print(f"   ⏳ Esperando {PAUSA_ENTRE_INTENTOS}s antes del siguiente intento...")
                time.sleep(PAUSA_ENTRE_INTENTOS)

            elementos = ejecutar_query(nombre_intento, query)

            if elementos == "SIN_CONEXION":
                sin_conexion = True
                break

            if elementos is None:
                continue  # probar siguiente método

            resultado = procesar(elementos, capa_id)
            if resultado:
                print(f"   → {len(resultado)} puntos únicos con '{nombre_intento}'")
                break
            else:
                print(f"   Sin resultados con '{nombre_intento}', probando siguiente...")

        if sin_conexion:
            print("   Sin conexión — abortando.")
            break

        if not resultado:
            print("   ⚠ Sin datos en OSM para esta capa.")
            print("   → Agrega puntos manualmente al JSON (ver README).")
        else:
            guardar(resultado, capa_id)

        resumen[capa_id] = len(resultado)
        print(f"   ⏳ Esperando {PAUSA_ENTRE_CAPAS}s antes de la siguiente capa...")
        time.sleep(PAUSA_ENTRE_CAPAS)

    # ── Resumen final ──────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  RESUMEN FINAL")
    print(f"{'═'*55}")
    for capa_id, config in CAPAS.items():
        n      = resumen.get(capa_id, 0)
        estado = "✓" if n > 0 else "⚠"
        print(f"  {estado} {config['nombre_humano']:<35} {n:3d} puntos")
    print(f"{'═'*55}")
    print("\nSiguiente paso:")
    print("  python voronoi_tres_capas.py")