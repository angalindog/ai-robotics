# Desarrollo Taller 2

## Respuesta 1

Reglas básicas sobre comportamiento en distintos lugares.

```mermaid
mindmap
root((Comportamientos))
    Casa
        Mantener zona de estudio despejada
        Usar calzado cómodo
        Tener algún pasabocas para picar
        Mantener lleno el termo de agua
    Medio de transporte
        Bicicleta
            Revisar a diario calbración de llantas
            Estar atento a ruidos anormales en el trayecto
            Usar bloqueador solar
            Cubrir la cara con balaclava
        Transmilenio
            Recargar entre-semana en hora valle
            Usar el morral al frente
            Tener música lista
            No sacar el celular
    Universidad
        No usar redes sociales
        Tener siempre hoja y esfero
        Estar atento a correo institucional
```

## Respuesta 3

Para realizar la simulación podemos hacer uso de la herramienta vista en clase NetLogo Web, donde se crea un nuevo modelo con el siguiente código:

```
turtles-own [
  sensor-frontal
  sensor-izquierdo
  sensor-derecho
]

to setup
  clear-all
  
  create-turtles 1 [
    set color green
    set size 2
    setxy 0 0
  ]
  
  create-turtles 4 [
    set color red
    set size 2
    setxy random-xcor random-ycor
    set shape "circle"
  ]
  
  reset-ticks
end

to detectar
  ask turtle 0 [
    set sensor-frontal distance min-one-of other turtles [distance myself]
    
    rt 45
    set sensor-derecho distance min-one-of other turtles [distance myself]
    
    lt 90
    set sensor-izquierdo distance min-one-of other turtles [distance myself]
    
    rt 45
  ]
end

to mover
  ask turtle 0 [
    
    detectar
    
    if sensor-frontal < 3 [
      rt 90
    ]
    
    if sensor-izquierdo < 3 [
      rt 45
    ]
    
    if sensor-derecho < 3 [
      lt 45
    ]
    
    fd 1
  ]
end

to go
  mover
  tick
end
```

Para la interfaz se usa un botón de "setup" o configuración del entorno, y un botón "go", para iniciar el movimiento.

![Animación del resultado obtenido](/assignments/doc-02/netLogoEX.gif)