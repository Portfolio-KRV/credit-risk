# Credit-risk
Desarrollado en el curso de Introducción a la Ciencia de Datos por: Kevin Reyes.
## Objetivos
- Identificar clientes de riesgo al otorgar créditos utilizando Redes Bayesianas.
- Encontrar estructura óptima de la red bayesiana utilizando algoritmo Hill-Climbing y método BICScore..
- Mejorar manualmente estructura de la red de forma de que sea coherente con el contexto del negocio.

## Descripción
Modelo de red bayesiana para la predicción de riesgo crediticio.

## Conclusiones
- Las conexiones encontradas por el método Hill-climbing con BICScore tienen sentido en el contexto de los datos, sin embargo, algunas direcciones de las causalidades están invertidas.
- La construcción automatizada de la red bayesiana es un buen punto de partida para modelar un contexto de negocio, sin embargo, se debe analizar bajo la experiencia del negocio para realizar modificaciones pertinentes.
- Los créditos de mayor duración tiene una probabilidad de riesgo alto mayor.

## Stack de tecnologías
- Pgmpy.
- Networkx.
- Pylab.
- Pandas.
- Numpy.