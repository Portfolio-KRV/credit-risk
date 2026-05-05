# Evaluación de Riesgo Crediticio

Modelo de red bayesiana para predicción de riesgo crediticio usando algoritmo Hill-Climbing y método BICScore.

## Objetivos

- Identificar clientes de riesgo al otorgar préstamos usando Redes Bayesianas.
- Encontrar la estructura óptima de la red bayesiana usando algoritmo Hill-Climbing y método BICScore.
- Mejorar manualmente la estructura de la red para que sea consistente con el contexto de negocio.

## Tecnologías

![Pgmpy](https://img.shields.io/badge/Pgmpy-2C8EBB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Networkx](https://img.shields.io/badge/NetworkX-FF6F00?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white)

## Hallazgos Clave

- La evaluacion crediticia historica es el predictor mas fuerte: ECH=0 (Muy Malo) genera 64.5% de probabilidad de alto riesgo vs ECH=4 (Excelente) con solo 18.8%.
- Las conexiones encontradas por Hill-climbing con BICScore tienen sentido en el contexto de los datos, aunque algunas direcciones de causalidad estan invertidas.
- Los prestamos con duracion 24-72 meses tienen 52.2% de probabilidad de alto riesgo, significativamente mayor que plazos mas cortos.
- La construccion automatizada de redes bayesianas es un buen punto de partida pero debe analizarse bajo la experiencia de negocio.

## Curso

Introducción a la Ciencia de Datos

## Demo

Este proyecto tiene una demo disponible en el sitio web del portfolio.

## Repositorio

[https://github.com/Portfolio-KRV/credit-risk](https://github.com/Portfolio-KRV/credit-risk)
