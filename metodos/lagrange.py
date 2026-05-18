import numpy as np
from sympy import Symbol, Poly, expand, sympify, diff, lambdify,N

def calcular_lagrange(datax, datay, valor_x):
    x = Symbol('x')
    n = len(datax)
    polinomio = 0
    iteraciones = []

    for i in range(n):
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (x - datax[j]) / (datax[i] - datax[j])

        termino = datay[i] * Li
        polinomio += termino

        polinomio_actual = expand(polinomio)
        texto = str(polinomio_actual).replace("**", "^")
        iteraciones.append(f"L{i}(x) agregado → {texto}")

    polinomio_final = expand(polinomio)
    resultado = N(polinomio_final.subs(x, valor_x), 10)
    polinomio_texto = str(polinomio_final).replace("**", "^")

    # Igual que newton: retornar la función lambdify lista para graficar
    f = lambdify(x, polinomio_final, 'numpy')

    return polinomio_texto, resultado, iteraciones, f
