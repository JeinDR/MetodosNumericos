import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import Symbol, Poly, expand, sympify, diff, lambdify

def lagrange(datax, datay):
    x = Symbol('x')
    n = len(datax)
    polinomio = 0
    iteraciones = []
    for i in range(n):
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (x - datax[j]) / (datax[i] - datax[j])
        term = datay[i] * Li
        polinomio += term
        p = calcPoly(polinomio)
        iteraciones.append((i, f"L{i}(x) agregado → {poly_to_string(p)}"))
    return calcPoly(polinomio), iteraciones
