import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import Symbol, Poly, expand, sympify, diff, lambdify


def neville(datax, datay):
    x = Symbol('x')
    n = len(datax)
    poly = list(map(float, datay))
    iteraciones = []
    for k in range(1, n):
        for i in range(n - k):
            poly[i] = (
                (x - datax[i + k]) * poly[i] + (datax[i] - x) * poly[i + 1]
            ) / (datax[i] - datax[i + k])
        resultado = calcPoly(poly[0])
        iteraciones.append((k, poly_to_string(resultado)))
    return calcPoly(poly[0]), iteraciones

