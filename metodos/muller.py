import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import Symbol, Poly, expand, sympify, diff, lambdify


def muller(func_str, x0, x1, x2, tol, mx=100):
    x = Symbol('x')

    try:
        f_sym = sympify(func_str)
        f = lambdify(x, f_sym, 'numpy')
    except Exception as e:
        raise ValueError(f"Función inválida: {e}")

    iteraciones = []

    x0 = complex(float(x0))
    x1 = complex(float(x1))
    x2 = complex(float(x2))

    for i in range(1, mx + 1):
        f0 = complex(f(x0))
        f1 = complex(f(x1))
        f2 = complex(f(x2))

        h0 = x0 - x2
        h1 = x1 - x2
        h2 = x0 - x1

        if h0 == 0 or h1 == 0 or h2 == 0:
            raise ValueError("Los puntos iniciales no deben ser iguales.")

        d0 = (f1 - f2)
        d1 = (f0 - f2)

        a = (h1*d1- h0*d0)/(h0*h1*h2)
        b = (h0**2*d0- h1**2*d1)/(h0*h1*h2)
        c = f2

        discriminante = b**2 - 4*a*c
        raiz_disc = discriminante**0.5

        den1 = b + raiz_disc
        den2 = b - raiz_disc

        if abs(den1) > abs(den2):
            denominador = den1
        else:
            denominador = den2

        if abs(denominador) < 1e-12:
            raise ValueError("Denominador cercano a cero. Elige otros puntos iniciales.")

        x3 = x2 - (2 * c) / denominador
        error = abs(x3 - x2)

        iteraciones.append((i, x0, x1, x2, f2, x3, error))

        if error < tol:
            x2 = x3
            break

        x0 = x1
        x1 = x2
        x2 = x3

    if abs(x2.imag) < 1e-10:
        x2 = x2.real

    return x2, iteraciones, f