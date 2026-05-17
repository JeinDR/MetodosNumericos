import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import Symbol, Poly, expand, sympify, diff, lambdify
from sympy import exp, E, pi, sin, cos, tan, log, sqrt


def calcular_newton_raphson(func_str, x0, tol=1e-7, max_iter=100):
    x = Symbol('x')
    try:
        func_str = func_str.replace("^", "**")
        func_str = func_str.replace("ln(", "log(")
        func_str = func_str.replace("e", "E")

        f_sym = sympify(func_str)
        f_prime_sym = diff(f_sym, x)
        f = lambdify(x, f_sym, 'numpy')
        fp = lambdify(x, f_prime_sym, 'numpy')
    except Exception as e:
        raise ValueError(f"Función inválida: {e}")

    iteraciones = []
    xi = float(x0)
    for i in range(1, max_iter + 1):
        fxi = f(xi)
        fpxi = fp(xi)
        if abs(fpxi) < 1e-12:
            raise ValueError("Derivada cercana a cero. Elige otro punto inicial.")
        xi1 = xi - fxi / fpxi
        error = abs(xi1 - xi)
        iteraciones.append((i, xi, fxi, fpxi, xi1, error))
        if error < tol:
            xi = xi1
            break
        xi = xi1

    return xi, iteraciones, f, fp
