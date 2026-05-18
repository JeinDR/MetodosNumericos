from flask import Flask, render_template, request
import math
import re
from metodos.newton_raphson import calcular_newton_raphson
from metodos.lagrange import calcular_lagrange
import numpy as np
import json

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/newton-raphson", methods=["GET", "POST"])
def newton_raphson():
    resultado = ""
    polinomio = ""
    grafica = None

    if request.method == "POST":
        funcion = request.form.get("funcion")
        valor_x = request.form.get("valor_x")
        tolerancia = request.form.get("tolerancia")

        try:
            raiz, iteraciones, f, fp = calcular_newton_raphson(
                funcion,
                float(valor_x),
                float(tolerancia)
            )

            resultado = "Método: Newton-Raphson\n"
            resultado += f"Función: {funcion}\n"
            resultado += f"Valor inicial: {valor_x}\n"
            resultado += f"Tolerancia: {tolerancia}\n\n"
            resultado += "Iteraciones:\n"
            resultado += " i \t Xi \t\tf(Xi)\t\tf'(Xi)\t\t|x(i)-x(i-1)|\t\tError\n"

            for it in iteraciones:
                i, xi, fxi, fpxi, xi1, error = it
                resultado += f"{i}\t{xi:.6f}\t{fxi:.6f}\t{fpxi:.6f}\t{xi1:.6f}\t{error:.6f}\n"

            resultado += f"\nRaíz aproximada: {raiz:.6f}"
            
            xs = np.linspace(raiz - 5, raiz + 5, 100)
            ys = f(xs)
            
            grafica = [
            {"x": float(xs[i]), "y": float(ys[i])}
            for i in range(len(xs))
        ]

        except Exception as e:
            resultado = f"Error: {e}"

    return render_template(
        "ventana_newton_raphson.html",
        resultado=resultado,
        grafica=grafica


    )

@app.route("/muller", methods=["GET", "POST"])
def muller():
    resultado = ""

    if request.method == "POST":
        funcion = request.form.get("funcion")
        x0 = request.form.get("x0")
        x1 = request.form.get("x1")
        x2 = request.form.get("x2")
        tolerancia = request.form.get("tolerancia")

        resultado = f"""
            Método: Müller
            Función: {funcion}
            x0: {x0}
            x1: {x1}
            x2: {x2}
            Tolerancia: {tolerancia}
            """

    return render_template("ventana_muller.html", resultado=resultado)


@app.route("/lagrange", methods=["GET", "POST"])
def lagrange():
    resultado = ""
    iteraciones = []
    grafica = None

    if request.method == "POST":
        try:
            xs = request.form.getlist("x")
            fxs = request.form.getlist("fx")
            valor_x = request.form.get("valor_x")

            datax = [float(v) for v in xs if v.strip() != ""]
            datay = [float(v) for v in fxs if v.strip() != ""]
            vx = float(valor_x)

            polinomio_texto, resultado_val, iteraciones, f = calcular_lagrange(datax, datay, vx)

            resultado = f"P(x) = {polinomio_texto}<br>P({vx}) ≈ {resultado_val}"

            # Generar gráfica igual que newton_raphson
            x_min, x_max = min(datax) - 1, max(datax) + 1
            x_vals = np.linspace(x_min, x_max, 200).tolist()
            y_vals = [float(f(xi)) for xi in x_vals]

            grafica = json.dumps([{"x": xi, "y": yi} for xi, yi in zip(x_vals, y_vals)])

        except Exception as e:
            resultado = f"Error: {e}"

    return render_template(
        "ventana_lagrange.html",
        resultado=resultado,
        iteraciones=iteraciones,
        grafica=grafica
    )
@app.route("/neville", methods=["GET", "POST"])
def neville():
    resultado = ""

    if request.method == "POST":
        x = request.form.getlist("x")
        fx = request.form.getlist("fx")
        valor_x = request.form.get("valor_x")

        resultado = f"""
Método: Neville
Valores de x: {x}
Valores de f(x): {fx}
Valor a evaluar: {valor_x}
"""

    return render_template("ventana_neville.html", resultado=resultado)


if __name__ == "__main__":
    app.run(debug=True)