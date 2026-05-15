from flask import Flask, render_template, request
import math
import re

app = Flask(__name__)


def convertir_numero(valor):
    return float(valor.replace(",", "."))


def crear_funcion(texto):
    texto = texto.lower()
    texto = texto.replace(" ", "")
    texto = texto.replace("^", "**")
    texto = texto.replace("sen", "sin")
    texto = texto.replace("ln", "log")
    texto = texto.replace("√", "sqrt")
    texto = texto.replace("π", "pi")
    texto = texto.replace("²", "**2")
    texto = texto.replace("³", "**3")

    # Permite escribir 2x en vez de 2*x
    texto = re.sub(r"(\d)(x)", r"\1*\2", texto)

    # Permite escribir 2(x+1) en vez de 2*(x+1)
    texto = re.sub(r"(\d)\(", r"\1*(", texto)

    funciones_permitidas = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "abs": abs
    }

    def f(x):
        return eval(texto, {"__builtins__": {}}, {**funciones_permitidas, "x": x})

    return f


def derivada(f, x):
    h = 0.000001
    return (f(x + h) - f(x - h)) / (2 * h)


def newton_raphson(f, x_inicial, tolerancia):
    x = x_inicial
    resultado = "Newton-Raphson\n\n"

    for i in range(1, 21):
        fx = f(x)
        dfx = derivada(f, x)

        if abs(dfx) < tolerancia:
            resultado += "La derivada es muy cercana a cero. No se puede continuar."
            return resultado

        nuevo_x = x - fx / dfx

        resultado += f"Iteración {i}: x = {nuevo_x:.6f}\n"

        if abs(nuevo_x - x) < tolerancia:
            resultado += f"\nRaíz aproximada: {nuevo_x:.6f}"
            return resultado

        x = nuevo_x

    resultado += "\nNo se encontró una raíz en 20 iteraciones."
    return resultado


def generar_grafica(f, xs=None):
    puntos = []

    if xs and len(xs) > 0:
        minimo = min(xs) - 2
        maximo = max(xs) + 2
    else:
        minimo = -10
        maximo = 10

    paso = (maximo - minimo) / 80

    x = minimo
    while x <= maximo:
        try:
            y = f(x)

            if math.isfinite(y):
                puntos.append({
                    "x": round(x, 4),
                    "y": round(y, 6)
                })
        except:
            pass

        x += paso

    return puntos


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = ""
    grafica = None

    if request.method == "POST":
        metodo = request.form.get("metodo")
        valor_x = request.form.get("valor_x")
        funcion_texto = request.form.get("funcion")

        xs_texto = request.form.getlist("x")
        fxs_texto = request.form.getlist("fx")

        xs = []
        fxs = []

        for i in range(len(xs_texto)):
            try:
                x = convertir_numero(xs_texto[i])
                fx = convertir_numero(fxs_texto[i])

                xs.append(x)
                fxs.append(fx)
            except:
                pass

        try:
            if metodo == "newton-raphson":
                x_inicial = convertir_numero(valor_x)
                f = crear_funcion(funcion_texto)

                resultado = newton_raphson(f, x_inicial)
                grafica = generar_grafica(f)

            elif metodo == "newton-modificado":
                x_inicial = convertir_numero(valor_x)
                m = int(multiplicidad)

                f = crear_funcion(funcion_texto)

                resultado = newton_raphson_modificado(f, x_inicial, m)
                grafica = generar_grafica(f)

            elif metodo == "muller":
                x_inicial = convertir_numero(valor_x)
                f = crear_funcion(funcion_texto)

                resultado = muller(f, x_inicial - 1, x_inicial, x_inicial + 1)
                grafica = generar_grafica(f)

            elif metodo == "lagrange":
                if len(xs) < 2:
                    resultado = "Debe ingresar al menos 2 puntos."
                else:
                    x_eval = convertir_numero(valor_x)
                    f = lagrange(xs, fxs)

                    resultado = "Polinomio Interpolante de Lagrange\n\n"
                    resultado += f"P({x_eval}) = {f(x_eval):.6f}"

                    grafica = generar_grafica(f, xs)

            elif metodo == "neville":
                if len(xs) < 2:
                    resultado = "Debe ingresar al menos 2 puntos."
                else:
                    x_eval = convertir_numero(valor_x)
                    respuesta = neville(xs, fxs, x_eval)

                    resultado = "Método de Neville\n\n"
                    resultado += f"P({x_eval}) = {respuesta:.6f}"

                    grafica = generar_grafica(lambda x: neville(xs, fxs, x), xs)

            elif metodo == "newton-dd":
                if len(xs) < 2:
                    resultado = "Debe ingresar al menos 2 puntos."
                else:
                    x_eval = convertir_numero(valor_x)
                    f = newton_diferencias(xs, fxs)

                    resultado = "Polinomio Interpolante de Newton\n\n"
                    resultado += f"P({x_eval}) = {f(x_eval):.6f}"

                    grafica = generar_grafica(f, xs)

        except:
            resultado = "Ocurrió un error. Revise los datos ingresados."

    return render_template(
        "index.html",
        resultado=resultado,
        grafica=grafica
    )


if __name__ == "__main__":
    app.run(debug=True)