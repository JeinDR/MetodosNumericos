#!/usr/bin/env python
# coding: utf-8

import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import Symbol, Poly, expand, sympify, diff, lambdify


# ══════════════════════════════════════════════════════════════════
#   LÓGICA MATEMÁTICA
# ══════════════════════════════════════════════════════════════════

def calcPoly(expr):
    x = Symbol('x')
    try:
        p = Poly(expand(expr), x)
        coeffs = [float(c) for c in p.all_coeffs()]
        return np.poly1d(coeffs)
    except Exception:
        return np.poly1d([0])


def poly_to_string(poly):
    s = ""
    power = len(poly.coefficients) - 1
    for coef in poly.coefficients:
        if abs(coef) > 1e-10:
            if power == 0:
                s += "{:.4f}".format(coef)
            else:
                s += "({:.4f})x^{} + ".format(coef, power)
        power -= 1
    return s.rstrip(" + ") or "0"


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


def newton_raphson(func_str, x0, tol=1e-7, max_iter=100):
    x = Symbol('x')
    try:
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



# ══════════════════════════════════════════════════════════════════
#   UTILIDADES DE UI
# ══════════════════════════════════════════════════════════════════

AZUL    = "#2c3e6b"
AZUL2   = "#4a90d9"
VERDE   = "#5cb85c"
ROJO    = "#d9534f"
FONDO   = "#f4f6fb"
BLANCO  = "#ffffff"
GRIS    = "#dde3ef"

def estilo_boton(btn, color=AZUL2):
    btn.configure(bg=color, fg=BLANCO, font=("Arial", 10, "bold"),
                  relief=tk.FLAT, padx=10, pady=5, cursor="hand2")

def frame_seccion(parent, titulo):
    f = tk.LabelFrame(parent, text=titulo, bg=FONDO,
                      font=("Arial", 10, "bold"), fg=AZUL, padx=8, pady=6)
    return f

def tabla_header(parent, cols):
    for c, col in enumerate(cols):
        tk.Label(parent, text=col, bg=AZUL, fg=BLANCO,
                 font=("Arial", 9, "bold"), width=14, relief=tk.FLAT
                 ).grid(row=0, column=c, padx=1, pady=1)

def ventana_base(titulo, ancho=980, alto=700):
    win = tk.Toplevel()
    win.title(titulo)
    win.geometry(f"{ancho}x{alto}")
    win.configure(bg=FONDO)
    win.grab_set()
    return win


# ══════════════════════════════════════════════════════════════════
#   VENTANA: NEVILLE
# ══════════════════════════════════════════════════════════════════

class VentanaNeville:
    def __init__(self):
        self.win = ventana_base("Interpolación de Neville")
        self.filas = []
        self._build()

    def _build(self):
        # ── izquierda ──
        left = tk.Frame(self.win, bg=FONDO, padx=12, pady=12)
        left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(left, text="Interpolación de Neville",
                 font=("Arial", 14, "bold"), bg=FONDO, fg=AZUL).pack(pady=(0,10))

        # tabla de puntos
        sec = frame_seccion(left, "Puntos (x, y)")
        sec.pack(fill=tk.X)
        self.tframe = tk.Frame(sec, bg=FONDO)
        self.tframe.pack()
        tabla_header(self.tframe, ["X", "Y", ""])
        self.next_row = 1
        for _ in range(4):
            self._add_row()

        # botones de tabla
        bf = tk.Frame(left, bg=FONDO)
        bf.pack(pady=6)
        b1 = tk.Button(bf, text="+ Punto", command=self._add_row)
        estilo_boton(b1, VERDE)
        b1.pack(side=tk.LEFT, padx=4)
        b2 = tk.Button(bf, text="Limpiar", command=self._limpiar)
        estilo_boton(b2, ROJO)
        b2.pack(side=tk.LEFT, padx=4)

        # calcular
        bc = tk.Button(left, text="▶  Calcular", command=self._calcular)
        estilo_boton(bc)
        bc.pack(fill=tk.X, pady=8)

        # resultado
        sec2 = frame_seccion(left, "Polinomio resultante")
        sec2.pack(fill=tk.X)
        self.lbl_result = tk.Label(sec2, text="—", wraplength=260,
                                   font=("Courier", 9), bg=BLANCO,
                                   relief=tk.SUNKEN, padx=4, pady=4)
        self.lbl_result.pack(fill=tk.X)

        # iteraciones
        sec3 = frame_seccion(left, "Iteraciones")
        sec3.pack(fill=tk.X, pady=(8,0))
        self.iter_text = tk.Text(sec3, height=12, width=36,
                                 font=("Courier", 8), state=tk.DISABLED,
                                 bg=BLANCO, relief=tk.SUNKEN)
        self.iter_text.pack(fill=tk.X)

        # ── derecha (gráfica) ──
        right = tk.Frame(self.win, bg=FONDO, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(6,5))
        self.fig.patch.set_facecolor(FONDO)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_row(self):
        r = self.next_row
        ex = tk.Entry(self.tframe, width=10, font=("Arial", 10))
        ey = tk.Entry(self.tframe, width=10, font=("Arial", 10))
        ex.grid(row=r, column=0, padx=2, pady=2)
        ey.grid(row=r, column=1, padx=2, pady=2)
        bd = tk.Button(self.tframe, text="✕", fg=ROJO, bg=FONDO,
                       relief=tk.FLAT,
                       command=lambda p=(ex,ey): self._del_row(p))
        bd.grid(row=r, column=2, padx=2)
        self.filas.append((ex, ey))
        self.next_row += 1

    def _del_row(self, par):
        if len(self.filas) <= 2:
            messagebox.showwarning("Aviso", "Mínimo 2 puntos.", parent=self.win)
            return
        for w in self.tframe.grid_slaves():
            if w in par:
                w.grid_forget(); w.destroy()
        self.filas.remove(par)

    def _limpiar(self):
        for ex, ey in self.filas:
            ex.delete(0, tk.END); ey.delete(0, tk.END)
        self.lbl_result.config(text="—")
        self._set_iter("")
        self.ax.clear(); self.canvas.draw()

    def _set_iter(self, txt):
        self.iter_text.config(state=tk.NORMAL)
        self.iter_text.delete("1.0", tk.END)
        self.iter_text.insert(tk.END, txt)
        self.iter_text.config(state=tk.DISABLED)

    def _calcular(self):
        dx, dy = [], []
        for ex, ey in self.filas:
            sx, sy = ex.get().strip(), ey.get().strip()
            if sx and sy:
                try:
                    dx.append(float(sx)); dy.append(float(sy))
                except ValueError:
                    messagebox.showerror("Error", f"Valor inválido: {sx} / {sy}", parent=self.win)
                    return
        if len(dx) < 2:
            messagebox.showwarning("Aviso", "Ingresa al menos 2 puntos.", parent=self.win)
            return
        if len(set(dx)) != len(dx):
            messagebox.showerror("Error", "Los X deben ser únicos.", parent=self.win)
            return
        try:
            poly, iters = neville(dx, dy)
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.win); return

        self.lbl_result.config(text=poly_to_string(poly))
        self._set_iter("\n".join(f"Iter {k}: {p}" for k, p in iters))
        self._graficar(dx, dy, poly)

    def _graficar(self, dx, dy, poly):
        self.ax.clear()
        xmin, xmax = min(dx), max(dx)
        m = (xmax - xmin) * 0.2 or 1
        t = np.linspace(xmin - m, xmax + m, 300)
        self.ax.plot(t, poly(t), '--g', lw=2, label='Neville')
        self.ax.scatter(dx, dy, color='blue', zorder=5, s=60, label='Puntos')
        self.ax.set_title("Neville"); self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.legend(fontsize=8); self.ax.grid(True)
        self.fig.tight_layout(); self.canvas.draw()


# ══════════════════════════════════════════════════════════════════
#   VENTANA: LAGRANGE
# ══════════════════════════════════════════════════════════════════

class VentanaLagrange:
    def __init__(self):
        self.win = ventana_base("Interpolación de Lagrange")
        self.filas = []
        self._build()

    def _build(self):
        left = tk.Frame(self.win, bg=FONDO, padx=12, pady=12)
        left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(left, text="Interpolación de Lagrange",
                 font=("Arial", 14, "bold"), bg=FONDO, fg=AZUL).pack(pady=(0,10))

        sec = frame_seccion(left, "Valores de x")
        sec.pack(fill=tk.X)
        self.tframe = tk.Frame(sec, bg=FONDO)
        self.tframe.pack()
        tabla_header(self.tframe, ["X"])
        self.next_row = 1
        for _ in range(4):
            self._add_row()

        bf = tk.Frame(left, bg=FONDO)
        bf.pack(pady=6)
        b1 = tk.Button(bf, text="+ Punto", command=self._add_row)
        estilo_boton(b1, VERDE); b1.pack(side=tk.LEFT, padx=4)
        b2 = tk.Button(bf, text="Limpiar", command=self._limpiar)
        estilo_boton(b2, ROJO); b2.pack(side=tk.LEFT, padx=4)

        bc = tk.Button(left, text="▶  Calcular", command=self._calcular)
        estilo_boton(bc); bc.pack(fill=tk.X, pady=8)

        sec2 = frame_seccion(left, "Polinomio resultante")
        sec2.pack(fill=tk.X)
        self.lbl_result = tk.Label(sec2, text="—", wraplength=260,
                                   font=("Courier", 9), bg=BLANCO,
                                   relief=tk.SUNKEN, padx=4, pady=4)
        self.lbl_result.pack(fill=tk.X)

        sec3 = frame_seccion(left, "Términos L_i(x)")
        sec3.pack(fill=tk.X, pady=(8,0))
        self.iter_text = tk.Text(sec3, height=12, width=36,
                                 font=("Courier", 8), state=tk.DISABLED,
                                 bg=BLANCO, relief=tk.SUNKEN)
        self.iter_text.pack(fill=tk.X)

        right = tk.Frame(self.win, bg=FONDO, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(6,5))
        self.fig.patch.set_facecolor(FONDO)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_row(self):
        r = self.next_row
        ex = tk.Entry(self.tframe, width=10, font=("Arial", 10))
        ey = tk.Entry(self.tframe, width=10, font=("Arial", 10))
        ex.grid(row=r, column=0, padx=2, pady=2)
        ey.grid(row=r, column=1, padx=2, pady=2)
        bd = tk.Button(self.tframe, text="✕", fg=ROJO, bg=FONDO,
                       relief=tk.FLAT,
                       command=lambda p=(ex,ey): self._del_row(p))
        bd.grid(row=r, column=2, padx=2)
        self.filas.append((ex, ey))
        self.next_row += 1

    def _del_row(self, par):
        if len(self.filas) <= 2:
            messagebox.showwarning("Aviso", "Mínimo 2 puntos.", parent=self.win)
            return
        for w in self.tframe.grid_slaves():
            if w in par:
                w.grid_forget(); w.destroy()
        self.filas.remove(par)

    def _limpiar(self):
        for ex, ey in self.filas:
            ex.delete(0, tk.END); ey.delete(0, tk.END)
        self.lbl_result.config(text="—")
        self._set_iter("")
        self.ax.clear(); self.canvas.draw()

    def _set_iter(self, txt):
        self.iter_text.config(state=tk.NORMAL)
        self.iter_text.delete("1.0", tk.END)
        self.iter_text.insert(tk.END, txt)
        self.iter_text.config(state=tk.DISABLED)

    def _calcular(self):
        dx, dy = [], []
        for ex, ey in self.filas:
            sx, sy = ex.get().strip(), ey.get().strip()
            if sx and sy:
                try:
                    dx.append(float(sx)); dy.append(float(sy))
                except ValueError:
                    messagebox.showerror("Error", f"Valor inválido: {sx} / {sy}", parent=self.win)
                    return
        if len(dx) < 2:
            messagebox.showwarning("Aviso", "Ingresa al menos 2 puntos.", parent=self.win)
            return
        if len(set(dx)) != len(dx):
            messagebox.showerror("Error", "Los X deben ser únicos.", parent=self.win)
            return
        try:
            poly, iters = lagrange(dx, dy)
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.win); return

        self.lbl_result.config(text=poly_to_string(poly))
        self._set_iter("\n".join(f"Paso {k}: {p}" for k, p in iters))
        self._graficar(dx, dy, poly)

    def _graficar(self, dx, dy, poly):
        self.ax.clear()
        xmin, xmax = min(dx), max(dx)
        m = (xmax - xmin) * 0.2 or 1
        t = np.linspace(xmin - m, xmax + m, 300)
        self.ax.plot(t, poly(t), '--b', lw=2, label='Lagrange')
        self.ax.scatter(dx, dy, color='red', zorder=5, s=60, label='Puntos')
        self.ax.set_title("Lagrange"); self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.legend(fontsize=8); self.ax.grid(True)
        self.fig.tight_layout(); self.canvas.draw()


# ══════════════════════════════════════════════════════════════════
#   VENTANA: NEWTON-RAPHSON
# ══════════════════════════════════════════════════════════════════

class VentanaNewtonRaphson:
    def __init__(self):
        self.win = ventana_base("Método de Newton-Raphson", ancho=1000, alto=680)
        self._build()

    def _build(self):
        left = tk.Frame(self.win, bg=FONDO, padx=12, pady=12)
        left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(left, text="Newton-Raphson",
                 font=("Arial", 14, "bold"), bg=FONDO, fg=AZUL).pack(pady=(0,10))

        # entradas
        sec = frame_seccion(left, "Parámetros")
        sec.pack(fill=tk.X)

        tk.Label(sec, text="f(x) =", bg=FONDO, font=("Arial", 10)).grid(row=0, column=0, sticky="w", pady=3)
        self.entry_f = tk.Entry(sec, width=22, font=("Arial", 11))
        self.entry_f.insert(0, "x**3 - x - 2")
        self.entry_f.grid(row=0, column=1, padx=4, pady=3)

        tk.Label(sec, text="x₀ =", bg=FONDO, font=("Arial", 10)).grid(row=1, column=0, sticky="w", pady=3)
        self.entry_x0 = tk.Entry(sec, width=10, font=("Arial", 11))
        self.entry_x0.insert(0, "1.5")
        self.entry_x0.grid(row=1, column=1, padx=4, pady=3, sticky="w")

        tk.Label(sec, text="Tolerancia =", bg=FONDO, font=("Arial", 10)).grid(row=2, column=0, sticky="w", pady=3)
        self.entry_tol = tk.Entry(sec, width=10, font=("Arial", 11))
        self.entry_tol.insert(0, "1e-7")
        self.entry_tol.grid(row=2, column=1, padx=4, pady=3, sticky="w")

        tk.Label(sec, text="Máx iteraciones =", bg=FONDO, font=("Arial", 10)).grid(row=3, column=0, sticky="w", pady=3)
        self.entry_max = tk.Entry(sec, width=10, font=("Arial", 11))
        self.entry_max.insert(0, "100")
        self.entry_max.grid(row=3, column=1, padx=4, pady=3, sticky="w")

        bc = tk.Button(left, text="▶  Calcular", command=self._calcular)
        estilo_boton(bc); bc.pack(fill=tk.X, pady=10)

        # resultado
        sec2 = frame_seccion(left, "Raíz encontrada")
        sec2.pack(fill=tk.X)
        self.lbl_result = tk.Label(sec2, text="—", font=("Courier", 11, "bold"),
                                   bg=BLANCO, relief=tk.SUNKEN, padx=4, pady=6)
        self.lbl_result.pack(fill=tk.X)

        # tabla iteraciones
        sec3 = frame_seccion(left, "Iteraciones")
        sec3.pack(fill=tk.BOTH, expand=True, pady=(8,0))

        cols_frame = tk.Frame(sec3, bg=FONDO)
        cols_frame.pack(fill=tk.X)
        for c, h in enumerate(["n", "xₙ", "f(xₙ)", "f'(xₙ)", "xₙ₊₁", "Error"]):
            tk.Label(cols_frame, text=h, bg=AZUL, fg=BLANCO,
                     font=("Arial", 8, "bold"), width=10, relief=tk.FLAT
                     ).grid(row=0, column=c, padx=1, pady=1)

        self.iter_text = tk.Text(sec3, height=14, font=("Courier", 8),
                                 state=tk.DISABLED, bg=BLANCO, relief=tk.SUNKEN)
        self.iter_text.pack(fill=tk.BOTH, expand=True)

        # derecha (gráfica)
        right = tk.Frame(self.win, bg=FONDO, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(6,5))
        self.fig.patch.set_facecolor(FONDO)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _calcular(self):
        func_str = self.entry_f.get().strip()
        try:
            x0  = float(self.entry_x0.get())
            tol = float(self.entry_tol.get())
            mx  = int(self.entry_max.get())
        except ValueError:
            messagebox.showerror("Error", "Verifica los valores numéricos.", parent=self.win)
            return
        try:
            raiz, iters, f, fp = newton_raphson(func_str, x0, tol, mx)
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.win); return

        self.lbl_result.config(text=f"x ≈ {raiz:.10f}")

        # tabla iteraciones
        self.iter_text.config(state=tk.NORMAL)
        self.iter_text.delete("1.0", tk.END)
        for n, xi, fxi, fpxi, xi1, err in iters:
            linea = f"{n:>3}  {xi:>12.7f}  {fxi:>12.7f}  {fpxi:>12.7f}  {xi1:>12.7f}  {err:>12.2e}\n"
            self.iter_text.insert(tk.END, linea)
        self.iter_text.config(state=tk.DISABLED)

        self._graficar(func_str, raiz, iters, f)

    def _graficar(self, func_str, raiz, iters, f):
        self.ax.clear()
        x0_val = float(self.entry_x0.get())
        margen = max(abs(raiz - x0_val) * 1.5, 2)
        t = np.linspace(raiz - margen, raiz + margen, 400)
        try:
            yt = f(t)
        except Exception:
            return

        self.ax.plot(t, yt, 'b-', lw=2, label=f'f(x) = {func_str}')
        self.ax.axhline(0, color='black', lw=0.8)
        self.ax.axvline(raiz, color='red', lw=1.2, linestyle='--', label=f'Raíz ≈ {raiz:.6f}')
        self.ax.scatter([raiz], [0], color='red', zorder=5, s=80)

        # trazar aproximaciones
        xs = [x0_val] + [it[4] for it in iters]
        for xi in xs[:-1]:
            try:
                fxi_val = float(f(xi))
                self.ax.scatter([xi], [fxi_val], color='orange', zorder=4, s=30)
                self.ax.plot([xi, xi], [0, fxi_val], 'orange', lw=0.7, linestyle=':')
            except Exception:
                pass

        self.ax.set_title("Newton-Raphson")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("f(x)")
        self.ax.legend(fontsize=8); self.ax.grid(True)
        self.fig.tight_layout(); self.canvas.draw()


# ══════════════════════════════════════════════════════════════════
#   VENTANA: MULLER
# ══════════════════════════════════════════════════════════════════

class VentanaMuller:
    def __init__(self):
        self.win = ventana_base("Método de Muller", ancho=1000, alto=680)
        self._build()

    def _build(self):
        left = tk.Frame(self.win, bg=FONDO, padx=12, pady=12)
        left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(left, text="Muller",
                 font=("Arial", 14, "bold"), bg=FONDO, fg=AZUL).pack(pady=(0,10))

        # entradas
        sec = frame_seccion(left, "Parámetros")
        sec.pack(fill=tk.X)

        tk.Label(sec, text="f(x) =", bg=FONDO, font=("Arial", 10)).grid(row=0, column=0, sticky="w", pady=3)
        self.entry_f = tk.Entry(sec, width=22, font=("Arial", 11))
        self.entry_f.insert(0, "x**3 - x - 2")
        self.entry_f.grid(row=0, column=1, padx=4, pady=3)

        tk.Label(sec, text="x₀ =", bg=FONDO, font=("Arial", 10)).grid(row=1, column=0, sticky="w", pady=3)
        self.entry_x0 = tk.Entry(sec, width=10, font=("Arial", 11))
        self.entry_x0.insert(0, "0.5")
        self.entry_x0.grid(row=1, column=1, padx=4, pady=3, sticky="w")

        tk.Label(sec, text="x₁ =", bg=FONDO, font=("Arial", 10)).grid(row=2, column=0, sticky="w", pady=3)
        self.entry_x1 = tk.Entry(sec, width=10, font=("Arial", 11))
        self.entry_x1.insert(0, "1")
        self.entry_x1.grid(row=2, column=1, padx=4, pady=3, sticky="w")

        tk.Label(sec, text="x₂ =", bg=FONDO, font=("Arial", 10)).grid(row=3, column=0, sticky="w", pady=3)
        self.entry_x2 = tk.Entry(sec, width=10, font=("Arial", 11))
        self.entry_x2.insert(0, "1.5")
        self.entry_x2.grid(row=3, column=1, padx=4, pady=3, sticky="w")

        tk.Label(sec, text="Tolerancia =", bg=FONDO, font=("Arial", 10)).grid(row=4, column=0, sticky="w", pady=3)
        self.entry_tol = tk.Entry(sec, width=10, font=("Arial", 11))
        self.entry_tol.insert(0, "1e-7")
        self.entry_tol.grid(row=4, column=1, padx=4, pady=3, sticky="w")

        tk.Label(sec, text="Máx iteraciones =", bg=FONDO, font=("Arial", 10)).grid(row=5, column=0, sticky="w", pady=3)
        self.entry_max = tk.Entry(sec, width=10, font=("Arial", 11))
        self.entry_max.insert(0, "100")
        self.entry_max.grid(row=5, column=1, padx=4, pady=3, sticky="w")

#######################################################

        bc = tk.Button(left, text="▶  Calcular", command=self._calcular)
        estilo_boton(bc); bc.pack(fill=tk.X, pady=10)

        # resultado
        sec2 = frame_seccion(left, "Raíz encontrada")
        sec2.pack(fill=tk.X)
        self.lbl_result = tk.Label(sec2, text="—", font=("Courier", 11, "bold"),
                                   bg=BLANCO, relief=tk.SUNKEN, padx=4, pady=6)
        self.lbl_result.pack(fill=tk.X)

        # tabla iteraciones
        sec3 = frame_seccion(left, "Iteraciones")
        sec3.pack(fill=tk.BOTH, expand=True, pady=(8,0))

        cols_frame = tk.Frame(sec3, bg=FONDO)
        cols_frame.pack(fill=tk.X)
        columnas = [
            ("n", 5),
            ("x₀", 11),
            ("x₁", 11),
            ("x₂", 11),
            ("f(x₂)", 11),
            ("x₃", 11),
            ("Error", 11)
        ]
        for c, (h,ancho) in enumerate(columnas):
            tk.Label(cols_frame, text=h, bg=AZUL, fg=BLANCO,
                     font=("Arial", 8, "bold"), width=ancho, relief=tk.FLAT
                     ).grid(row=0, column=c, padx=1, pady=1)

        self.iter_text = tk.Text(sec3, height=14, font=("Courier", 8),
                                 state=tk.DISABLED, bg=BLANCO, relief=tk.SUNKEN)
        self.iter_text.pack(fill=tk.BOTH, expand=True)

        # derecha (gráfica)
        right = tk.Frame(self.win, bg=FONDO, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(6,5))
        self.fig.patch.set_facecolor(FONDO)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def formatear_real_complejo(self, valor, dec=7):
        valor = complex(valor)

        if abs(valor.imag) < 1e-10:
            return f"{valor.real:.{dec}f}"

        signo = "+" if valor.imag >= 0 else "-"
        return f"{valor.real:.{dec}f}{signo}{abs(valor.imag):.{dec}f}j"

    def _calcular(self):
        func_str = self.entry_f.get().strip()
        try:
            x0  = float(self.entry_x0.get())
            x1  = float(self.entry_x1.get())
            x2  = float(self.entry_x2.get())
            tol = float(self.entry_tol.get())
            mx  = int(self.entry_max.get())
        except ValueError:
            messagebox.showerror("Error", "Verifica los valores numéricos.", parent=self.win)
            return
        try:
            raiz, iters, f = muller(func_str, x0, x1, x2, tol, mx)
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.win); 
            return

        self.lbl_result.config(text=f"x ≈ {self.formatear_real_complejo(raiz, 7)}")

        # tabla iteraciones
        self.iter_text.config(state=tk.NORMAL)
        self.iter_text.delete("1.0", tk.END)

        for n, x0_i, x1_i, x2_i, f2_i, x3_i, err in iters:
            linea = (f"{n:>3}  "
                    f"{self.formatear_real_complejo(x0_i):>11}  "
                    f"{self.formatear_real_complejo(x1_i):>10}  "
                    f"{self.formatear_real_complejo(x2_i):>10}  "
                    f"{self.formatear_real_complejo(f2_i):>10}  "
                    f"{self.formatear_real_complejo(x3_i):>10}  "
                    f"{err:>11.2e}\n"
            )
            self.iter_text.insert(tk.END, linea)
        self.iter_text.config(state=tk.DISABLED)

        self._graficar(func_str, raiz, iters, f)

    def _graficar(self, func_str, raiz, iters, f):
        self.ax.clear()
        raiz_c = complex(raiz)

        if abs(raiz_c.imag) > 1e-10:
            self.ax.text(
                0.5, 0.5,
                "La raíz encontrada es compleja.\nNo se puede marcar en una gráfica real.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=11
            )
            self.ax.set_title("Método de Müller")
            self.canvas.draw()
            return
    
        raiz_real = raiz_c.real

        x0_val = float(self.entry_x0.get())
        x1_val = float(self.entry_x1.get())
        x2_val = float(self.entry_x2.get())

        margen = max(
            abs(raiz_real - x0_val) * 1.5,
            abs(raiz_real - x1_val) * 1.5,
            abs(raiz_real - x2_val) * 1.5,
            2
        )

        t = np.linspace(raiz_real - margen, raiz_real + margen, 400)

        try:
            yt = f(t)
            yt = np.real_if_close(yt)
        except Exception:
            return

        self.ax.plot(t, yt, 'b-', lw=2, label=f'f(x) = {func_str}')
        self.ax.axhline(0, color='black', lw=0.8)
        self.ax.axvline(
            raiz_real,
            color='red',
            lw=1.2,
            linestyle='--',
            label=f'Raíz ≈ {raiz_real:.6f}'
        )
        self.ax.scatter([raiz_real], [0], color='red', zorder=5, s=80)

        # trazar aproximaciones
        xs = [x0_val, x1_val, x2_val]
        for it in iters:
            x3 = complex(it[5])
            if abs(x3.imag) < 1e-10:
                xs.append(x3.real)
                
        for xi in xs:
            try:
                fxi_val = f(xi)
                fxi_val = complex(fxi_val)

                if abs(fxi_val.imag) < 1e-10:
                    fxi_val = fxi_val.real
                    self.ax.scatter([xi], [fxi_val], color='orange', zorder=4, s=30)
                    self.ax.plot([xi, xi], [0, fxi_val], 'orange', lw=0.7, linestyle=':')
            except Exception:
                pass

        self.ax.set_title("Muller")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("f(x)")
        self.ax.legend(fontsize=8); self.ax.grid(True)
        self.fig.tight_layout(); self.canvas.draw()


# ══════════════════════════════════════════════════════════════════
#   MENÚ PRINCIPAL
# ══════════════════════════════════════════════════════════════════

class MenuPrincipal:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora de Métodos Numéricos")
        self.root.geometry("500x400")
        self.root.configure(bg=AZUL)
        self.root.resizable(False, False)
        self._build()

    def _build(self):
        # Título
        tk.Label(self.root, text="Métodos Numéricos",
                 font=("Arial", 22, "bold"), bg=AZUL, fg=BLANCO).pack(pady=(40, 4))
        tk.Label(self.root, text="Selecciona un método para comenzar",
                 font=("Arial", 11), bg=AZUL, fg=GRIS).pack(pady=(0, 30))

        # Botones de métodos
        metodos = [
            ("📐  Interpolación de Neville",  self._abrir_neville,  "#3a7ebf"),
            ("📏  Interpolación de Lagrange", self._abrir_lagrange, "#2e6da4"),
            ("🔍  Newton-Raphson (Raíces)",   self._abrir_newton,   "#1a5276"),
            ("🔍  Muller",   self._abrir_muller,   "#10354d"),
        ]
        for txt, cmd, color in metodos:
            btn = tk.Button(self.root, text=txt, command=cmd,
                            font=("Arial", 13, "bold"), bg=color, fg=BLANCO,
                            relief=tk.FLAT, cursor="hand2",
                            activebackground=AZUL2, activeforeground=BLANCO,
                            pady=12, width=30)
            btn.pack(pady=6)

        tk.Label(self.root, text="© Métodos Numéricos",
                 font=("Arial", 8), bg=AZUL, fg=GRIS).pack(side=tk.BOTTOM, pady=8)

    def _abrir_neville(self):
        VentanaNeville()

    def _abrir_lagrange(self):
        VentanaLagrange()

    def _abrir_newton(self):
        VentanaNewtonRaphson()

    def _abrir_muller(self):
        VentanaMuller()


# ══════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = MenuPrincipal(root)
    root.mainloop()
