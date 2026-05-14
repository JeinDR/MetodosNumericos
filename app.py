#!/usr/bin/env python
# coding: utf-8

import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import Symbol, Poly, expand


# ─── Lógica de Neville ────────────────────────────────────────────────────────

def calcPoly(expr):
    x = Symbol('x')
    try:
        p = Poly(expand(expr), x)
        coeffs = [float(c) for c in p.all_coeffs()]
        return np.poly1d(coeffs)
    except Exception:
        return np.poly1d([0])


def poly_to_string(poly):
    polyString = ""
    power = len(poly.coefficients) - 1
    for coef in poly.coefficients:
        if coef != 0:
            if power == 0:
                polyString += "{:.4f}".format(coef)
            else:
                polyString += "({:.4f})x^{} + ".format(coef, power)
        power -= 1
    return polyString or "0"


def neville(datax, datay):
    x = Symbol('x')
    n = len(datax)
    poly = [datay[i] for i in range(n)]
    iteraciones = []

    for k in range(1, n):
        for i in range(n - k):
            poly[i] = (
                (x - datax[i + k]) * poly[i] + (datax[i] - x) * poly[i + 1]
            ) / (datax[i] - datax[i + k])
        resultado = calcPoly(poly[0])
        iteraciones.append((k, poly_to_string(resultado)))

    return calcPoly(poly[0]), iteraciones


# ─── Interfaz Gráfica ─────────────────────────────────────────────────────────

class NevilleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interpolación de Neville")
        self.root.geometry("950x680")
        self.root.configure(bg="#f0f0f0")
        self.filas = []
        self._build_ui()

    def _build_ui(self):
        # ── Panel izquierdo ──────────────────────────────────────────
        left = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(left, text="Interpolación de Neville",
                 font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=(0, 10))

        # Cabecera de tabla
        header = tk.Frame(left, bg="#f0f0f0")
        header.pack()
        tk.Label(header, text="X", width=10, font=("Arial", 11, "bold"),
                 bg="#4a90d9", fg="white").grid(row=0, column=0, padx=2, pady=2)
        tk.Label(header, text="Y", width=10, font=("Arial", 11, "bold"),
                 bg="#4a90d9", fg="white").grid(row=0, column=1, padx=2, pady=2)

        # Frame con scroll para los puntos
        self.table_frame = tk.Frame(left, bg="#f0f0f0")
        self.table_frame.pack()
        self.next_row = 0

        # 4 filas iniciales
        for _ in range(4):
            self._agregar_fila()

        # Botones
        btn_frame = tk.Frame(left, bg="#f0f0f0")
        btn_frame.pack(pady=8)

        tk.Button(btn_frame, text="+ Agregar punto", command=self._agregar_fila,
                  bg="#5cb85c", fg="white", font=("Arial", 10),
                  relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=4)

        tk.Button(btn_frame, text="Limpiar", command=self._limpiar,
                  bg="#d9534f", fg="white", font=("Arial", 10),
                  relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=4)

        tk.Button(left, text="▶  Calcular e Interpolar",
                  command=self._calcular,
                  bg="#4a90d9", fg="white", font=("Arial", 12, "bold"),
                  relief=tk.FLAT, padx=10, pady=6).pack(pady=10, fill=tk.X)

        # Resultado
        tk.Label(left, text="Polinomio resultante:",
                 font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w")
        self.result_label = tk.Label(left, text="—", wraplength=280,
                                     font=("Arial", 9), bg="#ffffff",
                                     relief=tk.SUNKEN, padx=6, pady=6)
        self.result_label.pack(fill=tk.X, pady=4)

        # Iteraciones
        tk.Label(left, text="Iteraciones:",
                 font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w", pady=(8, 0))
        self.iter_text = tk.Text(left, height=10, width=38, font=("Courier", 8),
                                 state=tk.DISABLED, bg="#ffffff", relief=tk.SUNKEN)
        self.iter_text.pack(fill=tk.X)

        # ── Panel derecho (gráfica) ───────────────────────────────────
        right = tk.Frame(self.root, bg="#f0f0f0")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.fig.patch.set_facecolor("#f0f0f0")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _agregar_fila(self):
        row = self.next_row
        ex = tk.Entry(self.table_frame, width=10, font=("Arial", 10))
        ey = tk.Entry(self.table_frame, width=10, font=("Arial", 10))
        ex.grid(row=row, column=0, padx=2, pady=2)
        ey.grid(row=row, column=1, padx=2, pady=2)

        btn_del = tk.Button(self.table_frame, text="✕", fg="red",
                            bg="#f0f0f0", relief=tk.FLAT,
                            command=lambda p=(ex, ey): self._eliminar_fila(p))
        btn_del.grid(row=row, column=2, padx=2)

        self.filas.append((ex, ey))
        self.next_row += 1

    def _eliminar_fila(self, par):
        if len(self.filas) <= 2:
            messagebox.showwarning("Aviso", "Se necesitan al menos 2 puntos.")
            return
        ex, ey = par
        for widget in self.table_frame.grid_slaves():
            if widget == ex or widget == ey:
                widget.grid_forget()
                widget.destroy()
        self.filas.remove(par)

    def _limpiar(self):
        for ex, ey in self.filas:
            ex.delete(0, tk.END)
            ey.delete(0, tk.END)
        self.result_label.config(text="—")
        self._set_iter_text("")
        self.ax.clear()
        self.canvas.draw()

    def _set_iter_text(self, texto):
        self.iter_text.config(state=tk.NORMAL)
        self.iter_text.delete("1.0", tk.END)
        self.iter_text.insert(tk.END, texto)
        self.iter_text.config(state=tk.DISABLED)

    def _calcular(self):
        datax, datay = [], []
        for ex, ey in self.filas:
            sx, sy = ex.get().strip(), ey.get().strip()
            if sx and sy:
                try:
                    datax.append(float(sx))
                    datay.append(float(sy))
                except ValueError:
                    messagebox.showerror("Error", f"Valor inválido: '{sx}' o '{sy}'")
                    return

        if len(datax) < 2:
            messagebox.showwarning("Aviso", "Ingresa al menos 2 puntos.")
            return

        if len(set(datax)) != len(datax):
            messagebox.showerror("Error", "Los valores de X deben ser únicos.")
            return

        try:
            poly, iteraciones = neville(datax, datay)
        except Exception as e:
            messagebox.showerror("Error en cálculo", str(e))
            return

        # Mostrar resultado
        self.result_label.config(text=poly_to_string(poly))

        # Mostrar iteraciones
        iter_str = ""
        for k, p in iteraciones:
            iter_str += f"Iter {k}: {p}\n"
        self._set_iter_text(iter_str)

        # Graficar
        self.ax.clear()
        x_min, x_max = min(datax), max(datax)
        margen = (x_max - x_min) * 0.2 or 1
        t = np.linspace(x_min - margen, x_max + margen, 300)

        self.ax.plot(t, poly(t), '--g', linewidth=2,
                     label=f'Neville: {poly_to_string(poly)}')
        self.ax.scatter(datax, datay, color='blue', zorder=5,
                        s=60, label='Puntos ingresados')
        self.ax.set_title("Interpolación de Neville", fontsize=12)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend(fontsize=7)
        self.ax.grid(True)
        self.fig.tight_layout()
        self.canvas.draw()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = NevilleApp(root)
    root.mainloop()