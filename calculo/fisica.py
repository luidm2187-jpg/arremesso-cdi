"""Modelo físico do arremesso (lançamento oblíquo, sem resistência do ar).

    x(t) = v·cos(θ)·t                       -> função afim/linear
    h(t) = h0 + v·sen(θ)·t − (g/2)·t²       -> função quadrática
"""
from dataclasses import dataclass
import math
import numpy as np
import sympy as sp

from .entrada import EntradaInvalida, t, th

G = 9.8          # gravidade (m/s²)
H0 = 2.0         # altura da mão no lançamento (m)
CESTA_H = 3.05   # altura oficial do aro (m)
CESTA_D = 4.6    # distância horizontal até o centro do aro, lance livre (m)
RAIO_ARO = 0.23  # raio do aro (m) — a bola é tratada como um ponto (modelo simplificado)


@dataclass
class Arremesso:
    v: float = 7.5          # velocidade inicial (m/s)
    ang: float = 52.0       # ângulo de lançamento (graus)
    h0: float = H0
    dist: float = CESTA_D

    def __post_init__(self):
        if not (0 < self.v <= 30):
            raise EntradaInvalida("A velocidade deve estar entre 0 e 30 m/s.")
        if not (0 < self.ang < 90):
            raise EntradaInvalida("O ângulo deve estar entre 0° e 90° (sem incluir).")
        if not (0 <= self.h0 <= 3):
            raise EntradaInvalida("A altura de lançamento deve estar entre 0 e 3 m.")
        if not (0.5 <= self.dist <= 15):
            raise EntradaInvalida("A distância até a cesta deve estar entre 0,5 e 15 m.")

    # componentes da velocidade (trigonometria)
    @property
    def rad(self): return math.radians(self.ang)
    @property
    def vx(self): return self.v * math.cos(self.rad)
    @property
    def vy0(self): return self.v * math.sin(self.rad)

    # funções do movimento (SymPy)
    def x_t(self): return sp.Float(self.vx) * t
    def h_t(self): return self.h0 + sp.Float(self.vy0) * t - sp.nsimplify(G / 2) * t**2
    def vy_t(self): return sp.diff(self.h_t(), t)

    # grandezas
    def t_pico(self): return self.vy0 / G                       # h'(t) = 0
    def h_max(self): return self.h0 + self.vy0**2 / (2 * G)
    def t_chao(self):                                           # h(t) = 0, raiz positiva
        return (self.vy0 + math.sqrt(self.vy0**2 + 2 * G * self.h0)) / G
    def alcance(self): return self.vx * self.t_chao()

    def resultado(self) -> dict:
        """Verifica a cesta: a bola precisa DESCER pela altura do aro com x dentro do aro."""
        r = {"acertou": False, "t": None, "x": None, "erro": None, "h_max": self.h_max()}
        dh = self.h_max() - CESTA_H
        if dh < 0:
            r["msg"] = f"A bola não alcançou a altura da cesta (faltaram {abs(dh):.2f} m)."
            return r
        tc = (self.vy0 + math.sqrt(self.vy0**2 - 2 * G * (CESTA_H - self.h0))) / G  # raiz descendente
        xc = self.vx * tc
        erro = abs(xc - self.dist)                               # função modular
        r.update(t=tc, x=xc, erro=erro, acertou=erro <= RAIO_ARO)
        if r["acertou"]:
            r["msg"] = f"CESTA! A bola passou a {erro:.2f} m do centro do aro."
        else:
            lado = "longa (passou do aro)" if xc > self.dist else "curta (antes do aro)"
            r["msg"] = f"Errou: bola {lado}, {erro:.2f} m do centro do aro."
        return r

    def pontos(self, n: int = 120):
        """Pontos (x, h) para o gráfico, até a bola tocar o chão."""
        ts = np.linspace(0, self.t_chao(), n)
        return ts, self.vx * ts, self.h0 + self.vy0 * ts - G / 2 * ts**2


def v2_necessaria(h0=H0, dist=CESTA_D):
    """v²(θ) necessária para a bola passar no centro do aro, em função do ângulo θ (rad).

    Vem de h(x) = h0 + x·tg θ − g·x² / (2 v² cos² θ) com x = dist e h = CESTA_H.
    """
    dh = CESTA_H - h0
    return G * dist**2 / (2 * sp.cos(th)**2 * (dist * sp.tan(th) - dh))


def angulo_otimo(h0=H0, dist=CESTA_D):
    """Ângulo que exige a MENOR velocidade: resolve d(v²)/dθ = 0 (aplicação de derivada)."""
    v2 = v2_necessaria(h0, dist)
    dv2 = sp.diff(v2, th)
    chute = math.pi / 4 + 0.5 * math.atan((CESTA_H - h0) / dist)
    th_opt = float(sp.nsolve(dv2, th, chute))
    v_min = math.sqrt(float(v2.subs(th, th_opt)))
    return math.degrees(th_opt), v_min, v2, dv2
