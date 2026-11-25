"""AULA 2 — LIMITES: propriedades, limites laterais, no infinito, infinitos e continuidade."""
import sympy as sp

from .entrada import t, tex, th
from .fisica import Arremesso, CESTA_D

dt = sp.symbols("Delta_t", real=True)
Q = sp.nsimplify  # usa frações exatas para evitar erro de arredondamento


def fmt(v) -> str:
    """Formata um resultado: ∞, −∞ ou número com até 4 casas."""
    if v is None:
        return "—"
    if v == sp.oo:
        return "+∞"
    if v == -sp.oo:
        return "−∞"
    try:
        return f"{float(v):.4g}".replace(".", ",")
    except TypeError:
        return str(v)


def _lim(expr, var, ponto, lado):
    """Limite de um lado. Para funções por partes, escolhe o ramo válido daquele lado."""
    if isinstance(expr, sp.Piecewise) and ponto.is_finite:
        teste = ponto - sp.Rational(1, 10**6) if lado == "-" else ponto + sp.Rational(1, 10**6)
        for ramo, cond in expr.args:
            if cond == True or bool(cond.subs(var, teste)):
                return sp.limit(ramo, var, ponto, dir=lado)
    return sp.limit(expr, var, ponto, dir=lado)


def calcular(expr, var, ponto, lado: str | None = None) -> dict:
    """Calcula os limites laterais e decide se o limite existe.

    - Laterais iguais e finitos  -> existe
    - Laterais diferentes         -> NÃO existe (salto)
    - Laterais ±∞                 -> limite infinito (não existe como número real)
    """
    ponto = sp.sympify(ponto)
    if ponto in (sp.oo, -sp.oo) or lado in ("-", "+"):
        val = _lim(expr, var, ponto, lado or "+-"[ponto == sp.oo])
        esq = dir_ = val
    else:
        esq = _lim(expr, var, ponto, "-")
        dir_ = _lim(expr, var, ponto, "+")
    r = dict(esq=esq, dir=dir_, existe=False, valor=None, continua=None)
    if esq.has(sp.AccumBounds) or dir_.has(sp.AccumBounds):
        r["motivo"] = "A função oscila sem se aproximar de um valor: o limite não existe."
    elif esq != dir_:
        r["motivo"] = f"Limites laterais diferentes ({fmt(esq)} ≠ {fmt(dir_)}): o limite não existe."
    elif esq.is_infinite:
        r["valor"] = esq
        r["motivo"] = f"A função cresce sem limite ({fmt(esq)}): limite infinito, não existe como número real."
    else:
        r.update(existe=True, valor=esq, motivo=f"Laterais iguais: o limite existe e vale {fmt(esq)}.")
    if ponto.is_finite and lado is None:  # continuidade: f(a) existe e é igual ao limite
        try:
            fa = expr.subs(var, ponto)
            fa = None if fa.has(sp.zoo, sp.nan) or fa.is_finite is False else fa
        except Exception:
            fa = None
        r["f_ponto"] = fa
        r["continua"] = bool(r["existe"] and fa is not None and sp.simplify(fa - r["valor"]) == 0)
    return r


def aproximacoes(expr, var, ponto, passos=(0.1, 0.01, 0.001, 0.0001)):
    """Tabela numérica: valores de f perto do ponto, pela esquerda e pela direita."""
    f = sp.lambdify(var, expr, "math")
    linhas = []
    for h in passos:
        def ok(z):
            try:
                return round(float(f(z)), 5)
            except (ZeroDivisionError, ValueError):
                return None
        linhas.append((h, ok(float(ponto) - h), ok(float(ponto) + h)))
    return linhas


def exemplos(a: Arremesso | None = None) -> list[dict]:
    a = a or Arremesso()
    vy0, g2 = Q(round(a.vy0, 2)), Q(4.9)
    h = Q(a.h0) + vy0 * t - g2 * t**2
    t1 = Q(round(a.t_pico() / 2, 2))  # um instante durante a subida
    g, vq = Q(9.8), Q(6.86)
    quique = sp.Piecewise((-g * t, t < 1), (vq - g * (t - 1), True))
    altura_quique = sp.Piecewise((g2 * (1 - t**2), t < 1), (vq * (t - 1) - g2 * (t - 1)**2, True))
    return [
        dict(titulo="Velocidade instantânea (indeterminação 0/0)",
             conceito="Propriedades dos limites + definição de derivada",
             var=dt, ponto=0,
             expr=sp.expand(h.subs(t, t1 + dt) - h.subs(t, t1)) / dt,
             latex=rf"\lim_{{\Delta t \to 0}} \frac{{h({tex(t1)}+\Delta t) - h({tex(t1)})}}{{\Delta t}}",
             interp=f"Substituir Δt = 0 dá 0/0. Simplificando, o limite é a velocidade vertical da bola "
                    f"no instante t = {float(t1)} s (positiva = ainda subindo)."),
        dict(titulo="Velocidade no quique (limites laterais diferentes)",
             conceito="Limites laterais e inexistência",
             var=t, ponto=1, expr=quique,
             latex=r"\lim_{t \to 1} v(t),\quad v(t)=\begin{cases}-9{,}8\,t & t<1\\ 6{,}86-9{,}8(t-1) & t\ge 1\end{cases}",
             interp="Antes de bater no chão a bola desce a −9,8 m/s; logo depois sobe a +6,86 m/s "
                    "(perde 30% da energia). Como os laterais são diferentes, o limite NÃO existe: há um salto."),
        dict(titulo="Altura no quique (continuidade)",
             conceito="Continuidade",
             var=t, ponto=1, expr=altura_quique,
             latex=r"\lim_{t \to 1} h(t),\quad h(t)=\begin{cases}4{,}9(1-t^2) & t<1\\ 6{,}86(t-1)-4{,}9(t-1)^2 & t\ge 1\end{cases}",
             interp="A altura vale 0 dos dois lados e h(1) = 0: a função é CONTÍNUA. "
                    "A bola não 'teleporta' — só a velocidade muda de repente."),
        dict(titulo="Ângulo tendendo a 90° (limite infinito)",
             conceito="Limite infinito / lateral",
             var=th, ponto=sp.pi / 2, lado="-",
             expr=Q(CESTA_D) / (Q(round(a.v, 2)) * sp.cos(th)),
             latex=rf"\lim_{{\theta \to \pi/2^-}} \frac{{{CESTA_D}}}{{{round(a.v, 2)}\cos\theta}}",
             interp="Tempo para a bola chegar à cesta. Quanto mais 'na vertical' o arremesso, menor a "
                    "velocidade horizontal: o tempo cresce sem limite (+∞). Jogando a 90° a bola nunca chega."),
        dict(titulo="Resistência do ar no longo prazo (limite no infinito)",
             conceito="Limite no infinito",
             var=t, ponto=sp.oo,
             expr=Q(round(a.v, 2)) * sp.exp(-Q(0.05) * t),
             latex=rf"\lim_{{t \to \infty}} {round(a.v, 2)}\,e^{{-0{{,}}05t}}",
             interp="Com o tempo a velocidade tende a 0, mas nunca fica negativa: a assíntota horizontal é v = 0."),
    ]
