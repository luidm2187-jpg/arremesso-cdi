"""AULA 1 — FUNÇÕES: tipos, domínio, imagem, entradas/saídas, tabela e gráfico.

Cada função do catálogo modela uma parte do arremesso.
"""
import numpy as np
import sympy as sp
from sympy.calculus.util import continuous_domain, function_range

from .entrada import t, th, x
from .fisica import CESTA_D, CESTA_H, G, Arremesso

v = sp.symbols("v", positive=True)


def r2(n):
    return sp.Float(round(float(n), 2))


def catalogo(a: Arremesso | None = None) -> list[dict]:
    """8 funções de tipos distintos, com coeficientes tirados do arremesso atual."""
    a = a or Arremesso()
    k = sp.Float(0.05)  # coeficiente de arrasto do ar (modelo exponencial ilustrativo)
    return [
        dict(nome="Altura do aro", tipo="Constante", var=x, expr=sp.Float(CESTA_H, 3),
             ctx=(0, a.dist), ent="x: posição horizontal (m)", sai="altura do aro (m)",
             interp="O aro está sempre a 3,05 m, não importa a posição: a saída não depende da entrada."),
        dict(nome="Posição horizontal", tipo="Linear", var=t, expr=r2(a.vx) * t,
             ctx=(0, round(a.t_chao(), 2)), ent="t: tempo (s)", sai="x: distância percorrida (m)",
             interp=f"A bola anda {a.vx:.2f} m a cada segundo na horizontal (velocidade constante, sem resistência do ar)."),
        dict(nome="Velocidade vertical", tipo="Afim", var=t, expr=r2(a.vy0) - sp.Float(G, 3) * t,
             ctx=(0, round(a.t_chao(), 2)), ent="t: tempo (s)", sai="vy: velocidade vertical (m/s)",
             interp="Começa positiva (subindo), zera no ponto mais alto e fica negativa (descendo). Coef. angular = −g."),
        dict(nome="Altura da bola", tipo="Quadrática", var=t, expr=a.h0 + r2(a.vy0) * t - sp.Float(G / 2, 3) * t**2,
             ctx=(0, round(a.t_chao(), 2)), ent="t: tempo (s)", sai="h: altura (m)",
             interp=f"Parábola com concavidade para baixo (a < 0): o vértice é a altura máxima ≈ {a.h_max():.2f} m."),
        dict(nome="Erro até o centro do aro", tipo="Modular", var=x, expr=sp.Abs(x - sp.Float(a.dist, 3)),
             ctx=(0, round(2 * a.dist, 2)), ent="x: onde a bola cruza o aro (m)", sai="erro (m)",
             interp="Distância é sempre ≥ 0: errar 0,3 m antes ou depois do aro dá o mesmo erro."),
        dict(nome="Velocidade com resistência do ar", tipo="Exponencial", var=t, expr=r2(a.v) * sp.exp(-k * t),
             ctx=(0, 10), ent="t: tempo (s)", sai="velocidade (m/s)",
             interp="O ar 'rouba' 5% da velocidade por segundo: decai sem nunca chegar a zero."),
        dict(nome="Tempo para a velocidade cair a v", tipo="Logarítmica", var=v, expr=sp.log(r2(a.v) / v) / k,
             ctx=(0.5, round(a.v, 2)), ent="v: velocidade desejada (m/s)", sai="tempo (s)",
             interp="É a inversa da exponencial: quanto menor a velocidade pedida, mais tempo — e cresce devagar (log)."),
        dict(nome="Alcance pelo ângulo", tipo="Trigonométrica", var=th,
             expr=r2(a.v**2 / G) * sp.sin(2 * th), ctx=(0, sp.pi / 2),
             ent="θ: ângulo (rad)", sai="alcance no chão (m)",
             interp="R(θ) = v²·sen(2θ)/g: máximo em θ = 45° (π/4), e zero em 0° e 90°."),
    ]


def dominio(expr, var):
    dom = continuous_domain(expr, var, sp.S.Reals)
    if var.is_positive:
        dom = dom.intersect(sp.Interval.open(0, sp.oo))
    return dom


def imagem(expr, var, dom):
    try:
        return function_range(expr, var, dom)
    except Exception:
        return None


def analisar(f: dict) -> dict:
    """Domínio e imagem matemáticos (em ℝ) e no contexto físico do arremesso."""
    e, var = f["expr"], f["var"]
    dom = dominio(e, var)
    ctx = sp.Interval(*f["ctx"])
    return dict(dominio=dom, imagem=imagem(e, var, dom),
                dominio_ctx=ctx, imagem_ctx=imagem(e, var, ctx))


def tabela(f: dict, n: int = 6):
    """Tabela de entradas → saídas no domínio do contexto."""
    fn = sp.lambdify(f["var"], f["expr"], "numpy")
    xs = np.linspace(*map(float, f["ctx"]), n)
    ys = np.broadcast_to(fn(xs), xs.shape)
    return [(round(float(a), 3), round(float(b), 3)) for a, b in zip(xs, ys)]


def pontos(f: dict, n: int = 200):
    fn = sp.lambdify(f["var"], f["expr"], "numpy")
    xs = np.linspace(*map(float, f["ctx"]), n)
    return xs, np.broadcast_to(fn(xs), xs.shape).astype(float)


def arredondar(obj, casas: int = 3):
    """Arredonda os números de uma expressão/conjunto para exibição."""
    if obj is None:
        return None
    return obj.xreplace({n: sp.Float(round(float(n), casas)) for n in obj.atoms(sp.Float)})


def texto_conjunto(c) -> str:
    """Conjunto em notação de sala de aula: ℝ, (0, +∞), [−5.74, 5.74], {3.05}."""
    if c is None:
        return "não determinada"
    if c == sp.S.Reals:
        return "ℝ"
    n = lambda v: "+∞" if v == sp.oo else "−∞" if v == -sp.oo else f"{float(v):.4g}"
    if isinstance(c, sp.Interval):
        return f"{'(' if c.left_open or c.start == -sp.oo else '['}{n(c.start)}, {n(c.end)}{')' if c.right_open or c.end == sp.oo else ']'}"
    if isinstance(c, sp.FiniteSet):
        return "{" + ", ".join(n(v) for v in c) + "}"
    if isinstance(c, sp.Union):
        return " ∪ ".join(texto_conjunto(a) for a in c.args)
    return str(c)


def classificar(expr, var) -> str:
    """Identifica o tipo de uma função digitada pelo usuário."""
    if not expr.has(var):
        return "Constante"
    if expr.has(sp.Abs):
        return "Modular"
    if expr.has(sp.sin, sp.cos, sp.tan):
        return "Trigonométrica"
    if expr.has(sp.log):
        return "Logarítmica"
    if expr.has(sp.exp) or any(p.exp.has(var) for p in expr.atoms(sp.Pow)):
        return "Exponencial"
    if expr.is_polynomial(var):
        g = sp.degree(expr, var)
        if g == 1:
            return "Linear" if expr.subs(var, 0) == 0 else "Afim"
        if g == 2:
            return "Quadrática"
        return f"Polinomial (grau {g})"
    return "Outra"
