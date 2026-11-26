"""AULA 3 — DERIVADAS: motor próprio que aplica as regras de derivação passo a passo.

Regras implementadas: constante, potência, soma/subtração, constante × função,
produto, quociente, cadeia, exponencial, logarítmica e trigonométricas.
O resultado é conferido com sp.diff (validação).
"""
import sympy as sp

from .entrada import t, tex, th
from .fisica import Arremesso

L = tex
Q = sp.nsimplify

# derivada externa f'(u) das funções elementares
TABELA = {
    sp.sin: (lambda u: sp.cos(u), "Trigonométrica"),
    sp.cos: (lambda u: -sp.sin(u), "Trigonométrica"),
    sp.tan: (lambda u: 1 / sp.cos(u)**2, "Trigonométrica"),
    sp.exp: (lambda u: sp.exp(u), "Exponencial"),
    sp.log: (lambda u: 1 / u, "Logarítmica"),
}


def derivar(expr, var, passos: list | None = None):
    """Deriva `expr` em relação a `var`, registrando (regra, latex) em `passos`."""
    passos = [] if passos is None else passos
    i = len(passos)
    passos.append(None)  # reserva o lugar para manter a ordem de cima para baixo
    d = lambda e: rf"\frac{{d}}{{d{L(var)}}}\left[{L(e)}\right]"

    if not expr.has(var):
        regra, res, como = "Constante", sp.S.Zero, "a derivada de uma constante é 0"
    elif expr == var:
        regra, res, como = "Potência", sp.S.One, r"x^1 \to 1\cdot x^0 = 1"
    elif expr.is_Add:
        regra, como = "Soma/Subtração", "deriva termo a termo"
        res = sp.Add(*[derivar(a, var, passos) for a in expr.args])
    elif expr.is_Mul and expr.as_independent(var)[0] != 1:
        c, f = expr.as_independent(var)
        regra, como = "Constante × função", rf"(c\cdot f)' = c\cdot f',\ c = {L(c)}"
        res = c * derivar(f, var, passos)
    elif expr.is_Mul and sp.fraction(expr)[1].has(var):
        n, m = sp.fraction(expr)
        dn, dm = derivar(n, var, passos), derivar(m, var, passos)
        regra, como = "Quociente", r"\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}"
        res = (dn * m - n * dm) / m**2
    elif expr.is_Mul:
        u, w = expr.args[0], sp.Mul(*expr.args[1:])
        du, dw = derivar(u, var, passos), derivar(w, var, passos)
        regra, como = "Produto", r"(u\cdot v)' = u'v + uv'"
        res = du * w + u * dw
    elif expr.is_Pow and not expr.exp.has(var):
        b, n = expr.args
        if b == var:
            regra, como, res = "Potência", rf"(x^n)' = n\,x^{{n-1}},\ n = {L(n)}", n * var**(n - 1)
        else:
            db = derivar(b, var, passos)
            regra, como = "Cadeia + Potência", rf"(u^n)' = n\,u^{{n-1}}\cdot u',\ u = {L(b)}"
            res = n * b**(n - 1) * db
    elif expr.is_Pow and not expr.base.has(var):
        a, u = expr.args
        du = derivar(u, var, passos) if u != var else sp.S.One
        regra = "Exponencial" + (" + Cadeia" if u != var else "")
        como, res = rf"(a^u)' = a^u \ln a\cdot u'", expr * sp.log(a) * du
    elif isinstance(expr, sp.Function) and expr.func in TABELA:
        u = expr.args[0]
        externa, nome = TABELA[expr.func]
        if u == var:
            regra, como, res = nome, "derivada tabelada", externa(u)
        else:
            du = derivar(u, var, passos)
            regra, como = f"Cadeia ({nome})", rf"[f(u)]' = f'(u)\cdot u',\ u = {L(u)}"
            res = externa(u) * du
    else:
        regra, como, res = "Geral (SymPy)", "regra fora do escopo do motor", sp.diff(expr, var)

    passos[i] = dict(regra=regra, como=como, latex=f"{d(expr)} = {L(res)}")
    return res


def resolver(expr, var) -> dict:
    """Deriva, simplifica e valida contra o SymPy."""
    passos = []
    bruto = derivar(expr, var, passos)
    final = sp.simplify(bruto)
    if final.is_polynomial(var):
        final = sp.expand(final)
    ok = sp.simplify(final - sp.diff(expr, var)) == 0
    regras = sorted({p["regra"] for p in passos})
    return dict(resultado=final, passos=passos, regras=regras, validado=ok)


def exemplos(a: Arremesso | None = None) -> list[dict]:
    """Uma regra por exemplo, todas no contexto do arremesso."""
    a = a or Arremesso()
    vy0, g2, h0 = Q(round(a.vy0, 2)), Q(4.9), Q(a.h0)
    R = Q(round(a.v**2 / 9.8, 2))  # v²/g
    tp = round(a.vy0 / 9.8, 3)
    return [
        dict(regra="Constante", var=t, expr=Q(3.05),
             interp="A altura do aro não muda com o tempo: taxa de variação zero."),
        dict(regra="Potência", var=t, expr=g2 * t**2,
             interp="A queda causada pela gravidade cresce com t²; sua taxa é 9,8·t (cada vez mais rápida)."),
        dict(regra="Soma/Subtração", var=t, expr=h0 + vy0 * t - g2 * t**2,
             interp=f"h'(t) é a VELOCIDADE vertical. h'(t) = 0 em t ≈ {tp} s: é o ponto mais alto "
                    f"(≈ {a.h_max():.2f} m)."),
        dict(regra="Produto", var=th, expr=2 * R * sp.sin(th) * sp.cos(th),
             interp="Alcance R(θ) = (v²/g)·2·senθ·cosθ. R'(θ) = 0 quando cos²θ = sen²θ → θ = 45°: alcance máximo."),
        dict(regra="Quociente", var=th, expr=sp.sin(th) / sp.cos(th),
             interp="senθ/cosθ é a inclinação inicial da trajetória. A derivada 1/cos²θ é sempre positiva: "
                    "aumentar o ângulo sempre deixa o arremesso mais 'em pé'."),
        dict(regra="Cadeia", var=th, expr=R * sp.sin(2 * th),
             interp="Mesmo alcance escrito como sen(2θ): pela cadeia, R'(θ) = 2·(v²/g)·cos(2θ), "
                    "que zera em 2θ = 90° → θ = 45°. Confirma o resultado da regra do produto."),
        dict(regra="Exponencial + Cadeia", var=t, expr=Q(round(a.v, 2)) * sp.exp(-Q(0.05) * t),
             interp="Com resistência do ar, a derivada (negativa) é a desaceleração: a bola perde velocidade "
                    "mais rápido no começo."),
    ]


def tangente(expr, var, x0):
    """Reta tangente em x0: y = f(x0) + f'(x0)(x − x0)."""
    m = sp.diff(expr, var).subs(var, x0)
    return sp.expand(expr.subs(var, x0) + m * (var - x0)), m
