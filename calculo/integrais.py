"""AULA 4 — INTEGRAIS: motor próprio que aplica as regras de integração passo a passo.

Regras implementadas: constante, potência (inclui 1/x → ln|x|), soma/subtração,
constante × função, exponencial, trigonométricas e substituição (u = g(x)).
Integrais definidas pelo Teorema Fundamental do Cálculo: F(b) − F(a).
O resultado é conferido derivando (F' = f) e com sp.integrate (validação).
"""
import math
import sympy as sp

from .entrada import t, tex, th
from .fisica import Arremesso

L = tex
Q = sp.nsimplify
u = sp.Symbol("u")

# primitivas tabeladas: ∫ f(u) du
TABELA = {
    sp.sin: (lambda z: -sp.cos(z), "Trigonométrica"),
    sp.cos: (lambda z: sp.sin(z), "Trigonométrica"),
    sp.exp: (lambda z: sp.exp(z), "Exponencial"),
}


class NaoIntegravel(Exception):
    pass


def _substituicao(expr, var, passos):
    """Tenta u = g(x) quando expr = k · f(g(x)) · g'(x)."""
    candidatos = []
    for f in sp.Mul.make_args(expr):
        if isinstance(f, sp.Function) and f.func in TABELA and f.args[0] != var:
            candidatos.append((f.args[0], f.func(u), f))        # sen(g), cos(g), e^g
        elif f.is_Pow and not f.exp.has(var) and f.base != var:
            candidatos.append((f.base, u**f.exp, f))             # g^n
        if f.has(var) and f != var:
            candidatos.append((f, u, f))                         # o próprio fator é u
    for g, externa, fator in candidatos:
        if not g.has(var):
            continue
        dg = sp.diff(g, var)
        k = sp.simplify(expr / fator / dg)
        if k.has(var) or k == 0:
            continue
        Fu = integrar(externa, u, passos)
        res = k * Fu.subs(u, g)
        return res, rf"u = {L(g)},\ du = {L(dg)}\,d{L(var)} \Rightarrow \int {L(k * externa)}\,du"
    return None


def integrar(expr, var, passos: list | None = None):
    """Primitiva de `expr` (sem +C), registrando (regra, latex) em `passos`."""
    passos = [] if passos is None else passos
    i = len(passos)
    passos.append(None)
    I = lambda e: rf"\int {L(e)}\,d{L(var)}"

    if not expr.has(var):
        regra, como, res = "Constante", r"\int k\,dx = kx", expr * var
    elif expr == var:
        regra, como, res = "Potência", r"\int x\,dx = \frac{x^2}{2}", var**2 / 2
    elif expr.is_Add:
        regra, como = "Soma/Subtração", "integra termo a termo"
        res = sp.Add(*[integrar(a, var, passos) for a in expr.args])
    elif expr.is_Mul and expr.as_independent(var)[0] != 1:
        c, f = expr.as_independent(var)
        regra, como = "Constante × função", rf"\int c\,f = c\int f,\ c = {L(c)}"
        res = c * integrar(f, var, passos)
    elif expr.is_Pow and expr.base == var and not expr.exp.has(var):
        n = expr.exp
        if n == -1:
            regra, como, res = "Potência (n = −1)", r"\int x^{-1}dx = \ln|x|", sp.log(sp.Abs(var))
        else:
            regra, como, res = "Potência", rf"\int x^n dx = \frac{{x^{{n+1}}}}{{n+1}},\ n = {L(n)}", var**(n + 1) / (n + 1)
    elif expr.is_Pow and not expr.base.has(var) and expr.exp == var:
        a = expr.base
        regra, como, res = "Exponencial", r"\int a^x dx = \frac{a^x}{\ln a}", expr / sp.log(a)
    elif isinstance(expr, sp.Function) and expr.func in TABELA and expr.args[0] == var:
        prim, regra = TABELA[expr.func]
        como, res = "primitiva tabelada", prim(var)
    else:
        sub = _substituicao(expr, var, passos)
        if sub:
            res, como = sub
            regra = "Substituição"
        else:
            res = sp.integrate(expr, var)
            if res.has(sp.Integral):
                del passos[i:]
                raise NaoIntegravel("Não há primitiva elementar para essa função.")
            regra, como = "Geral (SymPy)", "regra fora do escopo do motor"

    passos[i] = dict(regra=regra, como=como, latex=f"{I(expr)} = {L(res)}")
    return res


def _mesma_funcao(f, g, var) -> bool:
    """f ≡ g? Tenta simplificar; se não der, compara em vários pontos."""
    if sp.simplify(f - g) == 0:
        return True
    pts = [-2.3, -0.7, 0.4, 1.3, 2.9]
    vals = []
    for p in pts:
        try:
            a, b = complex(f.subs(var, p)), complex(g.subs(var, p))
            vals.append(abs(a - b) < 1e-9)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return bool(vals) and all(vals)


def resolver(expr, var, a=None, b=None) -> dict:
    """Indefinida (a, b = None) ou definida em [a, b]. Valida derivando e com sp.integrate."""
    passos = []
    F = sp.simplify(integrar(expr, var, passos))
    if F.is_polynomial(var):
        F = sp.expand(F)
    ok = _mesma_funcao(sp.diff(F, var), expr, var)
    r = dict(primitiva=F, passos=passos, regras=sorted({p["regra"] for p in passos}), validado=ok, valor=None)
    if a is not None and b is not None:
        a, b = sp.sympify(a), sp.sympify(b)
        if a == b:
            r.update(valor=sp.S.Zero, Fa=F.subs(var, a), Fb=F.subs(var, a))
            return r
        lo, hi = (a, b) if a < b else (b, a)
        # pontos onde f não é contínua dentro de (lo, hi): divide a integral neles
        fora = sp.Interval.open(lo, hi) - sp.calculus.util.continuous_domain(expr, var, sp.Interval(lo, hi))
        cortes = sorted(fora) if isinstance(fora, sp.FiniteSet) else []
        if fora != sp.S.EmptySet and not cortes:
            raise NaoIntegravel("A função não é contínua num intervalo inteiro dentro de [a, b].")
        nos = [lo, *cortes, hi]
        total = sp.S.Zero
        for p0, p1 in zip(nos, nos[1:]):
            parte = sp.limit(F, var, p1, "-") - sp.limit(F, var, p0, "+")
            if parte.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
                onde = f" (descontinuidade em {', '.join(map(str, cortes))})" if cortes else ""
                raise NaoIntegravel(f"Integral imprópria divergente: a área é infinita{onde}.")
            total += parte
        Fa, Fb = sp.limit(F, var, a, "+" if a < b else "-"), sp.limit(F, var, b, "-" if a < b else "+")
        val = sp.simplify(total if a < b else -total)
        if val.is_real is False or not sp.N(val).is_real:
            raise NaoIntegravel("O resultado não é real: a função não é real em todo o intervalo.")
        ref = sp.integrate(expr, (var, a, b))
        r.update(valor=val, Fa=Fa, Fb=Fb,
                 validado=ok and math.isclose(float(val), float(ref), rel_tol=1e-9, abs_tol=1e-9))
    return r


def exemplos(a: Arremesso | None = None) -> list[dict]:
    """Uma regra por exemplo, todas no contexto do arremesso."""
    a = a or Arremesso()
    vy0, g = Q(round(a.vy0, 2)), Q(9.8)
    tc = Q(round(a.resultado()["t"] or a.t_pico(), 3))
    v, ang = Q(round(a.v, 2)), sp.rad(Q(round(a.ang)))
    R = Q(round(a.v**2 / 9.8, 2))
    return [
        dict(regra="Constante", var=t, expr=-g, a=None, b=None,
             interp="Integrar a aceleração (−9,8) dá a velocidade: v(t) = −9,8t + C, onde C = v₀·senθ."),
        dict(regra="Potência", var=t, expr=g * t, a=0, b=1,
             interp="Quanto a gravidade 'puxa' a bola para baixo no 1º segundo: ∫ 9,8t dt = 4,9 m."),
        dict(regra="Soma/Subtração", var=t, expr=vy0 - g * t, a=0, b=tc,
             interp=f"A integral da velocidade vertical até t = {float(tc)} s é o deslocamento vertical. "
                    f"Somado à altura inicial ({a.h0} m) deve dar ≈ 3,05 m — a altura do aro."),
        dict(regra="Exponencial", var=t, expr=v * sp.exp(-Q(0.05) * t), a=0, b=1,
             interp="Distância percorrida no 1º segundo com resistência do ar — um pouco menor que sem ar."),
        dict(regra="Trigonométrica", var=th, expr=v * sp.cos(th), a=0, b=ang,
             interp=f"A integral de v·cosθ de 0 até {round(a.ang)}° dá v·senθ: exatamente a velocidade "
                    f"vertical inicial (≈ {a.vy0:.2f} m/s)."),
        dict(regra="Substituição", var=th, expr=R * sp.sin(2 * th), a=0, b=sp.pi / 2,
             interp="Com u = 2θ: soma dos alcances de 0° a 90°. Dividindo por π/2 obtemos o alcance MÉDIO "
                    "de um arremesso com ângulo aleatório."),
    ]
