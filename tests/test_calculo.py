"""Testes automáticos: rode com  python -m pytest -v"""
import math

import pytest
import sympy as sp

from calculo import derivadas, funcoes, integrais, limites
from calculo.entrada import EntradaInvalida, ler_expr, ler_numero, x
from calculo.fisica import CESTA_H, Arremesso, angulo_otimo


# ---------------- entrada / tratamento de erros ----------------
@pytest.mark.parametrize("txt, esperado", [
    ("2x^2 + 1", 2 * x**2 + 1),
    ("sen(x)", sp.sin(x)),
    ("ln(x)", sp.log(x)),
    ("e^x", sp.exp(x)),
    ("2,5x", sp.Rational(5, 2) * x),
    ("raiz(x)", sp.sqrt(x)),
])
def test_ler_expr_valida(txt, esperado):
    assert sp.simplify(ler_expr(txt) - esperado) == 0


@pytest.mark.parametrize("txt", ["", "   ", "2x +* 3", "y + 1", "__import__('os')", "1/0", "x" * 200])
def test_ler_expr_invalida(txt):
    with pytest.raises(EntradaInvalida):
        ler_expr(txt)


def test_ler_numero():
    assert ler_numero("pi/2") == sp.pi / 2
    assert ler_numero("oo") == sp.oo
    with pytest.raises(EntradaInvalida):
        ler_numero("abc")


@pytest.mark.parametrize("v, ang", [(0, 45), (40, 45), (8, 0), (8, 90), (8, -10)])
def test_arremesso_invalido(v, ang):
    with pytest.raises(EntradaInvalida):
        Arremesso(v, ang)


# ---------------- física ----------------
def test_lance_livre_padrao_e_cesta():
    r = Arremesso().resultado()
    assert r["acertou"] and r["erro"] < 0.23


def test_altura_maxima_bate_com_derivada():
    a = Arremesso(9, 60)
    tp = sp.solve(sp.diff(a.h_t(), sp.Symbol("t", real=True)), sp.Symbol("t", real=True))[0]
    assert math.isclose(float(a.h_t().subs(sp.Symbol("t", real=True), tp)), a.h_max(), rel_tol=1e-9)


def test_bola_nao_alcanca_o_aro():
    r = Arremesso(4, 30).resultado()
    assert not r["acertou"] and "não alcançou" in r["msg"]


def test_angulo_otimo_formula_fechada():
    ang, vmin, *_ = angulo_otimo()
    esperado = 45 + 0.5 * math.degrees(math.atan((CESTA_H - 2.0) / 4.6))
    assert math.isclose(ang, esperado, abs_tol=1e-6)
    assert Arremesso(vmin, ang).resultado()["erro"] < 1e-6  # passa exatamente no centro


# ---------------- Aula 1: funções ----------------
def test_catalogo_tem_cinco_tipos_ou_mais():
    tipos = {f["tipo"] for f in funcoes.catalogo()}
    assert len(tipos) >= 5


@pytest.mark.parametrize("txt, tipo", [("5", "Constante"), ("2x+1", "Afim"), ("3x", "Linear"), ("x^2-4", "Quadrática"),
                                       ("abs(x)", "Modular"), ("2^x", "Exponencial"), ("ln(x)", "Logarítmica"),
                                       ("cos(x)", "Trigonométrica")])
def test_classificar(txt, tipo):
    assert funcoes.classificar(ler_expr(txt), x) == tipo


def test_dominio_e_imagem():
    assert funcoes.dominio(sp.log(x), x) == sp.Interval.open(0, sp.oo)
    assert funcoes.imagem(x**2 - 4, x, sp.S.Reals) == sp.Interval(-4, sp.oo)
    assert funcoes.imagem(sp.sin(x), x, sp.S.Reals) == sp.Interval(-1, 1)
    assert funcoes.imagem(sp.Abs(x), x, sp.S.Reals) == sp.Interval(0, sp.oo)


# ---------------- Aula 2: limites ----------------
def test_limite_existe_indeterminacao():
    r = limites.calcular((x**2 - 1) / (x - 1), x, 1)
    assert r["existe"] and r["valor"] == 2 and r["continua"] is False  # descontinuidade removível


def test_limite_laterais_diferentes():
    r = limites.calcular(sp.Abs(x) / x, x, 0)
    assert not r["existe"] and r["esq"] == -1 and r["dir"] == 1


def test_limite_infinito_e_no_infinito():
    assert limites.calcular(1 / x**2, x, 0)["valor"] == sp.oo
    assert limites.calcular(1 / x, x, sp.oo)["valor"] == 0
    assert not limites.calcular(sp.sin(1 / x), x, 0)["existe"]  # oscila


def test_exemplos_de_limites():
    ex = {e["conceito"]: limites.calcular(e["expr"], e["var"], e["ponto"], e.get("lado")) for e in limites.exemplos()}
    assert ex["Limites laterais e inexistência"]["existe"] is False
    assert ex["Continuidade"]["continua"] is True
    assert ex["Limite infinito / lateral"]["valor"] == sp.oo


# ---------------- Aula 3: derivadas ----------------
CASOS_DERIVADA = {
    "7": "Constante", "x^5": "Potência", "x^2 + 3x": "Soma/Subtração", "x^2*sen(x)": "Produto",
    "(x+1)/(x-1)": "Quociente", "sen(3x)": "Cadeia (Trigonométrica)", "e^(x^2)": "Cadeia (Exponencial)",
    "ln(x)": "Logarítmica", "2^x": "Exponencial", "(2x+1)^4": "Cadeia + Potência",
}


@pytest.mark.parametrize("txt, regra", CASOS_DERIVADA.items())
def test_derivadas_regras_e_resultado(txt, regra):
    f = ler_expr(txt)
    r = derivadas.resolver(f, x)
    assert regra in r["regras"]
    assert sp.simplify(r["resultado"] - sp.diff(f, x)) == 0 and r["validado"]


def test_cinco_regras_distintas_de_derivacao():
    usadas = set()
    for e in derivadas.exemplos():
        usadas |= set(derivadas.resolver(e["expr"], e["var"])["regras"])
    assert len(usadas - {"Geral (SymPy)"}) >= 5


# ---------------- Aula 4: integrais ----------------
CASOS_INTEGRAL = {
    "5": "Constante", "x^3": "Potência", "3x^2 + 2x": "Soma/Subtração", "e^x": "Exponencial",
    "cos(x)": "Trigonométrica", "x*e^(x^2)": "Substituição", "1/x": "Potência (n = −1)", "cos(3x+1)": "Substituição",
}


@pytest.mark.parametrize("txt, regra", CASOS_INTEGRAL.items())
def test_integrais_regras_e_resultado(txt, regra):
    f = ler_expr(txt)
    r = integrais.resolver(f, x)
    assert regra in r["regras"] and r["validado"]


@pytest.mark.parametrize("txt, a, b, valor", [
    ("3x^2", 0, 2, 8), ("x^2", 2, 0, sp.Rational(-8, 3)), ("1/x^2", 1, sp.oo, 1), ("e^(-x)", 0, sp.oo, 1),
    ("sen(x)", 0, sp.pi, 2), ("1/raiz(x)", 0, 1, 2),
])
def test_integrais_definidas(txt, a, b, valor):
    r = integrais.resolver(ler_expr(txt), x, a, b)
    assert sp.simplify(r["valor"] - valor) == 0 and r["validado"]


@pytest.mark.parametrize("txt, a, b", [("1/x", -1, 1), ("1/x", 0, 1), ("1/x", 1, sp.oo), ("tan(x)", 0, 3)])
def test_integrais_divergentes(txt, a, b):
    with pytest.raises(integrais.NaoIntegravel):
        integrais.resolver(ler_expr(txt), x, a, b)


def test_deslocamento_vertical_chega_no_aro():
    """∫ vy dt de 0 até t_aro + h0 deve dar a altura do aro (3,05 m)."""
    a = Arremesso()
    r = integrais.resolver(a.vy_t(), sp.Symbol("t", real=True), 0, a.resultado()["t"])
    assert math.isclose(a.h0 + float(r["valor"]), CESTA_H, abs_tol=1e-6)


def test_cinco_regras_distintas_de_integracao():
    usadas = set()
    for e in integrais.exemplos():
        usadas |= set(integrais.resolver(e["expr"], e["var"], e["a"], e["b"])["regras"])
    assert len(usadas - {"Geral (SymPy)"}) >= 5
