"""Leitura segura de expressões digitadas pelo usuário (tratamento de entradas inválidas)."""
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor,
)

x, t, th = sp.symbols("x t theta", real=True)

TRANSF = (*standard_transformations, implicit_multiplication_application, convert_xor)
NOMES = {
    "x": x, "t": t, "theta": th, "e": sp.E, "pi": sp.pi, "oo": sp.oo,
    "sen": sp.sin, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "tg": sp.tan,
    "ln": sp.log, "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "raiz": sp.sqrt, "abs": sp.Abs,
}
PROIBIDO = ("__", "import", "lambda", "exec", "eval", "open", ";", "[", "{", "'", '"')


class EntradaInvalida(ValueError):
    """Erro com mensagem amigável para mostrar na interface."""


def ler_expr(texto: str, var=x):
    """Converte texto em expressão SymPy aceitando só a variável `var`.

    Aceita: 2x^2, sen(x), ln(x), e^x, raiz(x), abs(x), vírgula decimal (2,5).
    """
    if texto is None or not str(texto).strip():
        raise EntradaInvalida("Digite uma expressão.")
    texto = str(texto).strip()
    if len(texto) > 120:
        raise EntradaInvalida("Expressão muito longa (máx. 120 caracteres).")
    if any(p in texto.lower() for p in PROIBIDO):
        raise EntradaInvalida("A expressão contém caracteres ou palavras não permitidos.")
    texto = texto.replace(",", ".")
    try:
        expr = parse_expr(texto, local_dict=NOMES, transformations=TRANSF)
    except Exception:
        raise EntradaInvalida(f"Não entendi '{texto}'. Exemplo válido: 3x^2 + sen(x)")
    if not isinstance(expr, sp.Expr):
        raise EntradaInvalida("Isso não é uma expressão matemática.")
    extras = expr.free_symbols - {var}
    if extras:
        nomes = ", ".join(sorted(str(s) for s in extras))
        raise EntradaInvalida(f"Use apenas a variável {var}. Encontrei: {nomes}")
    if expr.has(sp.zoo, sp.nan):
        raise EntradaInvalida("A expressão é indefinida (ex.: divisão por zero).")
    return expr


def ler_numero(texto: str) -> sp.Expr:
    """Lê um número/limite: aceita 2, -1,5, pi/2, oo, -oo."""
    texto = str(texto).strip().replace("∞", "oo").replace("inf", "oo")
    if not texto:
        raise EntradaInvalida("Digite um número.")
    try:
        val = parse_expr(texto.replace(",", "."), local_dict=NOMES, transformations=TRANSF)
    except Exception:
        raise EntradaInvalida(f"'{texto}' não é um número válido.")
    if not isinstance(val, sp.Expr) or val.free_symbols or val.has(sp.zoo, sp.nan):
        raise EntradaInvalida(f"'{texto}' não é um número válido.")
    return val


def bonito(e):
    """Troca frações decimais exatas (49/10, 591/100) por 4.9 e 5.91 para exibição.

    Frações como 1/3 continuam como fração.
    """
    if not isinstance(e, sp.Basic):
        return e
    troca, expoentes = {}, {p.exp for p in e.atoms(sp.Pow)}
    for r in e.atoms(sp.Rational) - expoentes:
        q = int(r.q)
        while q % 2 == 0:
            q //= 2
        while q % 5 == 0:
            q //= 5
        if r.q != 1 and q == 1:
            troca[r] = sp.Float(r, 15)
    return e.xreplace(troca)


def tex(e) -> str:
    """LaTeX amigável: decimais exatos e ln no lugar de log."""
    return sp.latex(bonito(e)).replace(r"\log", r"\ln")
