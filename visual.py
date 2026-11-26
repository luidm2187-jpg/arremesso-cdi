"""Quadra interativa (modo jogo) — componente Streamlit feito em HTML/JavaScript.

O jogador mira com o mouse, escolhe a força numa barra que vai e volta e anda com ← →.
A física do arremesso no navegador (componentes/quadra/fisica.js) é a mesma de calculo/fisica.py;
o teste tests/test_fisica_js.py confere as duas versões. Depois de cada arremesso o componente
devolve {v, ang, dist} para o Python, que atualiza os sliders e todos os cálculos das abas.
"""
from pathlib import Path

import streamlit.components.v1 as components

from calculo.fisica import Arremesso

_quadra = components.declare_component("quadra", path=str(Path(__file__).parent / "componentes" / "quadra"))


def quadra(a: Arremesso, key: str = "quadra"):
    """Desenha a quadra e devolve o último arremesso feito no jogo (ou None)."""
    return _quadra(v=a.v, ang=a.ang, h0=a.h0, dist=a.dist, key=key, default=None)
