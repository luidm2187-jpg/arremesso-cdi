"""🏀 Arremesso Calculado — Projeto Final de CDI (AP2).

Execute com:  streamlit run app.py
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from calculo import derivadas, funcoes, integrais, limites
from calculo.entrada import EntradaInvalida, bonito, ler_expr, ler_numero, tex, th, x
from visual import quadra
from calculo.fisica import CESTA_D, CESTA_H, H0, RAIO_ARO, Arremesso, angulo_otimo, v2_necessaria

st.set_page_config(page_title="Arremesso Calculado", page_icon="🏀", layout="wide")

LARANJA, AZUL, CINZA = "#f97316", "#2563eb", "#9ca3af"
PADRAO = dict(v=7.5, ang=52.0, h0=H0, dist=CESTA_D)
for k, val in PADRAO.items():
    st.session_state.setdefault(k, val)

# arremesso feito no modo jogo → atualiza os sliders (e todos os cálculos) antes de desenhar a página
jogada = st.session_state.get("quadra")
if jogada and jogada.get("id") != st.session_state.get("_ultima_jogada"):
    st.session_state["_ultima_jogada"] = jogada["id"]
    st.session_state.update(v=float(jogada["v"]), ang=float(jogada["ang"]), dist=float(jogada["dist"]))


# ---------- utilidades de exibição ----------
def dec(e):
    return bonito(e)


def latex(e):
    return tex(e)


def num(v):
    return limites.fmt(v)


def mostrar_passos(passos):
    for i, p in enumerate(passos, 1):
        c1, c2 = st.columns([1, 3])
        formula = "\\" in p["como"]
        c1.markdown(f"**{i}. {p['regra']}**" + ("" if formula else f"  \n_{p['como']}_"))
        if formula:
            c1.latex(p["como"])
        c2.latex(p["latex"].replace(r"\log", r"\ln"))


def selo(ok):
    st.caption("✅ Validado: resultado conferido com o SymPy" if ok else "⚠️ Não foi possível validar automaticamente")


def curva(expr, var, lo, hi, n=400, corte=60):
    f = sp.lambdify(var, expr, ["numpy", {"Abs": np.abs}])
    xs = np.linspace(float(lo), float(hi), n)
    with np.errstate(all="ignore"):
        ys = np.asarray(np.broadcast_to(f(xs), xs.shape), dtype=complex)
    ys = np.where(np.abs(ys.imag) > 1e-9, np.nan, ys.real)
    ys[np.abs(ys) > corte] = np.nan
    return xs, ys


def fig_base(titulo="", xt="x", yt="y"):
    fig = go.Figure()
    fig.update_layout(title=titulo, xaxis_title=xt, yaxis_title=yt, height=360,
                      margin=dict(l=10, r=10, t=40 if titulo else 10, b=10), showlegend=True,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def erro(msg):
    st.error(f"❌ {msg}")


# ---------- barra lateral: parâmetros do arremesso ----------
with st.sidebar:
    st.title("🏀 Arremesso")
    st.slider("Velocidade inicial v₀ (m/s)", 3.0, 15.0, step=0.05, key="v")
    st.slider("Ângulo de lançamento θ (°)", 5.0, 85.0, step=0.5, key="ang")
    st.slider("Altura da mão h₀ (m)", 1.0, 2.8, step=0.05, key="h0")
    st.slider("Distância até o aro (m)", 1.0, 10.0, step=0.1, key="dist")
    st.button("↺ Lance livre padrão", on_click=lambda: st.session_state.update(PADRAO))
    st.caption(f"Aro a {CESTA_H} m de altura · raio {RAIO_ARO} m · g = 9,8 m/s²")

try:
    A = Arremesso(st.session_state.v, st.session_state.ang, st.session_state.h0, st.session_state.dist)
except EntradaInvalida as e:
    erro(e)
    st.stop()

st.title("🏀 Arremesso Calculado")
st.caption("Funções, limites, derivadas e integrais explicando por que a bola cai (ou não) na cesta.")

abas = st.tabs(["🏀 Arremessar", "📈 Funções", "🎯 Limites", "📐 Derivadas", "∫ Integrais", "🏆 Desafio", "ℹ️ Sobre"])

# ======================= ARREMESSAR =======================
with abas[0]:
    res = A.resultado()
    quadra(A)
    st.caption("🎮 Mire com o mouse, clique para a barra de força começar a correr e clique de novo para arremessar "
               "(ou use a barra de espaço). ← → ou A/D andam. Cada arremesso atualiza os sliders e os cálculos de todas as abas.")

    (st.success if res["acertou"] else st.warning)(("🎉 " if res["acertou"] else "😬 ") + res["msg"])
    c = st.columns(4)
    c[0].metric("Altura máxima", f"{A.h_max():.2f} m", help="h'(t) = 0 → t = v₀·senθ / g")
    c[1].metric("Tempo até o aro", f"{res['t']:.2f} s" if res["t"] else "—")
    c[2].metric("Erro |x − aro|", f"{res['erro']:.2f} m" if res["erro"] is not None else "—")
    c[3].metric("Componentes (vx, vy)", f"{A.vx:.2f} · {A.vy0:.2f}")
    with st.expander("Como o programa decide se foi cesta?"):
        st.latex(rf"h(t) = {A.h0} + {A.vy0:.2f}\,t - 4{{,}}9\,t^2 \qquad x(t) = {A.vx:.2f}\,t")
        st.markdown(
            "1. Resolve **h(t) = 3,05** (equação do 2º grau) e pega a raiz em que a bola está **descendo** (h'(t) < 0).\n"
            "2. Calcula onde a bola está nesse instante: **x(t)**.\n"
            f"3. Mede o erro com a função modular **|x − {A.dist}|**. Se o erro ≤ {RAIO_ARO} m (raio do aro), é cesta.\n\n"
            "_Modelo simplificado: sem resistência do ar, bola tratada como ponto, sem quicar no aro._")

# ======================= FUNÇÕES =======================
with abas[1]:
    st.subheader("Aula 1 — Funções do arremesso")
    cat = funcoes.catalogo(A)
    resumo = []
    for f in cat:
        an = funcoes.analisar(f)
        resumo.append({"Tipo": f["tipo"], "Função": f["nome"],
                       "Lei": str(dec(f["expr"])).replace("**", "^").replace("*", "·").replace("log", "ln"),
                       "Domínio": funcoes.texto_conjunto(an["dominio"]),
                       "Imagem": funcoes.texto_conjunto(an["imagem"]),
                       "Entrada → Saída": f"{f['ent']} → {f['sai']}"})
    st.dataframe(pd.DataFrame(resumo), hide_index=True, width="stretch")

    nome = st.selectbox("Escolha uma função para analisar", [f"{f['tipo']} — {f['nome']}" for f in cat])
    f = cat[[f"{g['tipo']} — {g['nome']}" for g in cat].index(nome)]
    an = funcoes.analisar(f)
    e1, e2 = st.columns([2, 3])
    with e1:
        st.latex(rf"f({sp.latex(f['var'])}) = {latex(f['expr'])}")
        st.markdown(f"**Tipo:** {f['tipo']}  \n**Entrada:** {f['ent']}  \n**Saída:** {f['sai']}")
        st.markdown("**Domínio e imagem**")
        st.latex(rf"D(f) = {sp.latex(funcoes.arredondar(an['dominio']))} \quad Im(f) = {sp.latex(funcoes.arredondar(an['imagem']))}")
        st.markdown("**No contexto do arremesso**")
        st.latex(rf"D = {sp.latex(funcoes.arredondar(an['dominio_ctx']))} \quad Im = {sp.latex(funcoes.arredondar(an['imagem_ctx']))}")
        st.info(f["interp"])
    with e2:
        xs_, ys_ = funcoes.pontos(f)
        g = fig_base(xt=f["ent"], yt=f["sai"])
        g.add_scatter(x=xs_, y=ys_, mode="lines", line=dict(color=LARANJA, width=3), name=f["tipo"])
        tab = funcoes.tabela(f)
        g.add_scatter(x=[a for a, _ in tab], y=[b for _, b in tab], mode="markers", marker=dict(color=AZUL, size=8),
                      name="pontos da tabela")
        st.plotly_chart(g, width="stretch")
        st.dataframe(pd.DataFrame(tab, columns=[f["ent"], f["sai"]]).T, width="stretch")

    st.divider()
    st.markdown("#### 🧪 Teste a sua função")
    txt = st.text_input("f(x) =", "x^2 - 4", help="Ex.: 2x+1, abs(x-3), 2^x, ln(x), sen(x), 5")
    try:
        ex = ler_expr(txt, x)
        dom = funcoes.dominio(ex, x)
        img = funcoes.imagem(ex, x, dom)
        k1, k2 = st.columns([2, 3])
        k1.latex(rf"f(x) = {latex(ex)}")
        k1.markdown(f"**Tipo identificado:** {funcoes.classificar(ex, x)}")
        k1.latex(rf"D(f) = {sp.latex(dom)}")
        k1.latex("Im(f) = " + (sp.latex(img) if img is not None else r"\text{não determinada}"))
        xs_, ys_ = curva(ex, x, -6, 6)
        g = fig_base(xt="x", yt="f(x)")
        g.add_scatter(x=xs_, y=ys_, mode="lines", line=dict(color=LARANJA, width=3), name="f(x)")
        k2.plotly_chart(g, width="stretch")
    except EntradaInvalida as e_:
        erro(e_)

# ======================= LIMITES =======================
with abas[2]:
    st.subheader("Aula 2 — Limites")
    for ex in limites.exemplos(A):
        with st.expander(f"**{ex['titulo']}** · {ex['conceito']}", expanded=True):
            r = limites.calcular(ex["expr"], ex["var"], ex["ponto"], ex.get("lado"))
            l1, l2 = st.columns([3, 2])
            with l1:
                st.latex(ex["latex"])
                if ex.get("lado") or ex["ponto"] in (sp.oo, -sp.oo):
                    st.markdown(f"**Resultado:** {num(r['esq'])}")
                else:
                    st.markdown(f"**Pela esquerda:** {num(r['esq'])} · **Pela direita:** {num(r['dir'])}")
                (st.success if r["existe"] else st.error)(r["motivo"])
                if r.get("continua") is not None:
                    st.markdown(f"**Contínua no ponto?** {'Sim ✅' if r['continua'] else 'Não ❌'} "
                                f"(f(ponto) = {num(r.get('f_ponto'))})")
                st.info(ex["interp"])
            with l2:
                p = sp.sympify(ex["ponto"])
                if p.is_finite:
                    ap = limites.aproximacoes(ex["expr"], ex["var"], p)
                    if ex.get("lado") != "-":
                        st.dataframe(pd.DataFrame(ap, columns=["distância h", "f(a − h)", "f(a + h)"]),
                                     hide_index=True, width="stretch")
                    lo, hi = (float(p) - 1.2, float(p) - 0.01) if ex.get("lado") == "-" else (float(p) - 1, float(p) + 1)
                else:
                    lo, hi = 0, 80
                xs_, ys_ = curva(ex["expr"], ex["var"], lo, hi, corte=200)
                g = fig_base(xt=str(ex["var"]).replace("Delta_t", "Δt"), yt="f")
                g.add_scatter(x=xs_, y=ys_, mode="lines", line=dict(color=LARANJA, width=3), name="f")
                if p.is_finite:
                    g.add_vline(x=float(p), line_dash="dash", line_color=CINZA)
                g.update_layout(height=280)
                st.plotly_chart(g, width="stretch")

    st.divider()
    st.markdown("#### 🧪 Calcule um limite")
    q1, q2, q3 = st.columns([3, 1, 1])
    txt = q1.text_input("f(x) =", "(x^2 - 1)/(x - 1)", key="lim_f", help="Teste também abs(x)/x em 0, 1/x^2 em 0, sen(x)/x em 0")
    pto = q2.text_input("x →", "1", help="Aceita pi, oo (infinito), -oo")
    lado = q3.selectbox("Lado", ["ambos", "esquerda (−)", "direita (+)"])
    try:
        ex, p = ler_expr(txt, x), ler_numero(pto)
        r = limites.calcular(ex, x, p, {"ambos": None, "esquerda (−)": "-", "direita (+)": "+"}[lado])
        seta = {"ambos": "", "esquerda (−)": "^-", "direita (+)": "^+"}[lado]
        st.latex(rf"\lim_{{x \to {sp.latex(p)}{seta}}} {latex(ex)}")
        st.markdown(f"Esquerda: **{num(r['esq'])}** · Direita: **{num(r['dir'])}**")
        (st.success if r["existe"] else st.error)(r["motivo"])
        if r.get("continua") is not None:
            st.markdown(f"Contínua em x = {sp.latex(p)}? **{'Sim' if r['continua'] else 'Não'}** "
                        f"(f({sp.latex(p)}) = {num(r.get('f_ponto'))})")
    except EntradaInvalida as e_:
        erro(e_)
    except Exception:
        erro("Não foi possível calcular esse limite. Verifique a expressão.")

# ======================= DERIVADAS =======================
with abas[3]:
    st.subheader("Aula 3 — Derivadas (regras aplicadas passo a passo)")
    for ex in derivadas.exemplos(A):
        r = derivadas.resolver(ex["expr"], ex["var"])
        v = sp.latex(ex["var"])
        with st.expander(f"**Regra: {ex['regra']}** → " + ", ".join(r["regras"])):
            st.latex(rf"\frac{{d}}{{d{v}}}\left[{latex(ex['expr'])}\right] = {latex(r['resultado'])}")
            mostrar_passos(r["passos"])
            st.info(ex["interp"])
            selo(r["validado"])

    st.divider()
    st.markdown("#### 🧪 Derive a sua função")
    d1, d2 = st.columns([3, 1])
    txt = d1.text_input("f(x) =", "x^2 * sen(x)", key="der_f", help="Ex.: (x^2+1)/(x-1), e^(3x), ln(x^2+1), cos(x)^3")
    x0 = d2.text_input("Reta tangente em x =", "1")
    try:
        ex = ler_expr(txt, x)
        r = derivadas.resolver(ex, x)
        st.latex(rf"f'(x) = {latex(r['resultado'])}")
        st.markdown("**Regras usadas:** " + ", ".join(r["regras"]))
        mostrar_passos(r["passos"])
        selo(r["validado"])
        p = ler_numero(x0)
        fp = ex.subs(x, p)
        if not fp.is_finite or not fp.is_real:
            raise EntradaInvalida(f"x = {x0} não está no domínio de f.")
        reta, m = derivadas.tangente(ex, x, p)
        st.markdown(f"Em x = {x0}: f = **{num(fp)}**, inclinação f'(x) = **{num(m)}** → reta tangente:")
        st.latex(rf"y = {latex(reta)}")
        xs_, ys_ = curva(ex, x, float(p) - 4, float(p) + 4)
        g = fig_base(xt="x", yt="y")
        g.add_scatter(x=xs_, y=ys_, mode="lines", line=dict(color=LARANJA, width=3), name="f(x)")
        xs2, ys2 = curva(r["resultado"], x, float(p) - 4, float(p) + 4)
        g.add_scatter(x=xs2, y=ys2, mode="lines", line=dict(color=AZUL, dash="dash"), name="f'(x)")
        xt_, yt_ = curva(reta, x, float(p) - 2, float(p) + 2)
        g.add_scatter(x=xt_, y=yt_, mode="lines", line=dict(color="#16a34a"), name="tangente")
        st.plotly_chart(g, width="stretch")
    except EntradaInvalida as e_:
        erro(e_)
    except (TypeError, ValueError):
        erro("Não foi possível avaliar a função nesse ponto.")

# ======================= INTEGRAIS =======================
with abas[4]:
    st.subheader("Aula 4 — Integrais (regras aplicadas passo a passo)")
    for ex in integrais.exemplos(A):
        r = integrais.resolver(ex["expr"], ex["var"], ex["a"], ex["b"])
        v = sp.latex(ex["var"])
        with st.expander(f"**Regra: {ex['regra']}** → " + ", ".join(r["regras"])):
            if r["valor"] is None:
                st.latex(rf"\int {latex(ex['expr'])}\,d{v} = {latex(r['primitiva'])} + C")
            else:
                st.latex(rf"\int_{{{latex(ex['a'])}}}^{{{latex(ex['b'])}}} {latex(ex['expr'])}\,d{v} = "
                         rf"\Big[{latex(r['primitiva'])}\Big]_{{{latex(ex['a'])}}}^{{{latex(ex['b'])}}} "
                         rf"= {latex(r['valor'])} \approx {num(r['valor'])}")
            mostrar_passos(r["passos"])
            st.info(ex["interp"])
            selo(r["validado"])

    st.divider()
    st.markdown("#### 🧪 Integre a sua função")
    i1, i2, i3 = st.columns([3, 1, 1])
    txt = i1.text_input("f(x) =", "3x^2 + 2x", key="int_f", help="Ex.: x*e^(x^2), cos(3x+1), 1/x, 2^x, sen(x)^2*cos(x)")
    ta = i2.text_input("de a =", "0", help="Deixe vazio para integral indefinida")
    tb = i3.text_input("até b =", "2", help="Aceita pi, oo")
    try:
        ex = ler_expr(txt, x)
        definida = bool(ta.strip()) and bool(tb.strip())
        if bool(ta.strip()) != bool(tb.strip()):
            raise EntradaInvalida("Preencha os dois limites (a e b) ou deixe os dois vazios.")
        a, b = (ler_numero(ta), ler_numero(tb)) if definida else (None, None)
        r = integrais.resolver(ex, x, a, b)
        if definida:
            st.latex(rf"\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} {latex(ex)}\,dx = F({sp.latex(b)}) - F({sp.latex(a)}) "
                     rf"= {latex(r['valor'])} \approx {num(r['valor'])}")
        st.latex(rf"F(x) = {latex(r['primitiva'])} + C")
        st.markdown("**Regras usadas:** " + ", ".join(r["regras"]))
        mostrar_passos(r["passos"])
        selo(r["validado"])
        if definida and a.is_finite and b.is_finite:
            lo, hi = sorted([float(a), float(b)])
            m = (hi - lo) * 0.3 + 0.5
            xs_, ys_ = curva(ex, x, lo - m, hi + m)
            xa, ya = curva(ex, x, lo, hi)
            g = fig_base(xt="x", yt="f(x)")
            g.add_scatter(x=xa, y=ya, fill="tozeroy", mode="none", fillcolor="rgba(249,115,22,.3)", name="área com sinal")
            g.add_scatter(x=xs_, y=ys_, mode="lines", line=dict(color=LARANJA, width=3), name="f(x)")
            st.plotly_chart(g, width="stretch")
    except (EntradaInvalida, integrais.NaoIntegravel) as e_:
        erro(e_)
    except Exception:
        erro("Não foi possível calcular essa integral. Verifique a expressão e os limites.")

# ======================= DESAFIO =======================
with abas[5]:
    st.subheader("🏆 Desafio: qual o arremesso que exige MENOS força?")
    st.markdown(
        "Para a bola passar no centro do aro, a velocidade necessária depende do ângulo. "
        "Igualando h(x) = 3,05 com x = distância do aro, isolamos v²:")
    st.latex(r"v^2(\theta) = \frac{g\,d^2}{2\cos^2\theta\,\big(d\,\tan\theta - \Delta h\big)}"
             rf"\qquad d = {A.dist},\ \Delta h = {CESTA_H} - {A.h0} = {CESTA_H - A.h0:.2f}")
    ang_opt, v_min, v2, dv2 = angulo_otimo(A.h0, A.dist)
    st.markdown("O mínimo acontece onde a **derivada é zero**:")
    st.latex(r"\frac{d(v^2)}{d\theta} = 0 \;\Rightarrow\; \theta^* = " + f"{ang_opt:.2f}^\\circ" +
             r"\quad\Rightarrow\quad v_{\min} = " + f"{v_min:.2f}" + r"\ \text{m/s}")
    st.caption(f"Conferência pela fórmula fechada θ* = 45° + ½·arctg(Δh/d) = "
               f"{45 + 0.5 * math.degrees(math.atan((CESTA_H - A.h0) / A.dist)):.2f}°")

    th_min = math.atan((CESTA_H - A.h0) / A.dist)  # assíntota: abaixo disso a bola nunca sobe o suficiente
    angs = np.linspace(math.degrees(th_min) + 0.5, 89, 300)
    vf = sp.lambdify(th, sp.sqrt(v2), "numpy")
    vs = vf(np.radians(angs))
    g = fig_base(xt="ângulo θ (°)", yt="velocidade necessária (m/s)")
    g.add_scatter(x=angs, y=vs, mode="lines", line=dict(color=LARANJA, width=3), name="v(θ)")
    g.add_scatter(x=[ang_opt], y=[v_min], mode="markers+text", text=[f"mínimo: {ang_opt:.1f}°"],
                  textposition="top center", marker=dict(size=12, color=AZUL), name="v'(θ) = 0")
    g.add_scatter(x=[A.ang], y=[A.v], mode="markers", marker=dict(size=12, color="#16a34a", symbol="star"),
                  name="seu arremesso")
    g.update_yaxes(range=[0, max(20, A.v + 2)])
    st.plotly_chart(g, width="stretch")
    st.caption("Repare nos limites: quando θ se aproxima do ângulo mínimo ou de 90°, v(θ) → +∞ (assíntotas).")

    # quais ângulos acertam com a velocidade atual? resolve v²(θ) = v₀²
    dif = vs**2 - A.v**2
    raizes = [float(sp.nsolve(v2 - A.v**2, th, math.radians(angs[i])))
              for i in range(len(dif) - 1) if np.sign(dif[i]) != np.sign(dif[i + 1])]
    if raizes:
        txt_r = " e ".join(f"**{math.degrees(r):.1f}°**" for r in raizes)
        st.success(f"Com v₀ = {A.v:.2f} m/s, os ângulos que acertam o centro do aro são {txt_r}.")
    else:
        st.warning(f"Com v₀ = {A.v:.2f} m/s nenhum ângulo leva a bola exatamente ao centro do aro: o mínimo é {v_min:.2f} m/s.")

    def usar_otimo():
        st.session_state.update(ang=round(ang_opt * 2) / 2, v=round(math.ceil(v_min / 0.05) * 0.05, 2))

    st.button("🎯 Usar o arremesso ótimo", on_click=usar_otimo, type="primary")

# ======================= SOBRE =======================
with abas[6]:
    st.markdown(f"""
### Sobre o projeto
**Problema:** por que alguns arremessos caem na cesta e outros não? Qual ângulo exige menos força?

**Solução:** o app modela o arremesso como um lançamento oblíquo e usa os 4 conteúdos de CDI:

| Aula | Onde aparece | Pergunta respondida |
|---|---|---|
| 1 — Funções | x(t), h(t), vy(t), erro modular, alcance R(θ)… | Onde a bola está em cada instante? |
| 2 — Limites | velocidade instantânea, quique, θ → 90°, t → ∞ | O que acontece nos extremos e nas transições? |
| 3 — Derivadas | h'(t) = velocidade, R'(θ) = 0, d(v²)/dθ = 0 | Qual a altura máxima e o ângulo ótimo? |
| 4 — Integrais | ∫ vy dt = deslocamento, ∫ v·cosθ dθ, alcance médio | Quanto a bola percorreu? |

**Modelo:** h(t) = h₀ + v₀·senθ·t − 4,9t², x(t) = v₀·cosθ·t, sem resistência do ar (exceto nos exemplos
exponenciais, que mostram o efeito do ar). Aro a {CESTA_H} m, lance livre a {CESTA_D} m.

**Validação:** as regras de derivação e integração são aplicadas por um motor próprio (`calculo/derivadas.py`,
`calculo/integrais.py`) e cada resultado é conferido com a biblioteca SymPy. Os testes automáticos estão em `tests/`.
""")
