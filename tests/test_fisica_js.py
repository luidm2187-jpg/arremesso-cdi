"""Confere se a física do jogo (JavaScript) é igual à do Python. Precisa do Node.js; sem ele, é pulado."""
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from calculo.entrada import th
from calculo.fisica import Arremesso, v2_necessaria

JS = Path(__file__).parent.parent / "componentes" / "quadra" / "fisica.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js não instalado")
def test_fisica_js_igual_python():
    random.seed(1)
    casos = []
    for _ in range(300):
        v, a = round(random.uniform(3, 15) / 0.05) * 0.05, round(random.uniform(5, 85) * 2) / 2
        h, d = round(random.uniform(1, 2.8), 2), round(random.uniform(1, 10), 1)
        r = Arremesso(v, a, h, d).resultado()
        vi = float(v2_necessaria(h, d).subs(th, math.radians(a)))
        casos.append([v, a, h, d, r["acertou"], r["erro"], math.sqrt(vi) if vi > 0 else None])
    casos.append([7.5, 52, 2, 4.6, True, Arremesso().resultado()["erro"], None])
    casos[-1][-1] = math.sqrt(float(v2_necessaria().subs(th, math.radians(52))))
    js = (f"const F=require({json.dumps(str(JS))});const C={json.dumps(casos)};let ruins=0;"
          "for(const [v,a,h,d,ok,e,vi] of C){const r=F.resultado(v,a,h,d),w=F.vIdeal(a,h,d);"
          "if(r.acertou!==ok||(e===null)!==(r.erro===null)||(e!==null&&Math.abs(e-r.erro)>1e-9)"
          "||(vi===null)!==(w===null)||(vi!==null&&Math.abs(vi-w)>1e-9))ruins++;}console.log(ruins)")
    saida = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout.strip()
    assert saida == "0"
