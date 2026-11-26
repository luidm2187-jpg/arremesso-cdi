// Física do arremesso — mesma modelagem de calculo/fisica.py (conferida nos testes).
// x(t) = v·cosθ·t      h(t) = h0 + v·senθ·t − (g/2)·t²
const FIS = { G: 9.8, ARO: 3.05, RAIO: 0.23, R_BOLA: 0.12 };

function geometria(dist) {
  const tabela = dist + 0.375, fundo = tabela + 1.2;
  return { tabela, fundo, poste: fundo + 0.45 };
}

// Mesmo critério de Arremesso.resultado(): a bola desce pela altura do aro com |x − d| ≤ raio.
function resultado(v, angGraus, h0, dist) {
  const { G, ARO, RAIO } = FIS, th = angGraus * Math.PI / 180;
  const vx = v * Math.cos(th), vy = v * Math.sin(th), hmax = h0 + vy * vy / (2 * G);
  const r = { acertou: false, erro: null, hmax, vx, vy, tpico: vy / G };
  if (hmax < ARO) { r.msg = `A bola não alcançou a altura do aro (faltaram ${(ARO - hmax).toFixed(2)} m).`; return r; }
  const tc = (vy + Math.sqrt(vy * vy - 2 * G * (ARO - h0))) / G, xc = vx * tc, erro = Math.abs(xc - dist);
  Object.assign(r, { t: tc, x: xc, erro, acertou: erro <= RAIO });
  r.msg = r.acertou ? `CESTA! A bola passou a ${erro.toFixed(2)} m do centro do aro.`
    : `Errou: bola ${xc > dist ? "longa (passou do aro)" : "curta (antes do aro)"}, ${erro.toFixed(2)} m do centro.`;
  return r;
}

// v²(θ) necessária para passar no centro do aro (mesma fórmula da aba Desafio).
function vIdeal(angGraus, h0, dist) {
  const th = angGraus * Math.PI / 180, dh = FIS.ARO - h0, den = 2 * Math.cos(th) ** 2 * (dist * Math.tan(th) - dh);
  return den > 0 ? Math.sqrt(FIS.G * dist * dist / den) : null;
}

// Trajetória para a animação, com colisões (aro, tabela, poste, chão) e lista de eventos para os sons.
// O resultado (cesta ou não) é SEMPRE o de resultado(): se ele diz "errou", a bola não pode cair dentro do aro;
// se diz "cesta", a bola passa limpa. Assim a animação nunca contradiz o cálculo.
function simular(v, angGraus, h0, dist, dtSaida = 0.01, tMax = 6) {
  const { G, ARO, RAIO, R_BOLA } = FIS, g = geometria(dist), th = angGraus * Math.PI / 180, dt = 0.001;
  const acerto = resultado(v, angGraus, h0, dist).acertou, RC = R_BOLA + 0.012;   // raio de contato bola-aro
  const aros = [[dist - RAIO, ARO], [dist + RAIO, ARO]];
  let x = 0, y = h0, vx = v * Math.cos(th), vy = v * Math.sin(th), t = 0, quiques = 0, bateu = false, entrou = null;
  const pts = [], eventos = [], passo = Math.round(dtSaida / dt);
  let n = 0, ultAro = -1, presa = 0, cunha = 0, noAro = 0;
  const evento = (tipo, forca) => eventos.push({ i: Math.floor(n / passo), tipo, forca: Math.min(1, forca) });
  while (t < tMax && quiques < 6) {
    const xa = x, ya = y;
    x += vx * dt; y += vy * dt - G / 2 * dt * dt; vy -= G * dt;
    if (entrou === null && acerto && !bateu && ya >= ARO && ARO > y && Math.abs(x - dist) <= RAIO) {
      entrou = Math.floor(n / passo); vx *= 0.05; evento("rede", 1);
    }
    if (!acerto) {
      for (const [ax, ay] of aros) {                     // choque com a borda do aro
        const dx = x - ax, dy = y - ay, d = Math.hypot(dx, dy);
        if (d < RC) {
          const nx = dx / d, ny = dy / d, vn = vx * nx + vy * ny;
          if (vn < 0) {
            const tx = -ny, ty = nx, vt = (vx * tx + vy * ty) * 0.85;
            vx = -0.55 * vn * nx + vt * tx; vy = -0.55 * vn * ny + vt * ty;
            if (n - ultAro > 40) evento("aro", Math.abs(vn) / 6);
            ultAro = n; bateu = true;
          }
          x = ax + nx * RC; y = ay + ny * RC;
        }
      }
      // num erro a bola nunca atravessa o aro por dentro: quica para fora
      if (ya >= ARO && ARO > y && Math.abs(x - dist) < RAIO - RC * 0.5) {
        const lado = x >= dist ? 1 : -1;
        y = ARO + 0.002; presa++;
        if (presa > 2) { vx = lado * 1.2; vy = 2.4; }               // girou no aro: escapa por cima da borda
        else { vy = Math.abs(vy) * 0.45; vx = lado * Math.max(Math.abs(vx), 1.1); }
        if (n - ultAro > 40) evento("aro", 0.7);
        ultAro = n; bateu = true;
      }
    }
    // parou equilibrada em cima do aro? depois de 0,5 s ela cai para fora (para frente)
    if (!acerto && bateu && y > ARO - 0.25 && Math.abs(x - dist) < 0.45) { if (++noAro > 500) { vx = -2.3; vy = 2.8; noAro = 0; } }
    else noAro = 0;
    if (vx > 0 && x >= g.tabela - R_BOLA && y >= 2.9 && y <= 3.95 && xa < g.tabela - R_BOLA) {
      x = g.tabela - R_BOLA; evento("tabela", Math.abs(vx) / 8); vx *= -0.6; bateu = true;
      if (Math.abs(y - ARO) < 0.3 && ++cunha > 1) { vx = -2.3; vy = 2.8; }   // presa entre aro e tabela: escapa
    }
    if (vx > 0 && x >= g.poste - 0.2 - R_BOLA) { x = g.poste - 0.2 - R_BOLA; evento("poste", Math.abs(vx) / 6); vx *= -0.5; bateu = true; }
    if (y <= R_BOLA && vy < 0) {
      y = R_BOLA; if (Math.abs(vy) > 0.6) evento("chao", Math.abs(vy) / 8);
      vy *= -0.55; vx *= 0.8; quiques++; bateu = true;
    }
    if (n % passo === 0) pts.push([x, y]);
    t += dt; n++;
  }
  return { pts, entrou, eventos, dt: dtSaida };
}

if (typeof module !== "undefined") module.exports = { FIS, geometria, resultado, vIdeal, simular };
