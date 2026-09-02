"""
canvas_processo.py

Cena visual do processo comercial: estacoes em sequencia horizontal, e
ANTES de cada estacao uma pilha visivel de bolinhas pequenas representando
leads que ja chegaram mas ainda nao foram processados por aquela estacao
-- isto e' a FILA DE VERDADE, sempre visivel, nao um numero escondido.

Cada estacao tem tres zonas visuais, da esquerda para a direita:
  1. pilha de espera (bolinhas pequenas empilhadas, cor neutra)
  2. a estacao em si (retangulo nomeado, mostra o que esta acontecendo)
  3. contador "concluidos" (numero fixo de quantos ja passaram por aqui)

Estacoes de IA (qualificacao_ia, match_closer_ia, proposta_ia) tem um
indicador extra abaixo do nome mostrando explicitamente qual modelo
esta sendo usado (RandomForest / kNN / LLM) e o resultado do calculo
mais recente -- isto e' o que torna a IA "visivel" e nao apenas um
numero a mais na tabela.
"""

import json

LAYOUTS = {
    "A": [
        {"id": "geracao_leads",     "nome": "Entrada",     "tipo": "fila",      "modelo": None},
        {"id": "qualificacao_sdr",  "nome": "SDR",          "tipo": "humano",    "modelo": None},
        {"id": "descoberta_closer", "nome": "Descoberta",   "tipo": "humano",    "modelo": None},
        {"id": "proposta_manual",   "nome": "Proposta",     "tipo": "humano",    "modelo": None},
        {"id": "negociacao",        "nome": "Negociação",   "tipo": "humano",    "modelo": None},
        {"id": "fechamento",        "nome": "Fechamento",   "tipo": "resultado", "modelo": None},
    ],
    "B": [
        {"id": "geracao_leads",     "nome": "Entrada",       "tipo": "fila",      "modelo": None},
        {"id": "qualificacao_sdr",  "nome": "SDR",            "tipo": "humano",    "modelo": None},
        {"id": "descoberta_closer", "nome": "Descoberta",     "tipo": "humano",    "modelo": None},
        {"id": "proposta_ia",       "nome": "Proposta",       "tipo": "ia_llm",    "modelo": "LLM (Llama 3.1)"},
        {"id": "negociacao",        "nome": "Negociação",     "tipo": "humano",    "modelo": None},
        {"id": "fechamento",        "nome": "Fechamento",     "tipo": "resultado", "modelo": None},
    ],
    "C": [
        {"id": "geracao_leads",     "nome": "Entrada",       "tipo": "fila",      "modelo": None},
        {"id": "qualificacao_ia",   "nome": "Triagem",        "tipo": "ia_score",  "modelo": "RandomForest"},
        {"id": "match_closer_ia",   "nome": "Match closer",   "tipo": "ia_knn",    "modelo": "kNN"},
        {"id": "descoberta_closer", "nome": "Descoberta",     "tipo": "humano",    "modelo": None},
        {"id": "proposta_ia",       "nome": "Proposta",       "tipo": "ia_llm",    "modelo": "LLM (Llama 3.1)"},
        {"id": "negociacao",        "nome": "Negociação",     "tipo": "humano",    "modelo": None},
        {"id": "fechamento",        "nome": "Fechamento",     "tipo": "resultado", "modelo": None},
    ],
    # Estado D — ablacao: so' a qualificacao recebe IA (mesmo RandomForest do
    # Estado C), sem kNN de match e sem LLM na proposta -- mesma cardinalidade
    # de intervencao do Estado B (1 etapa), mas em local diferente.
    "D": [
        {"id": "geracao_leads",     "nome": "Entrada",       "tipo": "fila",      "modelo": None},
        {"id": "qualificacao_ia",   "nome": "Triagem",        "tipo": "ia_score",  "modelo": "RandomForest"},
        {"id": "descoberta_closer", "nome": "Descoberta",     "tipo": "humano",    "modelo": None},
        {"id": "proposta_manual",   "nome": "Proposta",       "tipo": "humano",    "modelo": None},
        {"id": "negociacao",        "nome": "Negociação",     "tipo": "humano",    "modelo": None},
        {"id": "fechamento",        "nome": "Fechamento",     "tipo": "resultado", "modelo": None},
    ],
}

LARGURA_COLUNA = 130
LARGURA_ESTACAO = 100
ALTURA_ESTACAO = 70
Y_TOPO_ESTACAO = 80
MARGEM_X = 24
MAX_BOLINHAS_FILA = 10


def _calcular_x(n_estacoes: int) -> list:
    return [MARGEM_X + i * LARGURA_COLUNA for i in range(n_estacoes)]


def _contar_por_etapa(eventos_completos: list, etapas_ids: list) -> dict:
    vistos_por_etapa = {e: set() for e in etapas_ids}
    aprovados_por_etapa = {e: 0 for e in etapas_ids}
    for ev in eventos_completos:
        etapa = ev.get("etapa")
        if etapa not in vistos_por_etapa:
            continue
        lid = ev.get("lead_id")
        if lid not in vistos_por_etapa[etapa]:
            vistos_por_etapa[etapa].add(lid)
            if ev.get("passou"):
                aprovados_por_etapa[etapa] += 1
    return {
        e: {"total": len(vistos_por_etapa[e]), "aprovados": aprovados_por_etapa[e]}
        for e in etapas_ids
    }


def _ultimo_evento_por_etapa(eventos_completos: list, etapas_ids: list) -> dict:
    ultimo = {}
    for ev in eventos_completos:
        if ev.get("etapa") in etapas_ids:
            ultimo[ev["etapa"]] = ev
    return ultimo


def gerar_html_canvas(
    estado: str,
    eventos_completos: list,
    eventos_novos: list,
    altura_px: int = 280,
) -> str:
    layout = LAYOUTS[estado]
    etapas_ids = [s["id"] for s in layout]
    xs = _calcular_x(len(layout))
    for s, x in zip(layout, xs):
        s["x"] = x

    largura_total = xs[-1] + LARGURA_ESTACAO + MARGEM_X

    contadores = _contar_por_etapa(eventos_completos, etapas_ids)
    ultimo_por_etapa = _ultimo_evento_por_etapa(eventos_completos, etapas_ids)
    total_leads_no_sistema = len({ev.get("lead_id") for ev in eventos_completos}) if eventos_completos else 0

    filas = {}
    for i, s in enumerate(layout):
        if i == 0:
            filas[s["id"]] = 0
            continue
        anterior = layout[i - 1]["id"]
        cont_anterior = contadores.get(anterior, {"aprovados": 0})
        cont_atual = contadores.get(s["id"], {"total": 0})
        filas[s["id"]] = max(0, cont_anterior["aprovados"] - cont_atual["total"])

    layout_json = json.dumps(layout, ensure_ascii=False).replace("</", "<\\/")
    eventos_json = json.dumps(eventos_novos, ensure_ascii=False).replace("</", "<\\/")
    contadores_json = json.dumps(contadores, ensure_ascii=False).replace("</", "<\\/")
    filas_json = json.dumps(filas, ensure_ascii=False).replace("</", "<\\/")
    ultimo_json = json.dumps(
        {k: v.get("detalhes", {}) for k, v in ultimo_por_etapa.items()},
        ensure_ascii=False
    ).replace("</", "<\\/")

    return f"""
<div id="canvas-wrap" style="position:relative;width:100%;background:var(--color-background-secondary);
     border-radius:12px;padding:14px 12px;box-sizing:border-box;font-family:sans-serif;overflow-x:auto">
  <svg id="canvas-svg" width="{largura_total}" height="{altura_px}"
       viewBox="0 0 {largura_total} {altura_px}" style="display:block">
    <defs>
      <marker id="arrowc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M2 1L8 5L2 9" fill="none" stroke="#9c9a92" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </marker>
    </defs>
    <g id="trilhas-layer"></g>
    <g id="filas-layer"></g>
    <g id="estacoes-layer"></g>
    <g id="transito-layer"></g>
  </svg>
  <div id="status-line" style="margin-top:8px;font-size:11px;color:var(--color-text-secondary);min-height:16px"></div>
  <div id="legenda-line" style="margin-top:4px;font-size:10px;color:var(--color-text-secondary)">
    <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#9c9a92;margin-right:3px"></span>aguardando
    &nbsp;&nbsp;
    <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#378ADD;margin-right:3px"></span>em trânsito
    &nbsp;&nbsp;
    <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#1D9E75;margin-right:3px"></span>avançou
    &nbsp;&nbsp;
    <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#E24B4A;margin-right:3px"></span>não avançou
  </div>
</div>

<script>
(function() {{
  const LAYOUT = {layout_json};
  const EVENTOS_NOVOS = {eventos_json};
  const CONTADORES = {contadores_json};
  const FILAS = {filas_json};
  const ULTIMO_DETALHE = {ultimo_json};
  const TOTAL_NO_SISTEMA = {total_leads_no_sistema};
  const YT = {Y_TOPO_ESTACAO};
  const WE = {LARGURA_ESTACAO}, HE = {ALTURA_ESTACAO};
  const MAXB = {MAX_BOLINHAS_FILA};

  const estacoesLayer = document.getElementById('estacoes-layer');
  const trilhasLayer = document.getElementById('trilhas-layer');
  const filasLayer = document.getElementById('filas-layer');
  const transitoLayer = document.getElementById('transito-layer');
  const statusLine = document.getElementById('status-line');

  const NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs) {{
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }}

  const CORES = {{
    humano:    {{ fill: '#E6F1FB', stroke: '#378ADD', text: '#0C447C' }},
    ia_score:  {{ fill: '#EEEDFE', stroke: '#7F77DD', text: '#3C3489' }},
    ia_knn:    {{ fill: '#EEEDFE', stroke: '#7F77DD', text: '#3C3489' }},
    ia_llm:    {{ fill: '#FAECE7', stroke: '#D85A30', text: '#993C1D' }},
    fila:      {{ fill: '#F1EFE8', stroke: '#888780', text: '#444441' }},
    resultado: {{ fill: '#EAF3DE', stroke: '#639922', text: '#27500A' }},
  }};

  for (let i = 0; i < LAYOUT.length - 1; i++) {{
    const a = LAYOUT[i], b = LAYOUT[i+1];
    trilhasLayer.appendChild(el('line', {{
      x1: a.x + WE, y1: YT + HE/2, x2: b.x, y2: YT + HE/2,
      stroke: '#9c9a92', 'stroke-width': 1, opacity: 0.35, 'marker-end': 'url(#arrowc)'
    }}));
  }}

  LAYOUT.forEach((s, idx) => {{
    const cor = CORES[s.tipo] || CORES.humano;
    const x = s.x, y = YT;
    const cont = CONTADORES[s.id] || {{ total: 0, aprovados: 0 }};
    const fila = FILAS[s.id] || 0;

    if (fila > 0 && idx > 0) {{
      const xFila = x - 18;
      const nMostrar = Math.min(fila, MAXB);
      for (let b = 0; b < nMostrar; b++) {{
        filasLayer.appendChild(el('circle', {{
          cx: xFila, cy: (y + HE/2) - 10 - b * 9, r: 4.5,
          fill: '#9c9a92', opacity: 0.85
        }}));
      }}
      if (fila > MAXB) {{
        const txt = el('text', {{
          x: xFila, y: (y + HE/2) - 10 - MAXB * 9 - 8, 'text-anchor': 'middle',
          'font-size': 9, fill: '#9c9a92', 'font-weight': 600
        }});
        txt.textContent = '+' + (fila - MAXB);
        filasLayer.appendChild(txt);
      }}
      const labelFila = el('text', {{
        x: xFila, y: y + HE + 16, 'text-anchor': 'middle',
        'font-size': 9.5, fill: '#888780', 'font-weight': 500
      }});
      labelFila.textContent = fila + ' esperando';
      filasLayer.appendChild(labelFila);
    }}

    const rect = el('rect', {{
      x: x, y: y, width: WE, height: HE, rx: 10,
      fill: cor.fill, stroke: cor.stroke, 'stroke-width': 1, id: 'st-' + s.id
    }});
    estacoesLayer.appendChild(rect);

    const label = el('text', {{
      x: x + WE/2, y: y + 19, 'text-anchor': 'middle',
      'font-size': 12, 'font-weight': 500, fill: cor.text
    }});
    label.textContent = s.nome;
    estacoesLayer.appendChild(label);

    if (s.modelo) {{
      const modeloTxt = el('text', {{
        x: x + WE/2, y: y + 34, 'text-anchor': 'middle',
        'font-size': 9.5, fill: cor.text, opacity: 0.8, 'font-weight': 500
      }});
      modeloTxt.textContent = s.modelo;
      estacoesLayer.appendChild(modeloTxt);
    }}

    const sub = el('text', {{
      x: x + WE/2, y: y + (s.modelo ? 49 : 38), 'text-anchor': 'middle',
      'font-size': 10, fill: cor.text, opacity: 0.75, id: 'sub-' + s.id
    }});
    const detalheInicial = ULTIMO_DETALHE[s.id];
    if (s.tipo === 'ia_score' && detalheInicial && detalheInicial.probabilidade_fechamento !== undefined) {{
      sub.textContent = 'score: ' + (detalheInicial.probabilidade_fechamento * 100).toFixed(0) + '%';
    }} else if (s.tipo === 'ia_knn') {{
      sub.textContent = 'pronto';
    }} else if (s.tipo === 'ia_llm') {{
      sub.textContent = 'pronto';
    }}
    estacoesLayer.appendChild(sub);

    const contagem = el('text', {{
      x: x + WE/2, y: y + HE - 8, 'text-anchor': 'middle',
      'font-size': 9.5, fill: cor.text, opacity: 0.6
    }});
    contagem.textContent = cont.total + ' concluídos';
    estacoesLayer.appendChild(contagem);
  }});

  let delay = 0;
  EVENTOS_NOVOS.forEach((ev) => {{
    const idx = LAYOUT.findIndex(s => s.id === ev.etapa);
    if (idx === -1) return;
    const s = LAYOUT[idx];
    const sAnterior = idx > 0 ? LAYOUT[idx - 1] : null;

    setTimeout(() => {{
      const xDestino = s.x + WE/2;
      const xOrigem = sAnterior ? (sAnterior.x + WE) : (s.x - 20);
      const yLinha = YT + HE/2;

      const rect = document.getElementById('st-' + s.id);
      if (rect) {{
        rect.setAttribute('stroke-width', '2.5');
        setTimeout(() => {{ if (rect) rect.setAttribute('stroke-width', '1'); }}, 600);
      }}

      const sub = document.getElementById('sub-' + s.id);
      if (s.tipo === 'ia_score' && ev.detalhes && ev.detalhes.probabilidade_fechamento !== undefined) {{
        const p = (ev.detalhes.probabilidade_fechamento * 100).toFixed(0);
        if (sub) sub.textContent = 'score: ' + p + '%';
      }} else if (s.tipo === 'ia_knn') {{
        if (sub) {{ sub.textContent = 'comparando...'; setTimeout(() => {{ if (sub) sub.textContent = 'pronto'; }}, 500); }}
      }} else if (s.tipo === 'ia_llm') {{
        if (sub) {{ sub.textContent = 'escrevendo...'; setTimeout(() => {{ if (sub) sub.textContent = 'pronto'; }}, 600); }}
      }}

      const bolinha = el('circle', {{ cx: xOrigem, cy: yLinha, r: 5, fill: '#378ADD', opacity: 0.95 }});
      transitoLayer.appendChild(bolinha);

      const t0 = performance.now();
      const duracao = 600;
      function anima(t) {{
        const p = Math.min(1, (t - t0) / duracao);
        bolinha.setAttribute('cx', xOrigem + (xDestino - xOrigem) * p);
        if (p < 1) {{
          requestAnimationFrame(anima);
        }} else {{
          bolinha.setAttribute('fill', ev.passou ? '#1D9E75' : '#E24B4A');
          bolinha.setAttribute('r', 6);
          setTimeout(() => bolinha.remove(), 450);
        }}
      }}
      requestAnimationFrame(anima);

      const resultado = ev.passou ? 'avançou' : 'não avançou';
      statusLine.textContent = 'Lead ' + ev.lead_id + ' → ' + s.nome + ' (' + resultado + ')';
    }}, delay);
    delay += 140;
  }});

  if (EVENTOS_NOVOS.length === 0 && TOTAL_NO_SISTEMA > 0) {{
    statusLine.textContent = TOTAL_NO_SISTEMA + ' leads no sistema até agora.';
  }} else if (TOTAL_NO_SISTEMA === 0) {{
    statusLine.textContent = 'Aguardando início da simulação...';
  }}
}})();
</script>
"""
