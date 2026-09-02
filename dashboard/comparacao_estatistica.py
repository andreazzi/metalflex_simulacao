"""
comparacao_estatistica.py

Aba de comparacao estatistica entre os 4 estados (A, B, C, D), a partir do
CSV agregado gerado por experimento_30x.py -- nao do modo "ao vivo" (que
so' guarda uma execucao por estado). Le em modo tolerante a arquivo
incompleto: funciona mesmo com uma rodada ainda em andamento, mostrando
quantas replicas de cada estado ja' chegaram.

Os arquivos sao nomeados pelo numero de replicas (experimento_Nx_*.csv, ver
experimento_30x.py) para nunca colidir entre um teste rapido e a rodada
oficial -- esta aba acha automaticamente o N oficial mais recente (nunca um
arquivo _smoke) em vez de um nome fixo.
"""

import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

PASTA_SAIDAS = "../saidas"

CORES_ESTADO = {"Estado A": "#888780", "Estado B": "#D85A30", "Estado C": "#1D9E75", "Estado D": "#3A6EA5"}
# Ordem de exibicao A -> B -> D -> C (nao alfabetica): B e D lado a lado por
# terem a mesma cardinalidade de intervencao; C fecha como redesenho
# completo. Ver ESPEC_DASHBOARD_4_ESTADOS.md.
ORDEM_ESTADOS = ["Estado A", "Estado B", "Estado D", "Estado C"]


def _n_replicas_oficial() -> int | None:
    """Acha o N do experimento_Nx_replicas.csv oficial mais recente (maior N,
    nunca _smoke) -- os outros dois arquivos da mesma rodada (decomposicao,
    handoff) usam o mesmo N no nome."""
    candidatos = [
        os.path.basename(f) for f in glob.glob(os.path.join(PASTA_SAIDAS, "experimento_*x_replicas.csv"))
        if "_smoke" not in f
    ]
    if not candidatos:
        return None
    ns = []
    for nome in candidatos:
        try:
            ns.append(int(nome.split("_")[1].rstrip("x")))
        except (IndexError, ValueError):
            continue
    return max(ns) if ns else None


def _caminhos_rodada_oficial():
    n = _n_replicas_oficial()
    if n is None:
        return None, None, None
    base = f"experimento_{n}x"
    return (
        os.path.join(PASTA_SAIDAS, f"{base}_replicas.csv"),
        os.path.join(PASTA_SAIDAS, f"{base}_decomposicao_etapas.csv"),
        os.path.join(PASTA_SAIDAS, f"{base}_diagnostico_handoff.csv"),
    )


@st.cache_data(ttl=5)
def _carregar_csv(caminho: str) -> pd.DataFrame:
    if not os.path.exists(caminho):
        return pd.DataFrame()
    try:
        return pd.read_csv(caminho)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _br(x, dec):
    if pd.isna(x):
        return "—"
    return f"{x:.{dec}f}".replace(".", ",")


def _teste_pareado(df: pd.DataFrame, estado_x: str, estado_y: str, coluna: str):
    """So' usa replicas presentes nos DOIS estados -- funciona com dados parciais."""
    x = df[df.estado == estado_x].set_index("repeticao")[coluna]
    y = df[df.estado == estado_y].set_index("repeticao")[coluna]
    reps_comuns = x.index.intersection(y.index)
    if len(reps_comuns) < 2:
        return None
    xv, yv = x.loc[reps_comuns].values, y.loc[reps_comuns].values
    diff = xv - yv
    n = len(diff)
    sd = diff.std(ddof=1)
    if sd == 0:
        return {"n": n, "diff": 0.0, "t": None, "p": None, "d": None, "ci": (0.0, 0.0)}
    t, p = stats.ttest_rel(xv, yv)
    d = diff.mean() / sd
    se = sd / np.sqrt(n)
    ci_meia = stats.t.ppf(0.975, n - 1) * se
    return {"n": n, "diff": diff.mean(), "t": t, "p": p, "d": d, "ci": (diff.mean() - ci_meia, diff.mean() + ci_meia)}


def renderizar_comparacao_estatistica():
    st.subheader("Comparação estatística entre os 4 estados")
    st.caption(
        "Lê o `experimento_Nx_replicas.csv` oficial mais recente — o desenho pareado por "
        "(repetição, lead, etapa), não o modo \"ao vivo\". Atualiza sozinho a cada poucos "
        "segundos enquanto uma rodada roda em background."
    )

    caminho_replicas, caminho_decomposicao, caminho_handoff = _caminhos_rodada_oficial()
    if caminho_replicas is None:
        st.info("Nenhuma rodada oficial ainda (`experimento_Nx_replicas.csv`). "
                "Rode `experimento_30x.py --n-repeticoes N` no terminal.")
        return

    df = _carregar_csv(caminho_replicas)
    if df.empty:
        st.info(f"`{os.path.basename(caminho_replicas)}` ainda vazio ou sendo gerado.")
        return
    st.caption(f"Fonte: `{os.path.basename(caminho_replicas)}`")

    # --- progresso ---
    contagem = df.groupby("estado")["repeticao"].nunique().reindex(ORDEM_ESTADOS).fillna(0).astype(int)
    total_esperado = df["repeticao"].max()
    cols = st.columns(4)
    for col, estado in zip(cols, ORDEM_ESTADOS):
        n = contagem.get(estado, 0)
        with col:
            st.metric(estado.replace("Estado ", ""), f"{n}/{total_esperado}",
                      help=f"Réplicas concluídas de {estado}")
    if contagem.sum() < total_esperado * 4:
        st.progress(min(1.0, contagem.sum() / (total_esperado * 4)), text="Rodada em andamento...")

    estados_presentes = [e for e in ORDEM_ESTADOS if contagem.get(e, 0) > 0]
    if not estados_presentes:
        return

    # --- descritivas ---
    st.markdown("#### Descritivas")
    desc = df[df.estado.isin(estados_presentes)].groupby("estado").agg(
        n=("repeticao", "nunique"),
        fechados_media=("negocios_fechados", "mean"), fechados_dp=("negocios_fechados", "std"),
        conversao_media=("taxa_conversao_pct", "mean"), conversao_dp=("taxa_conversao_pct", "std"),
        tempo_media=("tempo_total_h", "mean"), tempo_dp=("tempo_total_h", "std"),
    ).reindex([e for e in ORDEM_ESTADOS if e in estados_presentes])
    tabela = pd.DataFrame({
        "Estado": desc.index,
        "n": desc["n"].astype(int),
        "Negócios fechados": [f"{_br(m,1)} ± {_br(d,1)}" for m, d in zip(desc.fechados_media, desc.fechados_dp)],
        "Taxa de conversão": [f"{_br(m,2)}% ± {_br(d,2)}" for m, d in zip(desc.conversao_media, desc.conversao_dp)],
        "Tempo total": [f"{_br(m,0)}h ± {_br(d,0)}" for m, d in zip(desc.tempo_media, desc.tempo_dp)],
    })
    st.dataframe(tabela, hide_index=True, use_container_width=True)

    # --- graficos ---
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(
            x=list(desc.index), y=desc.conversao_media,
            error_y=dict(type="data", array=desc.conversao_dp),
            text=[f"{_br(v,2)}%" for v in desc.conversao_media], textposition="outside",
            marker_color=[CORES_ESTADO[e] for e in desc.index],
        ))
        fig.update_layout(title="Taxa de conversão por estado", height=340,
                          yaxis=dict(range=[0, max(desc.conversao_media) * 1.4]))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Bar(
            x=list(desc.index), y=desc.tempo_media,
            error_y=dict(type="data", array=desc.tempo_dp),
            text=[f"{_br(v,0)}h" for v in desc.tempo_media], textposition="outside",
            marker_color=[CORES_ESTADO[e] for e in desc.index],
        ))
        fig.update_layout(title="Tempo total simulado por estado", height=340,
                          yaxis=dict(range=[0, max(desc.tempo_media) * 1.4]))
        st.plotly_chart(fig, use_container_width=True)

    # --- checkpoints de identidade (F-15/F-11) ---
    st.markdown("#### Checkpoints de identidade")
    st.caption("As duas comparações mais fortes possíveis: sob pareamento, a diferença deve ser exatamente zero.")
    c1, c2 = st.columns(2)
    for col, (ex, ey, label) in zip(
        [c1, c2],
        [("Estado A", "Estado B", "A ≡ B em conversão"), ("Estado C", "Estado D", "C ≡ D em conversão")],
    ):
        with col:
            if ex in estados_presentes and ey in estados_presentes:
                r = _teste_pareado(df, ex, ey, "taxa_conversao_pct")
                if r and r["diff"] == 0.0:
                    st.success(f"✅ {label} — diferença nula em {r['n']} réplicas comuns")
                elif r:
                    st.warning(f"⚠️ {label} — diferença de {_br(r['diff'],4)} p.p. em {r['n']} réplicas (esperada: 0)")
                else:
                    st.info(f"{label} — aguardando réplicas suficientes")
            else:
                st.info(f"{label} — aguardando dados de {ex if ex not in estados_presentes else ey}")

    # --- estatistica pareada completa ---
    st.markdown("#### Teste t pareado (todas as comparações disponíveis)")
    pares_conversao = [("Estado D", "Estado B"), ("Estado D", "Estado A"), ("Estado D", "Estado C"),
                        ("Estado C", "Estado A"), ("Estado C", "Estado B"), ("Estado A", "Estado B")]
    pares_tempo = pares_conversao + [("Estado B", "Estado A")]
    linhas = []
    for metrica, coluna, unidade, pares in [
        ("Conversão", "taxa_conversao_pct", "p.p.", pares_conversao),
        ("Tempo", "tempo_total_h", "h", pares_tempo),
    ]:
        for ex, ey in pares:
            if ex not in estados_presentes or ey not in estados_presentes:
                continue
            r = _teste_pareado(df, ex, ey, coluna)
            if r is None:
                continue
            linhas.append({
                "Métrica": metrica,
                "Comparação": f"{ex.replace('Estado ','')} × {ey.replace('Estado ','')}",
                "n": r["n"],
                "Diferença": f"{_br(r['diff'],3)} {unidade}",
                "t": "—" if r["t"] is None else _br(r["t"], 2),
                "p": ("identidade exata" if r["t"] is None else
                      ("< 0,001" if r["p"] < 0.001 else _br(r["p"], 3))),
                "d de Cohen": "—" if r["d"] is None else _br(r["d"], 2),
                "IC 95%": f"[{_br(r['ci'][0],3)}; {_br(r['ci'][1],3)}] {unidade}",
            })
    if linhas:
        st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)

    # --- decomposicao por etapa ---
    dec = _carregar_csv(caminho_decomposicao)
    if not dec.empty:
        st.markdown("#### Decomposição por etapa")
        st.caption("Horas médias e leads processados por etapa — calculada só ao final da rodada.")
        st.dataframe(dec, hide_index=True, use_container_width=True)

    # --- diagnostico de handoff ---
    handoff = _carregar_csv(caminho_handoff)
    if not handoff.empty:
        st.markdown("#### Diagnóstico de comparabilidade de escala (handoff)")
        st.dataframe(handoff, hide_index=True, use_container_width=True)
