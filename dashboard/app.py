"""
app.py — Dashboard Streamlit do experimento Metalflex.

Tem dois modos de observação:
  - MODO AO VIVO: lança o motor de simulação como subprocesso (com pausa
    entre cada lead) e acompanha o jsonl crescendo em tempo real -- mostra
    qual lead está sendo processado agora, o funil enchendo aos poucos, e
    os KPIs se atualizando minuto a minuto da simulação.
  - MODO HISTÓRICO: lê um jsonl já completo (de uma execução anterior) e
    mostra o resultado consolidado, sem replay.

E uma tela final de comparacao A vs B vs C lado a lado.

Uso:
    streamlit run app.py
"""

import datetime
import json
import os
import subprocess
import sys
import time

# O processo Streamlit carrega matplotlib, plotly e faz reruns a cada 1s com
# grandes DataFrames. Em Python 3.14 o limite padrão de 1000 pode ser
# atingido durante rendering de plot_tree ou serialização interna.
sys.setrecursionlimit(3000)

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

from canvas_processo import gerar_html_canvas
from analise_modelos import renderizar_analise
from setup_experimento import renderizar_setup
from resultados_completos import renderizar_resultados_completos
from comparacao_estatistica import renderizar_comparacao_estatistica

st.set_page_config(page_title="Metalflex — Simulação Comercial B2B", layout="wide")

# Fundo padronizado para todos os gráficos Plotly do dashboard.
# Feito via patch em go.Figure.__init__ porque pio.templates não é aplicado
# pelo caminho que o Streamlit usa (pio.to_json). O patch afeta automaticamente
# todos os módulos no mesmo processo (analise_modelos, setup_experimento, etc.)
# e também gráficos criados com plotly.express.
_CHART_BG = "#E0DDD6"
_orig_Figure_init = go.Figure.__init__

def _Figure_init_com_fundo(self, *args, **kwargs):
    _orig_Figure_init(self, *args, **kwargs)
    if self.layout.paper_bgcolor is None:
        self.layout.paper_bgcolor = _CHART_BG
    if self.layout.plot_bgcolor is None:
        self.layout.plot_bgcolor = _CHART_BG

go.Figure.__init__ = _Figure_init_com_fundo

# Exportação de gráfico em arquivo, nao dependente de captura de tela (ver
# ESPEC_DASHBOARD_4_ESTADOS.md): todo grafico Plotly do dashboard ja' tem um
# botao de download na propria barra de ferramentas (canto superior direito
# ao passar o mouse) -- so' precisava ser configurado para exportar em SVG
# (vetorial, qualidade de publicacao em qualquer tamanho) em vez do PNG de
# baixa resolucao que e' o padrao do Plotly. Corrigido aqui, uma vez so',
# via patch em st.plotly_chart -- afeta todos os graficos do app (mesmo
# raciocinio do patch de cor de fundo acima), sem precisar editar as
# dezenas de chamadas espalhadas pelos outros modulos do dashboard.
_orig_plotly_chart = st.plotly_chart

def _plotly_chart_com_export_svg(*args, **kwargs):
    if "config" not in kwargs:
        kwargs["config"] = {
            "toImageButtonOptions": {"format": "svg", "filename": "grafico_metalflex"},
            "displaylogo": False,
        }
    return _orig_plotly_chart(*args, **kwargs)

st.plotly_chart = _plotly_chart_com_export_svg

PASTA_DADOS = "../dados"
PASTA_SAIDAS = "../saidas"
ARQUIVOS = {
    "A": os.path.join(PASTA_SAIDAS, "historico_estado_a.jsonl"),
    "B": os.path.join(PASTA_SAIDAS, "historico_estado_b.jsonl"),
    "C": os.path.join(PASTA_SAIDAS, "historico_estado_c.jsonl"),
    "D": os.path.join(PASTA_SAIDAS, "historico_estado_d.jsonl"),
}

NOMES_ESTADOS = {
    "A": "Estado A — Baseline",
    "B": "Estado B — Melhoria pontual (IA na proposta)",
    "C": "Estado C — Orquestração sistêmica",
    "D": "Estado D — Ablação (só qualificação por IA)",
}

ETAPAS_POR_ESTADO = {
    "A": ["geracao_leads", "qualificacao_sdr", "descoberta_closer", "proposta_manual",
          "negociacao", "fechamento", "pos_venda"],
    "B": ["geracao_leads", "qualificacao_sdr", "descoberta_closer", "proposta_ia",
          "negociacao", "fechamento", "pos_venda"],
    "C": ["geracao_leads", "qualificacao_ia", "match_closer_ia", "descoberta_closer",
          "proposta_ia", "negociacao", "fechamento", "pos_venda"],
    "D": ["geracao_leads", "qualificacao_ia", "descoberta_closer", "proposta_manual",
          "negociacao", "fechamento", "pos_venda"],
}

CORES_ESTADO = {"A": "#888780", "B": "#D85A30", "C": "#1D9E75", "D": "#3A6EA5"}

ETAPA_PROPOSTA = {"A": "proposta_manual", "B": "proposta_ia", "C": "proposta_ia", "D": "proposta_manual"}

# Etapas em que uma decisao e' tomada por um modelo de IA (nao um humano) --
# usado para contar "numero de intervencoes de IA" por estado.
ETAPAS_IA = {"qualificacao_ia", "match_closer_ia", "proposta_ia"}


def n_intervencoes_ia(estado: str) -> int:
    return sum(1 for e in ETAPAS_POR_ESTADO[estado] if e in ETAPAS_IA)


def encontrar_replicas_oficial() -> str | None:
    """Acha o CSV de replicas oficial mais recente (experimento_Nx_replicas.csv,
    nunca o _smoke) -- usado pelas abas que provam as identidades exatas A≡B e
    C≡D a partir do desenho pareado, nao de uma unica execucao "ao vivo" (que
    poderia comparar replicas diferentes de cada estado por acidente)."""
    candidatos = [
        f for f in os.listdir(PASTA_SAIDAS)
        if f.startswith("experimento_") and f.endswith("x_replicas.csv") and "_smoke" not in f
    ]
    if not candidatos:
        return None
    # prioriza o N mais alto (rodada mais robusta), depois o mais recente
    def chave(nome):
        try:
            n = int(nome.split("_")[1].rstrip("x"))
        except (IndexError, ValueError):
            n = 0
        return (n, os.path.getmtime(os.path.join(PASTA_SAIDAS, nome)))
    candidatos.sort(key=chave, reverse=True)
    return os.path.join(PASTA_SAIDAS, candidatos[0])

NOMES_ETAPAS_LEGIVEIS = {
    "geracao_leads": "Geração de leads",
    "qualificacao_sdr": "Qualificação (SDR humano)",
    "qualificacao_ia": "Qualificação (IA — RandomForest)",
    "match_closer_ia": "Match de closer (kNN)",
    "descoberta_closer": "Descoberta (reunião com Closer)",
    "proposta_manual": "Proposta (manual)",
    "proposta_ia": "Proposta (IA — LLM local)",
    "negociacao": "Negociação",
    "fechamento": "Fechamento",
    "pos_venda": "Pós-venda",
}

VELOCIDADES = {
    "Lenta (1.0s/lead — bom para observar com calma)": "lenta",
    "Normal (0.3s/lead)": "normal",
    "Rápida (0.05s/lead)": "rapida",
}


@st.cache_data(ttl=2)
def carregar_eventos(caminho: str) -> pd.DataFrame:
    if not os.path.exists(caminho):
        return pd.DataFrame()
    linhas = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                try:
                    linhas.append(json.loads(linha))
                except json.JSONDecodeError:
                    continue  # ultima linha pode estar sendo escrita agora
    return pd.DataFrame(linhas)


def calcular_funil(df: pd.DataFrame, etapas: list) -> pd.DataFrame:
    contagens = []
    for etapa in etapas:
        df_etapa = df[df["etapa"] == etapa]
        total_na_etapa = df_etapa["lead_id"].nunique()
        passou = df_etapa[df_etapa["passou"]]["lead_id"].nunique()
        contagens.append({"etapa": etapa, "entraram": total_na_etapa, "passaram": passou})
    return pd.DataFrame(contagens)


def grafico_funil(funil_df: pd.DataFrame, titulo: str, etapas_legiveis: dict):
    labels = [etapas_legiveis.get(e, e) for e in funil_df["etapa"]]
    fig = go.Figure(go.Funnel(
        y=labels,
        x=funil_df["passaram"],
        textinfo="value+percent initial",
    ))
    fig.update_layout(title=titulo, height=380, margin=dict(t=40, b=10, l=10, r=10))
    return fig


def metricas_resumo(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_leads": 0, "fechados": 0, "taxa_conversao": 0.0, "tempo_total_h": 0.0}
    total_leads = df["lead_id"].nunique()
    fechados = df[(df["etapa"] == "fechamento") & (df["passou"])]["lead_id"].nunique()
    tempo_total_h = df["tempo_gasto_min"].sum() / 60
    return {
        "total_leads": total_leads,
        "fechados": fechados,
        "taxa_conversao": fechados / total_leads if total_leads else 0,
        "tempo_total_h": tempo_total_h,
    }


def processo_rodando(chave_estado: str) -> bool:
    proc = st.session_state.get(f"proc_{chave_estado}")
    return proc is not None and proc.poll() is None


def processo_terminou_com_erro(chave_estado: str) -> str | None:
    """Retorna a mensagem de erro se o subprocesso terminou com codigo != 0,
    ou None se nao rodou, ainda esta rodando, ou terminou normalmente."""
    proc = st.session_state.get(f"proc_{chave_estado}")
    if proc is None or proc.poll() is None:
        return None
    if proc.returncode != 0:
        log_path = os.path.join(PASTA_DADOS, f"log_ultima_execucao_{chave_estado}.txt")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                conteudo = f.read()
            return conteudo[-1500:]  # ultimos caracteres, o mais relevante pra debug
        return f"Processo terminou com código de erro {proc.returncode}, sem log disponível."
    return None


def iniciar_simulacao_ao_vivo(estado: str, velocidade: str, n_leads: int | None,
                               repeticao: int, exploratorio: bool):
    cmd = [sys.executable, "rodar_ao_vivo.py", "--estado", estado, "--velocidade", velocidade]
    if n_leads:
        cmd += ["--n-leads", str(n_leads)]
    if exploratorio:
        cmd += ["--exploratorio"]
    else:
        cmd += ["--repeticao", str(repeticao)]
    # redireciona stdout/stderr para um ARQUIVO em vez de PIPE -- um PIPE
    # nao lido pode encher o buffer do SO e travar o processo filho em
    # execucoes longas; um arquivo nao tem esse limite e ainda permite
    # diagnosticar erros depois (ve processo_terminou_com_erro acima).
    log_path = os.path.join(PASTA_DADOS, f"log_ultima_execucao_{estado}.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, cwd=PASTA_DADOS,
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    st.session_state[f"proc_{estado}"] = proc


def renderizar_comparacao(estados: list, subtitulo: str, caption_topo: str,
                           caption_conversao: str, caption_tempo: str, caption_nota: str):
    """Compara N estados lado a lado a partir dos arquivos de histórico "ao vivo"
    mais recentes de cada um -- mesma lógica para qualquer subconjunto de
    estados, só as legendas mudam (o que cada comparação especificamente
    demonstra)."""
    st.subheader(subtitulo)
    st.caption(caption_topo)

    n_total = len(estados)
    estados_disponiveis = [
        e for e in estados
        if os.path.exists(ARQUIVOS[e]) and os.path.getsize(ARQUIVOS[e]) > 0
    ]

    if not estados_disponiveis:
        st.info("Nenhum estado foi executado ainda. Rode pelo menos um estado na aba 'Rodar ao vivo'.")
        return

    st.markdown("**Execuções disponíveis:**")
    cols_disp = st.columns(n_total)
    for i, estado in enumerate(estados):
        caminho = ARQUIVOS[estado]
        with cols_disp[i]:
            if os.path.exists(caminho) and os.path.getsize(caminho) > 0:
                mtime = os.path.getmtime(caminho)
                data_str = datetime.datetime.fromtimestamp(mtime).strftime("%d/%m %H:%M")
                st.success(f"**{NOMES_ESTADOS[estado].split(' —')[0]}** — {data_str}")
            else:
                st.warning(f"**{NOMES_ESTADOS[estado].split(' —')[0]}** — não executado ainda")

    if len(estados_disponiveis) < n_total:
        st.info(
            f"{len(estados_disponiveis)}/{n_total} estados executados. "
            "Execute os estados faltantes na aba 'Rodar ao vivo' para comparação completa."
        )

    st.divider()

    dados_comparacao = []
    for estado in estados:
        df_e = carregar_eventos(ARQUIVOS[estado])
        m = metricas_resumo(df_e)
        tempo_proposta = "—"
        if not df_e.empty and "etapa" in df_e.columns:
            props = df_e[df_e["etapa"] == ETAPA_PROPOSTA[estado]]
            if not props.empty:
                tempo_proposta = f"{props['tempo_gasto_min'].mean():.1f} min"
        dados_comparacao.append({
            "Estado": NOMES_ESTADOS[estado],
            "Leads processados": m["total_leads"],
            "Negócios fechados": m["fechados"],
            "Taxa de conversão": f"{m['taxa_conversao']:.1%}" if m["total_leads"] > 0 else "—",
            "Tempo médio/proposta": tempo_proposta,
            "Tempo total (h)": f"{m['tempo_total_h']:.1f}" if m["total_leads"] > 0 else "—",
        })

    df_comp = pd.DataFrame(dados_comparacao)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    if len(estados_disponiveis) >= 2:
        dados_graf = []
        for estado in estados_disponiveis:
            df_e = carregar_eventos(ARQUIVOS[estado])
            m = metricas_resumo(df_e)
            if m["total_leads"] > 0:
                dados_graf.append({
                    "Estado": NOMES_ESTADOS[estado].split(" —")[0],
                    "cor": CORES_ESTADO[estado],
                    "taxa": m["taxa_conversao"],
                    "tempo": m["tempo_total_h"],
                })
        df_g = pd.DataFrame(dados_graf)

        col1, col2 = st.columns(2)
        with col1:
            fig_conv = go.Figure(go.Bar(
                x=df_g["Estado"], y=df_g["taxa"],
                text=[f"{v:.1%}" for v in df_g["taxa"]],
                textposition="outside", marker_color=df_g["cor"],
            ))
            fig_conv.update_layout(
                title="Taxa de conversão por estado",
                height=360, yaxis_tickformat=".0%",
                yaxis=dict(range=[0, max(df_g["taxa"]) * 1.3]),
            )
            st.plotly_chart(fig_conv, use_container_width=True)
            st.caption(caption_conversao)

        with col2:
            fig_tempo = go.Figure(go.Bar(
                x=df_g["Estado"], y=df_g["tempo"],
                text=[f"{v:.0f}h" for v in df_g["tempo"]],
                textposition="outside", marker_color=df_g["cor"],
            ))
            fig_tempo.update_layout(
                title="Tempo total investido por estado (horas)",
                height=360,
                yaxis=dict(range=[0, max(df_g["tempo"]) * 1.3]),
            )
            st.plotly_chart(fig_tempo, use_container_width=True)
            st.caption(caption_tempo)

        st.caption(caption_nota)


def renderizar_identidades_e_intervencoes(estados_ordem: list):
    """Torna visiveis, num so' lugar: (1) o numero de intervencoes de IA de
    cada estado: (2) as duas identidades exatas (A≡B, C≡D), provadas contra
    TODAS as replicas do desenho pareado -- nao uma unica execucao "ao vivo",
    que poderia comparar replicas diferentes por acidente; (3) a inversao
    entre dimensoes (B melhor em tempo, D melhor em conversao). Ver
    ESPEC_DASHBOARD_4_ESTADOS.md."""
    st.markdown("#### Intervenções de IA por estado")
    cols = st.columns(len(estados_ordem))
    for col, estado in zip(cols, estados_ordem):
        n = n_intervencoes_ia(estado)
        with col:
            st.metric(NOMES_ESTADOS[estado].split(" —")[0], f"{n} etapa{'s' if n != 1 else ''}")
    st.caption(
        "A = 0 (baseline). B e D = 1 cada, em locais diferentes (proposta vs. qualificação). "
        "C = 3 (qualificação + match + proposta) — o redesenho completo."
    )

    st.markdown("#### Identidades exatas (desenho pareado, todas as réplicas)")
    caminho_replicas = encontrar_replicas_oficial()
    if caminho_replicas is None:
        st.info("Nenhuma rodada oficial (`experimento_Nx_replicas.csv`) encontrada ainda. "
                "Rode `experimento_30x.py --n-repeticoes N` no terminal para gerar uma.")
    else:
        df_rep = pd.read_csv(caminho_replicas)
        nome_arquivo = os.path.basename(caminho_replicas)
        n_replicas = df_rep["repeticao"].nunique()
        col1, col2 = st.columns(2)
        for col, (ex, ey, label) in zip(
            [col1, col2],
            [("Estado A", "Estado B", "A ≡ B em conversão"), ("Estado C", "Estado D", "C ≡ D em conversão")],
        ):
            x = df_rep[df_rep.estado == ex].set_index("repeticao")["taxa_conversao_pct"]
            y = df_rep[df_rep.estado == ey].set_index("repeticao")["taxa_conversao_pct"]
            reps_comuns = x.index.intersection(y.index)
            with col:
                if len(reps_comuns) == 0:
                    st.info(f"{label} — sem réplicas em comum em {nome_arquivo}")
                elif (x.loc[reps_comuns].values == y.loc[reps_comuns].values).all():
                    st.success(f"✅ {label} — diferença nula em {len(reps_comuns)} réplicas", icon="✅")
                else:
                    n_dif = int((x.loc[reps_comuns].values != y.loc[reps_comuns].values).sum())
                    st.warning(f"⚠️ {label} — diverge em {n_dif}/{len(reps_comuns)} réplicas — investigar")
        st.caption(f"Fonte: `{nome_arquivo}` ({n_replicas} réplicas oficiais, desenho pareado).")

        # inversao entre dimensoes: B melhor em tempo, D melhor em conversao
        if all(e in df_rep.estado.unique() for e in ["Estado B", "Estado D"]):
            b = df_rep[df_rep.estado == "Estado B"]
            d = df_rep[df_rep.estado == "Estado D"]
            st.markdown("#### A inversão entre dimensões")
            colb, cold = st.columns(2)
            with colb:
                st.markdown(f"**{NOMES_ESTADOS['B'].split(' —')[0]}**")
                st.metric("Tempo médio", f"{b.tempo_total_h.mean():.0f}h", help="Menor é melhor")
                st.metric("Conversão média", f"{b.taxa_conversao_pct.mean():.2f}%")
            with cold:
                st.markdown(f"**{NOMES_ESTADOS['D'].split(' —')[0]}**")
                st.metric("Tempo médio", f"{d.tempo_total_h.mean():.0f}h", help="Maior que B")
                st.metric("Conversão média", f"{d.taxa_conversao_pct.mean():.2f}%", help="Maior que B")
            st.caption(
                "**B vence em tempo, D vence em conversão** — mesma cardinalidade de intervenção "
                "(uma etapa cada), resultado invertido conforme o local escolhido. É o argumento "
                "central do capítulo: não é quanto se automatiza, é onde."
            )


st.title("Metalflex — simulação de processo comercial B2B")
st.caption("Acompanhe a operação em tempo real, ou compare resultados já consolidados de cada estado")

(aba_setup, aba_ao_vivo, aba_historico, aba_comparacao, aba_comp_bd, aba_comp_cd,
 aba_analise, aba_resultados, aba_estatistica) = st.tabs(
    ["⚙️ Setup do experimento", "▶ Rodar ao vivo",
     "📋 Ver histórico", "📊 Comparar A / B / C / D", "🔀 Comparar B vs D", "🎯 Comparar C vs D",
     "🔬 Análise dos modelos de IA",
     "📄 Resultados completos (B/C/D)", "📈 Comparação estatística (4 estados)"]
)

# ============================================================
# ABA 0 — SETUP DO EXPERIMENTO
# ============================================================
with aba_setup:
    renderizar_setup()

# ============================================================
# ABA 1 — MODO AO VIVO
# Hierarquia visual: controles compactos no topo → processo em destaque
# no centro (canvas + decisões de IA logo abaixo dele) → dashboard de
# números como contexto secundário ao final da página.
# ============================================================
with aba_ao_vivo:
    # --- LINHA DE CONTROLES, compacta, no topo ---
    # O truque de CSS abaixo alinha o botão verticalmente com os campos
    # de input sem criar espaço extra -- st.write("") empurrava os botões
    # para baixo porque adicionava a altura de uma linha de texto antes deles.
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-child(4) .stButton button,
    div[data-testid="column"]:nth-child(5) .stButton button {
        margin-top: 25px;
        height: 38px;
        padding: 0 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1.5, 1.2, 1.0, 0.6, 0.6])

    # verifica se modelos estao disponiveis. Estado C precisa dos tres
    # (score + match + scaler); Estado D so' do score -- ver rodar_ao_vivo.py.
    modelo_score_disponivel = (
        os.path.exists(os.path.join("../modelos", "modelo_score.pkl"))
        and os.path.getsize(os.path.join("../modelos", "modelo_score.pkl")) > 0
    )
    modelos_disponiveis = modelo_score_disponivel and all(
        os.path.exists(p) and os.path.getsize(p) > 0
        for p in [
            os.path.join("../modelos", "modelo_match.pkl"),
            os.path.join("../modelos", "scaler_match.pkl"),
        ]
    )
    ESTADOS_SEM_MODELO = {"C": modelos_disponiveis, "D": modelo_score_disponivel}

    with ctrl1:
        def formatar_estado(x):
            nome = NOMES_ESTADOS[x]
            if x in ESTADOS_SEM_MODELO and not ESTADOS_SEM_MODELO[x]:
                return f"{nome} ⚠️ (treinar modelos primeiro)"
            return nome

        opcoes_vivo = ["A", "B", "C", "D"]
        estado_vivo = st.selectbox(
            "Estado a simular", options=opcoes_vivo,
            format_func=formatar_estado, key="sel_estado_vivo",
        )
    with ctrl2:
        velocidade_label = st.selectbox("Velocidade", options=list(VELOCIDADES.keys()))
        velocidade = VELOCIDADES[velocidade_label]
    with ctrl3:
        n_leads_demo = st.number_input(
            "Leads", min_value=10, max_value=2000, value=300, step=10,
        )
    rodando = processo_rodando(estado_vivo)
    estado_sem_modelo = estado_vivo in ESTADOS_SEM_MODELO and not ESTADOS_SEM_MODELO[estado_vivo]

    # --- Réplica a reproduzir (ver ADENDO_PARIDADE_AO_VIVO.md) ---
    # Por padrão, "Rodar ao vivo" reproduz uma réplica ESPECÍFICA do lote --
    # não sorteia livremente. É isso que garante que o painel nunca mostre
    # A e B (ou C e D) com conversões diferentes, contradizendo a identidade
    # exata que o Capítulo 5 documenta.
    rctrl1, rctrl2, rctrl3 = st.columns([1.0, 1.4, 2.6])
    with rctrl1:
        replica_vivo = st.number_input(
            "Réplica", min_value=1, max_value=1000, value=1, step=1,
            help="Reproduz exatamente esta réplica do lote (experimento_30x.py) -- "
                 "mesmos sorteios estocásticos, evento a evento.",
        )
    with rctrl2:
        exploratorio_vivo = st.checkbox(
            "Modo exploratório (sementes livres)",
            help="Desliga a reprodução de réplica oficial. Sorteia livremente a cada "
                 "execução -- útil para \"sentir\" o processo, mas os números NÃO "
                 "correspondem a nenhuma réplica reportada.",
        )
    with rctrl3:
        if exploratorio_vivo:
            st.warning("⚠️ Modo exploratório — resultados não correspondem a nenhuma réplica oficial.", icon="🔀")
        else:
            st.caption(f"Reproduzindo a **réplica {int(replica_vivo)}** do lote — "
                       f"idêntica, evento a evento, à réplica {int(replica_vivo)} de `experimento_30x.py`.")

    with ctrl4:
        iniciar_disabled = rodando or estado_sem_modelo
        if st.button("▶ Iniciar", disabled=iniciar_disabled, type="primary", use_container_width=True):
            st.session_state[f"pos_animada_{estado_vivo}"] = 0
            iniciar_simulacao_ao_vivo(estado_vivo, velocidade, n_leads_demo, int(replica_vivo), exploratorio_vivo)
            st.rerun()
    with ctrl5:
        if st.button("■ Parar", disabled=not rodando, use_container_width=True):
            proc = st.session_state.get(f"proc_{estado_vivo}")
            if proc:
                proc.terminate()
            st.rerun()

    if estado_sem_modelo:
        st.warning(
            f"⚠️ O {NOMES_ESTADOS[estado_vivo].split(' —')[0]} requer modelos de IA treinados. "
            "Vá até a aba **⚙️ Setup do experimento** e complete a Etapa 3 primeiro."
        )
    elif rodando:
        st.success(f"Simulação em andamento — {NOMES_ESTADOS[estado_vivo]}", icon="▶️")
    else:
        erro = processo_terminou_com_erro(estado_vivo)
        if erro:
            with st.expander("⚠️ A simulação parou por causa de um erro — clique para ver detalhes", expanded=True):
                st.code(erro, language=None)
                st.caption(
                    "Causa mais comum: o Ollama não está rodando. Use o script iniciar.sh "
                    "(reinicia o Ollama automaticamente), ou rode 'ollama serve' manualmente."
                )

    st.divider()

    # --- O PROCESSO EM DESTAQUE, largura total, centro da pagina ---
    df_vivo = carregar_eventos(ARQUIVOS[estado_vivo])
    eventos_completos = df_vivo.to_dict("records") if not df_vivo.empty else []
    for ev in eventos_completos:
        if not isinstance(ev.get("detalhes"), dict):
            ev["detalhes"] = {}

    chave_pos = f"pos_animada_{estado_vivo}"
    pos_anterior = st.session_state.get(chave_pos, 0)
    eventos_novos = eventos_completos[pos_anterior:]
    eventos_novos_para_animar = eventos_novos[-25:]
    st.session_state[chave_pos] = len(eventos_completos)

    st.markdown("#### O processo acontecendo")
    st.caption("Bolinhas viajam de uma estação para a outra; verde avançou, vermelho não avançou. "
               "A pilha de bolinhas cinzas antes de cada estação é a fila de leads aguardando.")
    components.html(
        gerar_html_canvas(estado_vivo, eventos_completos, eventos_novos_para_animar),
        height=210,
    )

    # --- DECISÕES DE IA, logo abaixo do processo, ainda em destaque ---
    if estado_vivo in ("B", "C", "D"):
        st.markdown("##### Modelos de IA em ação")
        eventos_ia = df_vivo[df_vivo["etapa"].isin(
            ["qualificacao_ia", "match_closer_ia", "proposta_ia"])] \
            if not df_vivo.empty else df_vivo

        col_ia1, col_ia2, col_ia3 = st.columns(3)

        if estado_vivo in ("C", "D"):
            with col_ia1:
                st.markdown("🟣 **RandomForest** — score de qualificação")
                st.caption(
                    "O modelo analisa orçamento, urgência e tamanho da empresa "
                    "e atribui uma probabilidade de fechamento. Scores acima do "
                    "threshold (≈50%) avançam; os demais são descartados agora."
                )
                ultimos_score = eventos_ia[eventos_ia["etapa"] == "qualificacao_ia"].tail(5) \
                    if not eventos_ia.empty else eventos_ia
                if not ultimos_score.empty:
                    for _, ev in ultimos_score[::-1].iterrows():
                        det = ev["detalhes"] if isinstance(ev["detalhes"], dict) else {}
                        proba = det.get("probabilidade_fechamento")
                        if proba is not None:
                            resultado = "✅ qualificado" if ev["passou"] else "❌ descartado"
                            cor = "green" if ev["passou"] else "orange"
                            barra = int(proba * 20)
                            barra_str = "█" * barra + "░" * (20 - barra)
                            st.markdown(
                                f"**Lead {int(ev['lead_id'])}** `{proba:.1%}` {resultado}  \n"
                                f"<span style='font-size:10px;font-family:monospace;"
                                f"color:{cor}'>{barra_str}</span>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("Aguardando primeiros leads...")

            if estado_vivo == "C":
                with col_ia2:
                    st.markdown("🟣 **kNN** — match de closer por perfil")
                    st.caption(
                        "Compara o lead atual com os 5 deals ganhos mais parecidos "
                        "no histórico do Estado A. Similaridade mais alta = lead tem "
                        "perfil próximo de quem já fechou negócio antes."
                    )
                    ultimos_knn = eventos_ia[eventos_ia["etapa"] == "match_closer_ia"].tail(5) \
                        if not eventos_ia.empty else eventos_ia
                    if not ultimos_knn.empty:
                        for _, ev in ultimos_knn[::-1].iterrows():
                            det = ev["detalhes"] if isinstance(ev["detalhes"], dict) else {}
                            sim = det.get("perfil_similaridade")
                            if sim is not None:
                                nivel = "alto" if sim > 0.003 else "baixo"
                                icone = "🟢" if sim > 0.003 else "🟡"
                                st.markdown(
                                    f"**Lead {int(ev['lead_id'])}** — similaridade `{sim:.4f}` "
                                    f"{icone} perfil {nivel}"
                                )
                    else:
                        st.caption("Aguardando primeiros leads...")
            else:  # D — sem kNN de match, cardinalidade 1 (so' a qualificacao)
                with col_ia2:
                    st.markdown("⚪ **Estado D** — sem match por kNN")
                    st.caption(
                        "Ablação deliberada: o Estado D isola a qualificação por IA "
                        "das demais intervenções do Estado C. Aqui o closer é "
                        "designado do mesmo jeito que nos Estados A e B."
                    )
        elif estado_vivo == "B":
            with col_ia1:
                st.markdown("🔵 **Estado B** — triagem ainda é manual (SDR humano)")
                st.caption(
                    "Neste estado a qualificação não usa modelo de IA — o SDR "
                    "decide com base em heurística e intuição, cometendo erros "
                    "sistemáticos. Compare com o Estado C para ver a diferença."
                )

        if estado_vivo == "D":
            with col_ia3:
                st.markdown("⚪ **Estado D** — proposta ainda é manual")
                st.caption(
                    "Mesma cardinalidade de intervenção do Estado B (uma etapa só), "
                    "mas em local diferente: aqui é a qualificação que recebe IA, "
                    "a proposta permanece manual como no Estado A (~3,5h por lead). "
                    "Compare com a aba 🔀 Comparar B vs D."
                )
        else:
            with col_ia3:
                st.markdown("🟠 **LLM (Llama local)** — proposta gerada")
                st.caption(
                    "A cada lead que chega à etapa de proposta, o Llama 3.1 "
                    "gera o texto de forma autônoma com base no perfil do cliente. "
                    "No Estado A isso era feito manualmente pelo Closer (horas)."
                )
                ultimas_props = eventos_ia[eventos_ia["etapa"] == "proposta_ia"].tail(3) \
                    if not eventos_ia.empty else eventos_ia
                if not ultimas_props.empty:
                    for _, ev in ultimas_props[::-1].iterrows():
                        det = ev["detalhes"] if isinstance(ev["detalhes"], dict) else {}
                        texto = det.get("proposta_texto", "")
                        if texto:
                            chave_widget = f"prop_{estado_vivo}_{int(ev['lead_id'])}_{ev.get('timestamp', '')}"
                            st.text_area(
                                f"Lead {int(ev['lead_id'])}",
                                texto, height=90, key=chave_widget,
                                disabled=True,
                            )
                else:
                    st.caption("Aguardando primeira proposta...")

    st.divider()

    # --- DASHBOARD SECUNDÁRIO: KPIs + funil + eventos brutos, contexto
    # de apoio, visualmente mais discreto que o processo acima ---
    st.markdown("##### Visão geral (dashboard)")
    if not df_vivo.empty:
        m = metricas_resumo(df_vivo)
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Leads que entraram", df_vivo["lead_id"].nunique())
            st.caption("Total de leads únicos que passaram pela geração de leads nesta execução.")
        with mc2:
            st.metric("Negócios fechados", m["fechados"])
            st.caption("Leads que completaram todas as etapas e fecharam negócio. Número parcial enquanto a simulação está em andamento.")
        with mc3:
            st.metric("Conversão (parcial)", f"{m['taxa_conversao']:.1%}")
            st.caption("Fechados ÷ entrados. Deve ser equivalente entre A, B e C — o ganho da IA é em tempo, não em taxa de conversão.")
        with mc4:
            st.metric("Tempo simulado", f"{m['tempo_total_h']:.1f} h")
            st.caption("Soma do esforço humano simulado em horas. Não é o tempo de execução do programa — é o custo operacional acumulado.")

        funil_df = calcular_funil(df_vivo, ETAPAS_POR_ESTADO[estado_vivo])
        st.plotly_chart(
            grafico_funil(funil_df, f"Funil — {NOMES_ESTADOS[estado_vivo]}", NOMES_ETAPAS_LEGIVEIS),
            use_container_width=True,
        )
        st.caption(
            "**Como ler o funil:** cada barra mostra quantos leads *passaram com sucesso* por aquela etapa. "
            "As porcentagens são relativas ao total que entrou no topo. "
            "Quanto maior a queda entre duas etapas consecutivas, maior a taxa de descarte naquele ponto."
        )

        with st.expander("Ver últimos 15 eventos brutos"):
            st.dataframe(df_vivo.tail(15)[["timestamp", "lead_id", "etapa", "passou", "tempo_gasto_min"]],
                         use_container_width=True, hide_index=True)
    else:
        st.caption("Inicie a simulação para ver os números aparecendo aqui.")

    if rodando:
        time.sleep(1)
        st.rerun()

# ============================================================
# ABA 2 — HISTÓRICO DE UMA EXECUÇÃO JÁ CONCLUÍDA
# ============================================================
with aba_historico:
    estado_hist = st.selectbox(
        "Escolha o estado:", options=["A", "B", "C", "D"],
        format_func=lambda x: NOMES_ESTADOS[x], key="sel_estado_hist",
    )
    df_hist = carregar_eventos(ARQUIVOS[estado_hist])

    if df_hist.empty:
        st.info(f"Nenhum dado encontrado. Rode a simulação na aba 'Rodar ao vivo' primeiro.")
    else:
        m = metricas_resumo(df_hist)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Leads processados", m["total_leads"])
            st.caption("Total de leads únicos que passaram por pelo menos uma etapa nesta execução.")
        with c2:
            st.metric("Negócios fechados", m["fechados"])
            st.caption("Leads que completaram todas as etapas do funil e fecharam negócio.")
        with c3:
            st.metric("Taxa de conversão", f"{m['taxa_conversao']:.1%}")
            st.caption("Fechados ÷ total de leads. Deve ser equivalente entre A, B e C — o ganho da IA está no tempo, não na taxa.")
        with c4:
            st.metric("Tempo total investido", f"{m['tempo_total_h']:.0f} h")
            st.caption("Soma do esforço humano simulado em horas. Indicador central de produtividade — compare entre estados.")

        funil_df = calcular_funil(df_hist, ETAPAS_POR_ESTADO[estado_hist])
        st.plotly_chart(
            grafico_funil(funil_df, NOMES_ESTADOS[estado_hist], NOMES_ETAPAS_LEGIVEIS),
            use_container_width=True,
        )
        st.caption(
            "**Como ler o funil:** cada barra representa quantos leads *passaram* por aquela etapa. "
            "As porcentagens ao lado de cada barra são relativas ao total de leads que entraram no topo. "
            "A queda entre etapas consecutivas é a taxa de descarte naquele ponto do processo."
        )

        with st.expander("Ver eventos"):
            st.dataframe(df_hist.tail(50), use_container_width=True)

# ============================================================
# ABA 3 — COMPARAÇÃO A vs B vs C
# ============================================================
with aba_comparacao:
    # Ordem de exibição A -> B -> D -> C (nao alfabetica): e' a ordem
    # argumentativa do capitulo -- B e D lado a lado por serem o contraste
    # de mesma cardinalidade de intervencao, e C fecha como redesenho
    # completo (ver ESPEC_DASHBOARD_4_ESTADOS.md).
    renderizar_comparacao(
        ["A", "B", "D", "C"],
        "Comparação final entre os quatro estados",
        "Esta aba lê os arquivos de histórico gerados pelas execuções mais recentes "
        "de cada estado. Execute cada estado na aba 'Rodar ao vivo' antes de comparar. "
        "Ordem A → B → D → C: B e D primeiro, lado a lado, por terem a mesma cardinalidade "
        "de intervenção (uma etapa cada); C fecha como o redesenho completo.",
        "**Como ler:** A e B devem ter conversão idêntica (a proposta não decide quem fecha "
        "negócio); C e D também devem ser idênticos entre si (nem o kNN nem o LLM afetam essa "
        "probabilidade — só a qualificação decide). Essas duas identidades exatas são o "
        "resultado mais forte do experimento — ver os selos de identidade abaixo.",
        "**Como ler:** a queda de A para B mostra o ganho de automatizar a proposta "
        "(~3,5h → ~1,5min). A queda de A para D mostra o ganho, menor, de automatizar só a "
        "qualificação (~25min → 0,2min por lead) — D economiza menos tempo bruto que B, mas "
        "converte mais. C soma as duas intervenções e chega à maior redução de todas.",
        "Nota metodológica: os quatro estados rodam sobre a MESMA base fixa de leads "
        "(leads_base.csv, seed=42), com sementes pareadas por (repetição, lead, etapa) — "
        "A≡B e C≡D não são aproximações, são identidades exatas por construção.",
    )

    st.divider()
    renderizar_identidades_e_intervencoes(["A", "B", "D", "C"])

# ============================================================
# ABA 3b — COMPARAÇÃO B vs D (mesma cardinalidade, local diferente)
# ============================================================
with aba_comp_bd:
    renderizar_comparacao(
        ["B", "D"],
        "B vs. D — mesma cardinalidade de intervenção, local diferente",
        "B e D têm exatamente UMA etapa substituída por IA cada — a diferença entre eles "
        "é só ONDE. B automatiza a etapa mais cara em tempo (a proposta); D automatiza a "
        "restrição do processo (a qualificação). Execute os dois na aba 'Rodar ao vivo' antes de comparar.",
        "**Como ler:** se D converte mais que B, é evidência de que intervir na restrição do "
        "processo — não na etapa mais visível ou mais cara — é o que move o resultado comercial. "
        "B, isoladamente, não deveria alterar a conversão frente ao baseline.",
        "**Como ler:** B costuma economizar mais tempo bruto que D, porque a etapa que ele "
        "automatiza (proposta, ~3,5h) é muito mais cara que a que D automatiza (qualificação, "
        "~25min) — mesmo D filtrando melhor quem chega à proposta manual.",
        "Nota metodológica: B e D isolam a MESMA quantidade de intervenção (uma etapa) em locais "
        "diferentes do funil — o contraste mais direto possível entre 'automatizar o que é caro' "
        "e 'automatizar a restrição'.",
    )

# ============================================================
# ABA 3c — COMPARAÇÃO C vs D (isola o efeito do kNN + LLM)
# ============================================================
with aba_comp_cd:
    renderizar_comparacao(
        ["C", "D"],
        "C vs. D — isolando o efeito do kNN e do LLM",
        "C e D usam o MESMO RandomForest na qualificação. D remove o kNN de match de closer e "
        "o LLM da proposta — se a conversão não mudar entre eles, o ganho comercial de C é "
        "atribuível especificamente à qualificação, não aos outros componentes.",
        "**Como ler:** conversão igual entre C e D confirma que nem o kNN nem o LLM afetam quem "
        "fecha negócio — só a qualificação decide isso. É o teste de atribuição causal mais "
        "direto do experimento (teste de ablação).",
        "**Como ler:** a diferença de tempo entre C e D é quase inteiramente o custo da proposta "
        "— LLM (~1,5min) em C contra manual (~3,5h) em D. Isola o valor específico desse componente.",
        "Nota metodológica: C e D usam o MESMO modelo_score.pkl, sem retreino — a única diferença "
        "de implementação entre eles é a ausência do kNN e do LLM em D.",
    )

# ============================================================
# ABA 4 — ANÁLISE DOS MODELOS DE IA (RandomForest e kNN)
# ============================================================
with aba_analise:
    renderizar_analise()

# ============================================================
# ABA 5 — RESULTADOS COMPLETOS (B/C)
# ============================================================
with aba_resultados:
    renderizar_resultados_completos()

# ============================================================
# ABA 6 — COMPARAÇÃO ESTATÍSTICA (4 ESTADOS, DESENHO PAREADO)
# ============================================================
with aba_estatistica:
    renderizar_comparacao_estatistica()

