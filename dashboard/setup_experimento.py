"""
setup_experimento.py

Aba de configuracao e laboratório do experimento. Mostra:
  1. Base de dados — amostra, distribuicoes, conceito de fit_real oculto
  2. Estado A em lote — roda rapidamente para gerar material de treino,
     exibe funil resultante e exemplos de leads ganhos vs perdidos
  3. Treinamento dos modelos — RandomForest e kNN com visualizacoes ricas:
     AUC, importancia de features, distribuicao de scores, vizinhos kNN
  4. Limpeza — botoes individuais com confirmacao antes de agir

Esta aba usa arquivos SEPARADOS do modo ao vivo:
  - historico_estado_a_treino.jsonl  (gerado aqui, em lote, para treino)
  - historico_estado_a.jsonl          (gerado na aba ao vivo, visual)
Isso garante que uma demo ao vivo nao sobrescreva o material de treino.
"""

import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PASTA_DADOS = "../dados"
PASTA_MODELOS = "../modelos"
PASTA_SAIDAS = "../saidas"

CAMINHO_LEADS = os.path.join(PASTA_DADOS, "leads_base.csv")
CAMINHO_HIST_TREINO = os.path.join(PASTA_SAIDAS, "historico_estado_a_treino.jsonl")
CAMINHO_HIST_CONSOLID = os.path.join(PASTA_SAIDAS, "historico_estado_a_consolidado.csv")
CAMINHO_MODELO_SCORE = os.path.join(PASTA_MODELOS, "modelo_score.pkl")
CAMINHO_MODELO_MATCH = os.path.join(PASTA_MODELOS, "modelo_match.pkl")
CAMINHO_SCALER_MATCH = os.path.join(PASTA_MODELOS, "scaler_match.pkl")
CAMINHO_THRESHOLD = os.path.join(PASTA_MODELOS, "threshold_score.pkl")

FEATURES = ["orcamento_declarado", "urgencia_declarada", "tamanho_funcionarios"]
FEATURES_PT = {
    "orcamento_declarado": "Orçamento declarado (R$)",
    "urgencia_declarada": "Urgência declarada (0-1)",
    "tamanho_funcionarios": "Tamanho da empresa (funcionários)",
}


# ── helpers ─────────────────────────────────────────────────────────────────

def _arquivo_existe(caminho):
    return os.path.exists(caminho) and os.path.getsize(caminho) > 0


def _mtime_str(caminho):
    import datetime
    if not _arquivo_existe(caminho):
        return None
    ts = os.path.getmtime(caminho)
    return datetime.datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M")


def _badge(ok, texto_ok, texto_nao):
    if ok:
        st.success(f"✅ {texto_ok}")
    else:
        st.warning(f"⏳ {texto_nao}")


@st.cache_data
def _carregar_leads():
    return pd.read_csv(CAMINHO_LEADS)


@st.cache_data(ttl=5)
def _carregar_historico_treino():
    if not _arquivo_existe(CAMINHO_HIST_TREINO):
        return pd.DataFrame()
    linhas = []
    with open(CAMINHO_HIST_TREINO, encoding="utf-8") as f:
        for linha in f:
            try:
                linhas.append(json.loads(linha))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(linhas)


@st.cache_data
def _carregar_consolidado():
    if not _arquivo_existe(CAMINHO_HIST_CONSOLID):
        return None
    return pd.read_csv(CAMINHO_HIST_CONSOLID)


@st.cache_data
def _carregar_modelos():
    if not all(_arquivo_existe(p) for p in [CAMINHO_MODELO_SCORE, CAMINHO_MODELO_MATCH]):
        return None, None, None, None
    with open(CAMINHO_MODELO_SCORE, "rb") as f:
        modelo_rf = pickle.load(f)
    with open(CAMINHO_MODELO_MATCH, "rb") as f:
        modelo_knn = pickle.load(f)
    scaler_knn = None
    if _arquivo_existe(CAMINHO_SCALER_MATCH):
        with open(CAMINHO_SCALER_MATCH, "rb") as f:
            scaler_knn = pickle.load(f)
    threshold = None
    if _arquivo_existe(CAMINHO_THRESHOLD):
        with open(CAMINHO_THRESHOLD, "rb") as f:
            threshold = pickle.load(f)
    return modelo_rf, modelo_knn, scaler_knn, threshold


def _rodar_script_em_lote(nome_script: str, placeholder):
    """Lanca um script Python em lote e exibe progresso em tempo real."""
    caminho = os.path.join(PASTA_DADOS, nome_script)
    proc = subprocess.Popen(
        [sys.executable, caminho],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=PASTA_DADOS,
    )
    linhas = []
    log_box = placeholder.empty()
    for linha in proc.stdout:
        linhas.append(linha.rstrip())
        log_box.code("\n".join(linhas[-20:]))
    proc.wait()
    return proc.returncode == 0


# ── renderizacao principal ───────────────────────────────────────────────────

def renderizar_setup():
    st.markdown("## Setup do experimento")
    st.caption(
        "Configure o experimento passo a passo. Cada etapa habilita a próxima. "
        "Esta aba é o laboratório — o modo ao vivo fica na aba 'Rodar ao vivo'."
    )

    with st.expander("📖 Os quatro estados do experimento", expanded=False):
        st.markdown(
            "| Estado | Qualificação | Match de closer | Proposta | Intervenções de IA |\n"
            "|---|---|---|---|---|\n"
            "| **A** — Baseline | manual (SDR) | manual | manual (~3,5h) | 0 |\n"
            "| **B** — Melhoria pontual | manual (SDR) | manual | **LLM** (~1,5min) | 1 |\n"
            "| **D** — Ablação | **RandomForest** | manual | manual (~3,5h) | 1 |\n"
            "| **C** — Orquestração sistêmica | **RandomForest** | **kNN** | **LLM** | 3 |\n"
        )
        st.caption(
            "**B e D têm a mesma cardinalidade de intervenção — uma única etapa cada — e diferem "
            "só na localização.** B automatiza a etapa mais cara em tempo (a proposta, ~3,5h por "
            "lead); D automatiza a restrição do processo (a qualificação, ~25min por lead). Essa "
            "é a comparação de par mais direta do experimento: mesma \"quantidade\" de IA, "
            "resultado diferente conforme onde ela entra (ver aba 🔀 Comparar B vs D). "
            "C soma as três intervenções — é o redesenho completo, não apenas mais automação."
        )

    tem_leads = _arquivo_existe(CAMINHO_LEADS)
    tem_historico_treino = _arquivo_existe(CAMINHO_HIST_TREINO)
    tem_consolidado = _arquivo_existe(CAMINHO_HIST_CONSOLID)
    tem_modelos = all(_arquivo_existe(p) for p in [CAMINHO_MODELO_SCORE, CAMINHO_MODELO_MATCH])

    # barra de progresso geral do experimento
    etapas_ok = sum([tem_leads, tem_historico_treino and tem_consolidado, tem_modelos])
    st.progress(etapas_ok / 3, text=f"Experimento configurado: {etapas_ok}/3 etapas concluídas")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 1 — BASE DE DADOS
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 1️⃣ Base de dados — os leads do experimento")
    st.caption(
        "2.000 leads sintéticos gerados com semente fixa (seed=42). "
        "Qualquer pessoa que gere esta base com os mesmos parâmetros obterá "
        "exatamente os mesmos dados — essa é a garantia de replicabilidade do experimento."
    )

    col_btn1, col_status1 = st.columns([1, 3])
    with col_btn1:
        if st.button(
            "🔄 Gerar base de leads" if not tem_leads else "♻️ Regenerar base",
            type="primary" if not tem_leads else "secondary",
        ):
            ph = st.empty()
            with st.spinner("Gerando base de leads..."):
                ok = _rodar_script_em_lote("gerar_base.py", ph)
            if ok:
                st.cache_data.clear()
                st.success("Base gerada com sucesso!")
                st.rerun()
            else:
                st.error("Erro ao gerar a base. Veja o log acima.")
    with col_status1:
        if tem_leads:
            st.success(f"✅ leads_base.csv — gerado em {_mtime_str(CAMINHO_LEADS)}")
        else:
            st.warning("⏳ Base de leads ainda não gerada.")

    if tem_leads:
        leads = _carregar_leads()

        # ── amostra dos dados ──
        st.markdown("#### Amostra dos dados (primeiros 10 leads)")
        st.caption(
            "Cada linha é um lead com suas características **visíveis** "
            "(o que o SDR ou a IA conseguem observar) e o **fit_real** oculto "
            "(a 'verdade' que ninguém vê diretamente, nem humano nem modelo)."
        )
        cols_mostrar = ["lead_id", "empresa", "setor", "canal_origem",
                        "orcamento_declarado", "urgencia_declarada",
                        "tamanho_funcionarios", "fit_real"]
        st.dataframe(
            leads[cols_mostrar].head(10).style.format({
                "orcamento_declarado": "R$ {:,.0f}",
                "urgencia_declarada": "{:.3f}",
                "fit_real": "{:.3f}",
            }),
            use_container_width=True, hide_index=True,
        )

        # ── explicacao do fit_real ──
        with st.expander("📖 O que é o fit_real e por que ele é oculto?", expanded=False):
            st.markdown("""
**fit_real** representa o ajuste genuíno entre o lead e o produto: uma combinação
de orçamento real disponível, necessidade legítima e momento de compra correto.
Na prática, esse valor **não existe como um campo em nenhum CRM** — ele só se
revela quando o negócio fecha ou não fecha.

Neste experimento, criamos o fit_real de forma semi-determinística: ele é
parcialmente derivado das features visíveis (orçamento + urgência + tamanho)
com ruído aleatório. Isso é necessário para que o problema seja genuinamente
solucionável por um modelo de ML — se fit_real fosse completamente independente
das features, nenhuma IA conseguiria prever quem vai fechar, o que tornaria
a comparação entre estados sem sentido.

**Na simulação:** o SDR humano (Estado A) **não vê** o fit_real e decide com
base em heurística imperfeita. O RandomForest (Estado C) também **não vê**
o fit_real diretamente — ele aprende a *estimá-lo* a partir das features
visíveis, usando os padrões do histórico de fechamentos do Estado A.
            """)

        # ── distribuicoes das variaveis ──
        st.markdown("#### Distribuição das variáveis")
        dcol1, dcol2, dcol3 = st.columns(3)

        with dcol1:
            fig = px.histogram(leads, x="orcamento_declarado", nbins=30,
                               title="Orçamento declarado (R$)",
                               color_discrete_sequence=["#378ADD"])
            fig.update_layout(height=240, showlegend=False,
                              margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Distribuição do orçamento que cada lead declarou. Variável visível para SDR e IA.")

        with dcol2:
            fig = px.histogram(leads, x="urgencia_declarada", nbins=20,
                               title="Urgência declarada (0–1)",
                               color_discrete_sequence=["#7F77DD"])
            fig.update_layout(height=240, showlegend=False,
                              margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Urgência autodeclarada pelo lead (0 = sem pressa, 1 = urgente). Variável visível.")

        with dcol3:
            fig = px.histogram(leads, x="fit_real", nbins=25,
                               title="fit_real oculto (verdade)",
                               color_discrete_sequence=["#1D9E75"])
            fig.update_layout(height=240, showlegend=False,
                              margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "**Variável oculta** — não existe no CRM real. "
                "Mostrada aqui só para fins educativos. "
                "Na empresa real, ninguém veria esta distribuição."
            )

        dcol4, dcol5 = st.columns(2)
        with dcol4:
            fig = px.histogram(leads, x="setor", title="Distribuição por setor",
                               color_discrete_sequence=["#EF9F27"])
            fig.update_layout(height=240, showlegend=False,
                              margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Setor de atuação de cada lead. Variável categórica não usada diretamente pelos modelos.")

        with dcol5:
            fig = px.histogram(leads, x="canal_origem", title="Canal de origem",
                               color_discrete_sequence=["#D85A30"])
            fig.update_layout(height=240, showlegend=False,
                              margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("De onde vieram os leads. Variável informativa — não entra como feature nos modelos.")

        # correlacao entre variaveis visiveis e fit_real
        st.markdown("#### Correlação entre variáveis visíveis e fit_real")
        st.caption(
            "Quanto maior a correlação, mais essa variável 'carrega sinal' "
            "sobre o fit real — ou seja, mais útil ela é para o modelo aprender."
        )
        corrs = {
            FEATURES_PT[f]: round(leads[f].corr(leads["fit_real"]), 3)
            for f in FEATURES
        }
        fig_corr = go.Figure(go.Bar(
            x=list(corrs.values()), y=list(corrs.keys()),
            orientation="h", marker_color="#378ADD",
            text=[f"{v:+.3f}" for v in corrs.values()],
            textposition="outside",
        ))
        fig_corr.update_layout(
            title="Correlação de Pearson com fit_real",
            xaxis=dict(range=[-0.1, max(corrs.values()) * 1.4]),
            height=220, margin=dict(t=40, b=10, l=10, r=60),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(
            "**Como ler:** barras mais longas indicam que a variável carrega mais sinal sobre o fit_real. "
            "O modelo usa essas três features para aprender a separar leads com alto e baixo fit real, "
            "mesmo sem nunca observar o fit_real diretamente."
        )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 2 — ESTADO A EM LOTE (para treino)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 2️⃣ Estado A — processo baseline (em lote, para treino)")
    st.caption(
        "Roda os 2.000 leads pelo processo manual completo (sem IA). "
        "O resultado gera o histórico que será usado para treinar os modelos. "
        "Este modo em lote é rápido — para a demonstração visual ao vivo, "
        "use a aba 'Rodar ao vivo'."
    )

    col_btn2, col_status2 = st.columns([1, 3])
    with col_btn2:
        btn_a_disabled = not tem_leads
        if st.button(
            "▶ Rodar Estado A",
            disabled=btn_a_disabled,
            type="primary" if not tem_historico_treino else "secondary",
        ):
            ph = st.empty()
            with st.spinner("Rodando Estado A em lote..."):
                ok = _rodar_script_em_lote("rodar_estado_a.py", ph)
            if ok:
                st.cache_data.clear()
                st.success("Estado A concluído!")
                st.rerun()
            else:
                st.error("Erro ao rodar o Estado A. Veja o log acima.")
    with col_status2:
        if tem_historico_treino and tem_consolidado:
            st.success(f"✅ Histórico de treino gerado em {_mtime_str(CAMINHO_HIST_TREINO)}")
        elif not tem_leads:
            st.info("Gere a base de leads primeiro (Etapa 1).")
        else:
            st.warning("⏳ Estado A ainda não rodado.")

    if tem_historico_treino and tem_consolidado:
        hist_df = _carregar_historico_treino()
        consolid = _carregar_consolidado()

        if not hist_df.empty and consolid is not None:
            fechados = consolid["fechou_negocio"].sum()
            total = len(consolid)
            taxa = fechados / total

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Leads processados", total)
                st.caption("Total de leads únicos que passaram pelo processo do Estado A.")
            with mcol2:
                st.metric("Negócios fechados", fechados)
                st.caption("Leads que completaram todas as 7 etapas — serão os exemplos positivos de treino.")
            with mcol3:
                st.metric("Taxa de conversão", f"{taxa:.1%}")
                st.caption("Fechados ÷ total. Taxa típica de funis B2B (~7%). Reflete a dificuldade real do problema.")
            with mcol4:
                st.metric("Este histórico vira", "material de treino")
                st.caption("O CSV consolidado será usado para treinar o RandomForest e o kNN na Etapa 3.")

            # funil do Estado A
            st.markdown("#### Funil do Estado A (material de treino)")
            etapas_a = ["geracao_leads", "qualificacao_sdr", "descoberta_closer",
                        "proposta_manual", "negociacao", "fechamento", "pos_venda"]
            nomes_a = ["Geração de leads", "Qualificação SDR", "Descoberta",
                       "Proposta manual", "Negociação", "Fechamento", "Pós-venda"]
            funil_data = []
            for etapa, nome in zip(etapas_a, nomes_a):
                df_et = hist_df[hist_df["etapa"] == etapa]
                passou = int(df_et["passou"].sum()) if not df_et.empty else 0
                funil_data.append({"etapa": nome, "passaram": passou})
            df_funil = pd.DataFrame(funil_data)
            fig_funil = go.Figure(go.Funnel(
                y=df_funil["etapa"], x=df_funil["passaram"],
                textinfo="value+percent initial",
                marker=dict(color="#378ADD"),
            ))
            fig_funil.update_layout(height=360, margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig_funil, use_container_width=True)
            st.caption(
                "**Como ler:** cada barra mostra quantos leads *passaram com sucesso* por aquela etapa. "
                "A queda entre etapas é a taxa de descarte — onde mais leads saem do funil. "
                "Este é o funil de referência que os Estados B e C vão tentar melhorar em eficiência."
            )

            # exemplos de leads ganhos vs perdidos
            st.markdown("#### Exemplos: leads que fecharam vs que não fecharam")
            st.caption(
                "Esses são os casos reais que o RandomForest vai usar para aprender "
                "a diferença entre um lead bom e um lead ruim."
            )
            col_ganhos, col_perdidos = st.columns(2)
            with col_ganhos:
                st.markdown("✅ **Fecharam negócio**")
                ganhos = consolid[consolid["fechou_negocio"] == 1][
                    ["empresa", "setor", "orcamento_declarado", "urgencia_declarada",
                     "tamanho_funcionarios", "fit_real"]
                ].head(8)
                st.dataframe(ganhos.style.format({
                    "orcamento_declarado": "R$ {:,.0f}",
                    "urgencia_declarada": "{:.3f}",
                    "fit_real": "{:.3f}",
                }), use_container_width=True, hide_index=True)

            with col_perdidos:
                st.markdown("❌ **Não fecharam**")
                perdidos = consolid[consolid["fechou_negocio"] == 0][
                    ["empresa", "setor", "orcamento_declarado", "urgencia_declarada",
                     "tamanho_funcionarios", "fit_real"]
                ].head(8)
                st.dataframe(perdidos.style.format({
                    "orcamento_declarado": "R$ {:,.0f}",
                    "urgencia_declarada": "{:.3f}",
                    "fit_real": "{:.3f}",
                }), use_container_width=True, hide_index=True)

            # scatter: onde estao os ganhos no espaco de features
            st.markdown("#### Onde vivem os leads que fecharam (espaço de features)")
            st.caption(
                "Se os pontos verdes (fecharam) e cinzas (não fecharam) estivessem "
                "completamente misturados, nenhum modelo conseguiria aprender a diferença. "
                "O fato de haver alguma separação é o que torna o problema solucionável por ML."
            )
            amostra = consolid.sample(min(600, len(consolid)), random_state=42)
            fig_scatter = go.Figure()
            nao_fechou_s = amostra[amostra["fechou_negocio"] == 0]
            fechou_s = amostra[amostra["fechou_negocio"] == 1]
            fig_scatter.add_trace(go.Scatter(
                x=nao_fechou_s["orcamento_declarado"],
                y=nao_fechou_s["urgencia_declarada"],
                mode="markers", name="Não fechou",
                marker=dict(color="#9c9a92", size=5, opacity=0.5),
                hovertemplate="Empresa: %{text}<br>Orçamento: R$ %{x:,.0f}<br>Urgência: %{y:.3f}",
                text=nao_fechou_s["empresa"],
            ))
            fig_scatter.add_trace(go.Scatter(
                x=fechou_s["orcamento_declarado"],
                y=fechou_s["urgencia_declarada"],
                mode="markers", name="Fechou negócio",
                marker=dict(color="#1D9E75", size=9, symbol="diamond"),
                hovertemplate="Empresa: %{text}<br>Orçamento: R$ %{x:,.0f}<br>Urgência: %{y:.3f}",
                text=fechou_s["empresa"],
            ))
            fig_scatter.update_layout(
                title="Orçamento × Urgência — leads do Estado A",
                xaxis_title="Orçamento declarado (R$)",
                yaxis_title="Urgência declarada",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 3 — TREINAMENTO DOS MODELOS
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 3️⃣ Treinamento dos modelos de IA")
    st.caption(
        "Treina o RandomForest e o kNN usando exclusivamente o histórico "
        "do Estado A. Os modelos aprendem quais padrões de lead estão associados "
        "a fechamento de negócio — sem nunca ver os dados dos Estados B ou C "
        "(isso evitaria contaminação, o chamado data leakage)."
    )

    col_btn3, col_status3 = st.columns([1, 3])
    with col_btn3:
        btn_treino_disabled = not (tem_historico_treino and tem_consolidado)
        if st.button(
            "🧠 Treinar modelos",
            disabled=btn_treino_disabled,
            type="primary" if not tem_modelos else "secondary",
        ):
            ph = st.empty()
            with st.spinner("Treinando RandomForest e kNN..."):
                ok = _rodar_script_em_lote("treinar_modelos.py", ph)
            if ok:
                st.cache_data.clear()
                st.success("Modelos treinados e salvos!")
                st.rerun()
            else:
                st.error("Erro no treinamento. Veja o log acima.")
    with col_status3:
        if tem_modelos:
            st.success(f"✅ Modelos treinados em {_mtime_str(CAMINHO_MODELO_SCORE)}")
        elif not (tem_historico_treino and tem_consolidado):
            st.info("Rode o Estado A primeiro (Etapa 2).")
        else:
            st.warning("⏳ Modelos ainda não treinados.")

    if tem_modelos:
        modelo_rf, modelo_knn, scaler_knn, threshold = _carregar_modelos()
        consolid = _carregar_consolidado()
        leads = _carregar_leads()

        if modelo_rf is not None and consolid is not None:
            X = leads[FEATURES]
            probas = modelo_rf.predict_proba(X)[:, 1]
            merged_m = consolid.merge(
                pd.DataFrame({"lead_id": leads["lead_id"], "proba_rf": probas}),
                on="lead_id",
            )

            # ── metricas de qualidade ──
            st.markdown("#### Qualidade do RandomForest")
            try:
                from sklearn.metrics import (
                    roc_auc_score, confusion_matrix, classification_report,
                )
                auc = roc_auc_score(merged_m["fechou_negocio"], merged_m["proba_rf"])
                y_pred = (merged_m["proba_rf"] >= threshold).astype(int)
                cm = confusion_matrix(merged_m["fechou_negocio"], y_pred)
                corr_fit = float(np.corrcoef(probas, leads["fit_real"])[0, 1])

                qcol1, qcol2, qcol3, qcol4 = st.columns(4)
                with qcol1:
                    st.metric("AUC-ROC", f"{auc:.3f}")
                    st.caption("Capacidade do modelo de separar leads bons dos ruins. 0,5 = acaso, 1,0 = perfeito.")
                with qcol2:
                    st.metric("Correlação com fit_real", f"{corr_fit:.3f}")
                    st.caption("O quanto o score captura a variável oculta. Confirma que o modelo aprendeu sinal real.")
                with qcol3:
                    st.metric("Threshold de corte", f"{threshold:.1%}")
                    st.caption("Ponto de corte calibrado. Leads com score acima deste valor avançam no Estado C.")
                with qcol4:
                    st.metric("Treinado com", f"{int(consolid['fechou_negocio'].sum())} deals ganhos")
                    st.caption("Exemplos positivos de treino. O RandomForest usa class_weight='balanced' para compensar o desbalanceamento.")

            except Exception as e:
                st.warning(f"Não foi possível calcular métricas: {e}")
                auc, cm = None, None

            # ── distribuicao de scores ──
            st.markdown("#### Distribuição dos scores previstos pelo RandomForest")
            st.caption(
                "A separação entre as duas curvas mostra que o modelo aprendeu sinal real — "
                "leads que efetivamente fecharam negócio recebem scores mais altos."
            )
            fechou_p = merged_m[merged_m["fechou_negocio"] == 1]["proba_rf"]
            nao_fechou_p = merged_m[merged_m["fechou_negocio"] == 0]["proba_rf"]
            fig_scores = go.Figure()
            fig_scores.add_trace(go.Histogram(
                x=nao_fechou_p, name="Não fechou", opacity=0.65,
                marker_color="#9c9a92", nbinsx=30,
            ))
            fig_scores.add_trace(go.Histogram(
                x=fechou_p, name="Fechou negócio", opacity=0.85,
                marker_color="#1D9E75", nbinsx=30,
            ))
            if threshold:
                fig_scores.add_vline(
                    x=threshold, line_dash="dash", line_color="#E24B4A",
                    annotation_text=f"threshold ({threshold:.1%})",
                    annotation_position="top right",
                )
            fig_scores.update_layout(
                barmode="overlay", height=320,
                xaxis_title="Score previsto pelo RandomForest",
                yaxis_title="Número de leads",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_scores, use_container_width=True)
            st.caption(
                f"A linha vermelha tracejada marca o threshold de {threshold:.1%}: "
                "leads à direita dela são qualificados pelo Estado C, "
                "os da esquerda são descartados automaticamente pela IA."
            )

            # ── importancia das features ──
            st.markdown("#### Importância das variáveis para o RandomForest")
            importancias = modelo_rf.feature_importances_
            labels = [FEATURES_PT[f] for f in FEATURES]
            ordem = np.argsort(importancias)
            fig_imp = go.Figure(go.Bar(
                x=importancias[ordem], y=[labels[i] for i in ordem],
                orientation="h", marker_color="#7F77DD",
                text=[f"{v:.1%}" for v in importancias[ordem]],
                textposition="outside",
            ))
            fig_imp.update_layout(
                height=240,
                xaxis=dict(range=[0, max(importancias) * 1.35]),
                margin=dict(t=10, b=10, l=10, r=60),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption(
                "As três variáveis contribuem de forma relativamente equilibrada, "
                "o que é bom — significa que o modelo não está dependendo de "
                "um único fator para decidir, reduzindo risco de viés."
            )

            # ── matriz de confusao ──
            if cm is not None:
                st.markdown("#### Matriz de confusão")
                st.caption(
                    "Mostra onde o modelo acerta e onde erra. "
                    "Verdadeiro Positivo (VP) = qualificou um lead que realmente fechou. "
                    "Falso Positivo (FP) = qualificou um lead que não fecharia. "
                    "Falso Negativo (FN) = descartou um lead que teria fechado."
                )
                labels_cm = ["Não fechou (0)", "Fechou (1)"]
                fig_cm = go.Figure(go.Heatmap(
                    z=cm, x=labels_cm, y=labels_cm,
                    colorscale="Blues",
                    text=cm, texttemplate="%{text}",
                    textfont=dict(size=18),
                ))
                fig_cm.update_layout(
                    xaxis_title="Previsto pelo modelo",
                    yaxis_title="Real (aconteceu de fato)",
                    height=280, width=380,
                )
                cm_col, cm_txt = st.columns([1, 1.4])
                with cm_col:
                    st.plotly_chart(fig_cm, use_container_width=True)
                with cm_txt:
                    vp = cm[1][1]
                    fp = cm[0][1]
                    fn = cm[1][0]
                    vn = cm[0][0]
                    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0
                    recall = vp / (vp + fn) if (vp + fn) > 0 else 0
                    st.markdown(f"""
**Interpretação:**
- **{vp} VP** — leads que o modelo qualificou e que realmente fecharam ✅
- **{fp} FP** — leads qualificados que não fechariam (trabalho desperdiçado) ⚠️
- **{fn} FN** — leads descartados que teriam fechado (oportunidade perdida) ⚠️
- **{vn} VN** — descartados corretamente ✅

**Precisão:** {precisao:.1%} dos leads qualificados eram de fato bons  
**Recall:** {recall:.1%} dos leads bons foram identificados pelo modelo
                    """)

            st.divider()

            # ── kNN ──
            st.markdown("#### kNN — base de referência para match de closer")
            n_deals_ganhos = modelo_knn.n_samples_fit_
            st.caption(
                f"O kNN foi treinado com **{n_deals_ganhos} deals ganhos** "
                "extraídos do histórico do Estado A. "
                "Quando um novo lead chega no Estado C, o modelo compara "
                "o perfil dele com esses {n_deals_ganhos} casos de sucesso "
                "e calcula qual é o mais parecido."
            )

            # distancias para todos os leads (mesma padronizacao do treino)
            X_leads = leads[FEATURES]
            X_leads_escalado = scaler_knn.transform(X_leads) if scaler_knn is not None else X_leads
            distancias, indices = modelo_knn.kneighbors(X_leads_escalado)
            dist_media = distancias.mean(axis=1)
            merged_knn = consolid.merge(
                pd.DataFrame({"lead_id": leads["lead_id"], "dist_media": dist_media}),
                on="lead_id",
            )
            fechou_d = merged_knn[merged_knn["fechou_negocio"] == 1]["dist_media"]
            nao_fechou_d = merged_knn[merged_knn["fechou_negocio"] == 0]["dist_media"]
            razao = nao_fechou_d.mean() / fechou_d.mean() if fechou_d.mean() > 0 else 0

            kcol1, kcol2 = st.columns(2)
            with kcol1:
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=nao_fechou_d, name="Não fechou", marker_color="#9c9a92",
                ))
                fig_box.add_trace(go.Box(
                    y=fechou_d, name="Fechou negócio", marker_color="#1D9E75",
                ))
                fig_box.update_layout(
                    title=f"Distância aos 5 vizinhos mais próximos",
                    yaxis_title="Distância euclidiana média",
                    height=320,
                )
                st.plotly_chart(fig_box, use_container_width=True)
                st.caption(
                    f"Leads que fecharam estão {razao:.1f}× mais próximos dos "
                    "deals ganhos do que os que não fecharam — evidência de que "
                    "a 'vizinhança' captura sinal real."
                )

            with kcol2:
                # exemplo concreto: pega o lead mais proximo de um deal ganho
                idx_mais_proximo = np.argmin(dist_media)
                lead_exemplo = leads.iloc[idx_mais_proximo]
                vizinhos_ids = indices[idx_mais_proximo]
                ganhos_ids = set(consolid[consolid["fechou_negocio"] == 1]["lead_id"])
                vizinhos_df = leads[leads.index.isin(vizinhos_ids)][
                    ["empresa", "orcamento_declarado", "urgencia_declarada"]
                ].copy()
                # isin retorna array numpy, nao pd.Series -- converter antes de .map()
                isin_result = pd.Series(
                    leads.index.isin(vizinhos_ids),
                    index=leads.index
                )
                vizinhos_df["é deal ganho?"] = isin_result[vizinhos_df.index].map(
                    {True: "✅", False: "❌"}
                )

                st.markdown("**Exemplo concreto de como o kNN funciona:**")
                st.markdown(
                    f"Lead **{lead_exemplo['empresa']}** "
                    f"(orçamento R$ {lead_exemplo['orcamento_declarado']:,.0f}, "
                    f"urgência {lead_exemplo['urgencia_declarada']:.3f}) "
                    f"tem distância de **{dist_media[idx_mais_proximo]:,.0f}** "
                    "ao seu vizinho mais próximo nos deals ganhos."
                )
                st.markdown("Os 5 vizinhos mais próximos (deals ganhos de referência):")
                ganhos_leads = leads[leads["lead_id"].isin(ganhos_ids)]
                vizinhos_ganhos = ganhos_leads.iloc[vizinhos_ids][
                    ["empresa", "setor", "orcamento_declarado", "urgencia_declarada"]
                ]
                st.dataframe(
                    vizinhos_ganhos.style.format({
                        "orcamento_declarado": "R$ {:,.0f}",
                        "urgencia_declarada": "{:.3f}",
                    }),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "O Closer alocado para o novo lead será aquele que "
                    "melhor performa com este perfil de cliente, com base "
                    "no padrão observado nestes vizinhos."
                )

            # curva ROC
            st.markdown("#### Curva ROC — capacidade discriminativa do RandomForest")
            try:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(merged_m["fechou_negocio"], merged_m["proba_rf"])
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"RandomForest (AUC={auc:.3f})",
                    line=dict(color="#1D9E75", width=2),
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Acaso (AUC=0.5)",
                    line=dict(color="#9c9a92", dash="dash"),
                ))
                fig_roc.update_layout(
                    title="Curva ROC",
                    xaxis_title="Taxa de Falso Positivo",
                    yaxis_title="Taxa de Verdadeiro Positivo",
                    height=320,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                st.caption(
                    "Quanto mais a curva verde se afasta da linha cinza (acaso), "
                    "melhor o modelo discrimina leads bons de leads ruins. "
                    f"AUC de {auc:.3f} indica capacidade discriminativa "
                    f"{'boa' if auc >= 0.7 else 'moderada'}."
                )
            except Exception:
                pass

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SEÇÃO DE LIMPEZA — sempre no final, visualmente destacada
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 🗑️ Limpeza — resetar o experimento")
    st.caption(
        "Use para começar um novo experimento do zero. "
        "Cada ação é irreversível — confirme antes de executar."
    )

    with st.expander("⚠️ Opções de limpeza (clique para expandir)", expanded=False):
        lcol1, lcol2, lcol3 = st.columns(3)

        with lcol1:
            st.markdown("**Limpar históricos de execução**")
            st.caption("Remove os jsonl de todos os estados e o CSV consolidado.")
            if st.button("🗑️ Limpar históricos", use_container_width=True):
                st.session_state["confirmar_hist"] = True

            if st.session_state.get("confirmar_hist"):
                st.warning("Tem certeza? Esta ação é irreversível.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Sim, limpar", use_container_width=True):
                        for f in [
                            CAMINHO_HIST_TREINO, CAMINHO_HIST_CONSOLID,
                            os.path.join(PASTA_SAIDAS, "historico_estado_a.jsonl"),
                            os.path.join(PASTA_SAIDAS, "historico_estado_b.jsonl"),
                            os.path.join(PASTA_SAIDAS, "historico_estado_c.jsonl"),
                            os.path.join(PASTA_SAIDAS, "cache_llm.json"),
                        ]:
                            if os.path.exists(f):
                                os.remove(f)
                        st.session_state.pop("confirmar_hist", None)
                        st.cache_data.clear()
                        st.success("Históricos removidos.")
                        st.rerun()
                with c2:
                    if st.button("❌ Cancelar", use_container_width=True):
                        st.session_state.pop("confirmar_hist", None)
                        st.rerun()

        with lcol2:
            st.markdown("**Limpar modelos treinados**")
            st.caption("Remove os .pkl do RandomForest, kNN e threshold.")
            if st.button("🗑️ Limpar modelos", use_container_width=True):
                st.session_state["confirmar_modelos"] = True

            if st.session_state.get("confirmar_modelos"):
                st.warning("Tem certeza? O Estado C não funcionará sem modelos.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Sim, limpar ", use_container_width=True):
                        for f in [CAMINHO_MODELO_SCORE, CAMINHO_MODELO_MATCH,
                                  CAMINHO_SCALER_MATCH, CAMINHO_THRESHOLD]:
                            if os.path.exists(f):
                                os.remove(f)
                        st.session_state.pop("confirmar_modelos", None)
                        st.cache_data.clear()
                        st.success("Modelos removidos.")
                        st.rerun()
                with c2:
                    if st.button("❌ Cancelar ", use_container_width=True):
                        st.session_state.pop("confirmar_modelos", None)
                        st.rerun()

        with lcol3:
            st.markdown("**Resetar tudo**")
            st.caption("Remove históricos, modelos e a base de leads.")
            if st.button("💥 Resetar tudo", type="primary", use_container_width=True):
                st.session_state["confirmar_tudo"] = True

            if st.session_state.get("confirmar_tudo"):
                st.error("⚠️ Isso apaga TUDO, inclusive a base de leads!")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Sim, resetar TUDO", use_container_width=True):
                        for f in [
                            CAMINHO_LEADS, CAMINHO_HIST_TREINO, CAMINHO_HIST_CONSOLID,
                            CAMINHO_MODELO_SCORE, CAMINHO_MODELO_MATCH,
                            CAMINHO_SCALER_MATCH, CAMINHO_THRESHOLD,
                            os.path.join(PASTA_SAIDAS, "historico_estado_a.jsonl"),
                            os.path.join(PASTA_SAIDAS, "historico_estado_b.jsonl"),
                            os.path.join(PASTA_SAIDAS, "historico_estado_c.jsonl"),
                            os.path.join(PASTA_SAIDAS, "cache_llm.json"),
                        ]:
                            if os.path.exists(f):
                                os.remove(f)
                        st.session_state.pop("confirmar_tudo", None)
                        st.cache_data.clear()
                        st.success("Tudo resetado. Comece pela Etapa 1.")
                        st.rerun()
                with c2:
                    if st.button("❌ Cancelar  ", use_container_width=True):
                        st.session_state.pop("confirmar_tudo", None)
                        st.rerun()
