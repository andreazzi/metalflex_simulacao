"""
resultados_completos.py

Aba de resultados completos dos modelos de IA para os Estados B, C e D.
Mostra todos os outputs produzidos numa execucao -- nao apenas amostras:
  - Estado B: todas as propostas geradas pelo LLM
  - Estado C: todas as decisoes do RandomForest + todos os matches do kNN
              + todas as propostas geradas pelo LLM
  - Estado D: todas as decisoes do RandomForest (mesmo modelo do Estado C) --
              sem kNN, sem LLM (ablacao: so' a qualificacao recebe IA)
"""

import json
import os

import pandas as pd
import streamlit as st

PASTA_DADOS = "../dados"
PASTA_SAIDAS = "../saidas"

ARQUIVOS = {
    "B": os.path.join(PASTA_SAIDAS, "historico_estado_b.jsonl"),
    "C": os.path.join(PASTA_SAIDAS, "historico_estado_c.jsonl"),
    "D": os.path.join(PASTA_SAIDAS, "historico_estado_d.jsonl"),
}


@st.cache_data(ttl=5)
def _carregar_eventos(caminho: str) -> pd.DataFrame:
    if not os.path.exists(caminho):
        return pd.DataFrame()
    linhas = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            s = linha.strip()
            if s:
                try:
                    linhas.append(json.loads(s))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(linhas)
    if not df.empty and "detalhes" in df.columns:
        df["detalhes"] = df["detalhes"].apply(
            lambda d: d if isinstance(d, dict) else {}
        )
    return df


@st.cache_data
def _carregar_leads() -> pd.DataFrame:
    caminho = os.path.join(PASTA_DADOS, "leads_base.csv")
    if os.path.exists(caminho):
        return pd.read_csv(caminho)
    return pd.DataFrame()


def _enriquecer(df_etapa: pd.DataFrame, leads: pd.DataFrame) -> pd.DataFrame:
    if not leads.empty and "lead_id" in df_etapa.columns:
        return df_etapa.merge(
            leads[["lead_id", "empresa", "setor",
                   "orcamento_declarado", "urgencia_declarada", "tamanho_funcionarios"]],
            on="lead_id", how="left",
        )
    return df_etapa


def _secao_propostas(propostas: pd.DataFrame, chave_prefix: str):
    """Filtros + tabela resumo + expanders com texto completo das propostas."""
    if propostas.empty:
        st.info("Nenhuma proposta encontrada no histórico.")
        return

    # ── filtros ──
    fc1, fc2, fc3 = st.columns([1.5, 1.5, 1])
    with fc1:
        setores = ["Todos"]
        if "setor" in propostas.columns:
            setores += sorted(propostas["setor"].dropna().unique().tolist())
        setor_sel = st.selectbox("Filtrar por setor", setores, key=f"{chave_prefix}_setor")
    with fc2:
        busca = st.text_input("Buscar empresa", "", key=f"{chave_prefix}_busca",
                              placeholder="Digite parte do nome…")
    with fc3:
        n = len(propostas)
        if n == 0:
            st.info("Nenhuma proposta encontrada com os filtros atuais.")
            return
        limite = st.number_input(
            "Mostrar até", min_value=1, max_value=max(1, n),
            value=max(1, min(50, n)), step=10, key=f"{chave_prefix}_limite",
            help="Número máximo de expanders exibidos abaixo. A tabela sempre mostra todos.",
        )

    filtrado = propostas.copy()
    if setor_sel != "Todos" and "setor" in filtrado.columns:
        filtrado = filtrado[filtrado["setor"] == setor_sel]
    if busca and "empresa" in filtrado.columns:
        filtrado = filtrado[filtrado["empresa"].str.contains(busca, case=False, na=False)]

    st.caption(
        f"Exibindo **{min(int(limite), len(filtrado))}** de **{len(filtrado)}** propostas filtradas "
        f"(total na execução: **{len(propostas)}**)."
    )

    # ── tabela resumo (todas as linhas filtradas) ──
    st.markdown("##### Tabela resumo")
    cols_tab = ["lead_id"]
    if "empresa" in filtrado.columns:
        cols_tab += ["empresa", "setor", "orcamento_declarado"]
    cols_tab += ["tempo_gasto_min", "proposta_texto"]
    tabela = filtrado[cols_tab].copy()
    tabela["proposta_texto"] = tabela["proposta_texto"].str[:80] + "…"
    tabela.rename(columns={
        "lead_id": "Lead", "empresa": "Empresa", "setor": "Setor",
        "orcamento_declarado": "Orçamento (R$)",
        "tempo_gasto_min": "Tempo (min)", "proposta_texto": "Prévia da proposta",
    }, inplace=True)
    st.dataframe(
        tabela, use_container_width=True, hide_index=True,
        column_config={
            "Orçamento (R$)": st.column_config.NumberColumn(format="R$ {:,.0f}"),
            "Tempo (min)": st.column_config.NumberColumn(format="%.1f min"),
        },
    )

    # ── expanders com texto completo ──
    st.markdown("##### Textos completos")
    st.caption(
        "Clique em cada lead para expandir e ler o texto integral gerado pelo LLM. "
        "Ajuste o filtro de setor ou o campo de busca para focar em um perfil específico."
    )
    for i, (_, row) in enumerate(filtrado.head(int(limite)).iterrows()):
        texto = row.get("proposta_texto", "")
        if not texto:
            continue
        empresa = row.get("empresa", f"Lead {int(row['lead_id'])}")
        setor = row.get("setor", "")
        orcamento = row.get("orcamento_declarado", 0)
        label = f"Lead {int(row['lead_id'])} — {empresa}"
        if setor:
            label += f" ({setor})"
        if isinstance(orcamento, (int, float)):
            label += f" | R$ {orcamento:,.0f}"
        with st.expander(label):
            st.text_area(
                "Proposta gerada pelo Llama local", texto, height=180,
                disabled=True, key=f"{chave_prefix}_txt_{int(row['lead_id'])}_{i}",
            )
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Tempo simulado", f"{row['tempo_gasto_min']:.1f} min")
                st.caption("1,5 min = chamada ao LLM. 0,5 min = recuperado do cache.")
            with m2:
                st.metric("Origem", "Cache ♻️" if row["tempo_gasto_min"] <= 0.5 else "LLM ao vivo 🤖")
                st.caption("LLM ao vivo: texto gerado pelo Llama nesta execução. Cache: proposta reutilizada de execução anterior.")


# ── renderizacao principal ───────────────────────────────────────────────────

def renderizar_resultados_completos():
    st.markdown("## Resultados completos dos modelos de IA")
    st.caption(
        "Todos os outputs produzidos pelos modelos em uma execução — não apenas amostras. "
        "Use para confirmar que os modelos foram consultados de verdade para cada lead "
        "e para explorar os resultados em detalhe antes de uma apresentação."
    )

    col_estado, _ = st.columns([1.5, 2.5])
    with col_estado:
        estado = st.selectbox(
            "Estado", ["B", "C", "D"],
            format_func=lambda x: {
                "B": "Estado B — Melhoria pontual (LLM na proposta)",
                "C": "Estado C — Orquestração sistêmica",
                "D": "Estado D — Ablação (só qualificação por IA)",
            }[x],
            key="sel_resultado_completo",
        )

    caminho = ARQUIVOS[estado]
    if not os.path.exists(caminho) or os.path.getsize(caminho) == 0:
        st.warning(
            f"Nenhum histórico encontrado para o Estado {estado}. "
            "Execute a simulação na aba ▶ **Rodar ao vivo** primeiro."
        )
        return

    df = _carregar_eventos(caminho)
    leads = _carregar_leads()

    if df.empty:
        st.info("O histórico está vazio ou ainda sendo gerado.")
        return

    if estado == "B":
        _renderizar_b(df, leads)
    elif estado == "C":
        _renderizar_c(df, leads)
    else:
        _renderizar_d(df, leads)


# ── Estado B ─────────────────────────────────────────────────────────────────

def _renderizar_b(df: pd.DataFrame, leads: pd.DataFrame):
    propostas = df[df["etapa"] == "proposta_ia"].copy()
    propostas["proposta_texto"] = propostas["detalhes"].apply(
        lambda d: d.get("proposta_texto", "") if isinstance(d, dict) else ""
    )
    propostas = _enriquecer(propostas, leads)

    total = len(propostas)
    com_texto = int((propostas["proposta_texto"].str.len() > 0).sum())
    tempo_medio = propostas["tempo_gasto_min"].mean() if total else 0

    st.markdown("### 🟠 Propostas geradas pelo LLM — Estado B")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total de propostas", total)
        st.caption("Leads que chegaram à etapa de proposta após qualificação do SDR e reunião de discovery.")
    with c2:
        st.metric("Com texto gerado", com_texto)
        st.caption("Propostas com texto real gerado pelo Llama. Erros de conexão (Ollama offline) aparecem como aviso.")
    with c3:
        st.metric("Tempo médio por proposta", f"{tempo_medio:.1f} min")
        st.caption("1,5 min = LLM ao vivo. 0,5 min = cache. Compare com ~210 min (3,5h) da proposta manual do Estado A.")

    st.info(
        "💡 **O que isto demonstra:** no Estado A, cada proposta era elaborada manualmente "
        "pelo Closer em ~3,5 horas (formatação de slide, precificação artesanal). "
        "Aqui, o LLM local gera um texto equivalente em ~1,5 minuto. "
        "A taxa de conversão final não muda — mas o tempo liberado para o Closer é real e mensurável."
    )
    st.divider()
    _secao_propostas(propostas, "b")


# ── Estado C ─────────────────────────────────────────────────────────────────

def _renderizar_c(df: pd.DataFrame, leads: pd.DataFrame):
    tab_rf, tab_knn, tab_prop = st.tabs([
        "🟣 RandomForest — qualificação",
        "🔵 kNN — match de closer",
        "🟠 LLM — propostas",
    ])

    with tab_rf:
        _renderizar_rf(df, leads)
    with tab_knn:
        _renderizar_knn(df, leads)
    with tab_prop:
        _renderizar_propostas_c(df, leads)


# ── Estado D (ablação) ──────────────────────────────────────────────────────

def _renderizar_d(df: pd.DataFrame, leads: pd.DataFrame):
    st.info(
        "💡 **O que é o Estado D:** teste de ablação — usa o **mesmo RandomForest** do "
        "Estado C na qualificação, mas sem kNN de match e sem LLM na proposta (que volta "
        "a ser manual, como no Estado A). Cardinalidade de intervenção 1, igual ao Estado B, "
        "só que em local diferente: a restrição do processo, não a etapa mais cara em tempo."
    )
    tab_rf, tab_prop = st.tabs([
        "🟣 RandomForest — qualificação",
        "🔧 Proposta — manual",
    ])
    with tab_rf:
        _renderizar_rf(df, leads)
    with tab_prop:
        _renderizar_proposta_manual_d(df, leads)


def _renderizar_proposta_manual_d(df: pd.DataFrame, leads: pd.DataFrame):
    propostas = df[df["etapa"] == "proposta_manual"].copy()
    propostas = _enriquecer(propostas, leads)

    total = len(propostas)
    tempo_medio_min = propostas["tempo_gasto_min"].mean() if total else 0

    st.markdown("### 🔧 Proposta manual — Estado D")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total de propostas", total)
        st.caption("Leads que chegaram à etapa de proposta após qualificação por RandomForest e reunião de discovery.")
    with c2:
        st.metric("Tempo médio por proposta", f"{tempo_medio_min/60:.1f} h" if total else "—")
        st.caption("Elaboração manual — slide e precificação artesanal, igual ao Estado A. Compare com ~1,5 min do LLM (Estados B/C).")

    st.info(
        "💡 **Por que ainda é manual aqui:** o Estado D isola deliberadamente a intervenção de "
        "IA na qualificação — a proposta permanece manual para medir o efeito de UMA intervenção "
        "por vez, na MESMA cardinalidade do Estado B, mas em local diferente. "
        "Compare com a aba 🔀 Comparar B vs D."
    )

    if total == 0:
        return

    st.markdown("##### Tabela de propostas")
    cols_tab = ["lead_id"]
    if "empresa" in propostas.columns:
        cols_tab += ["empresa", "setor", "orcamento_declarado"]
    cols_tab += ["tempo_gasto_min"]
    tabela = propostas[cols_tab].copy()
    tabela["tempo_gasto_min"] = (tabela["tempo_gasto_min"] / 60).round(2)
    tabela.rename(columns={
        "lead_id": "Lead", "empresa": "Empresa", "setor": "Setor",
        "orcamento_declarado": "Orçamento (R$)", "tempo_gasto_min": "Tempo (h)",
    }, inplace=True)
    st.dataframe(
        tabela, use_container_width=True, hide_index=True,
        column_config={
            "Orçamento (R$)": st.column_config.NumberColumn(format="R$ {:,.0f}"),
            "Tempo (h)": st.column_config.NumberColumn(format="%.2f h"),
        },
    )


def _renderizar_rf(df: pd.DataFrame, leads: pd.DataFrame):
    ev = df[df["etapa"] == "qualificacao_ia"].copy()
    ev["score_rf"] = ev["detalhes"].apply(
        lambda d: d.get("probabilidade_fechamento") if isinstance(d, dict) else None
    )
    ev["threshold"] = ev["detalhes"].apply(
        lambda d: d.get("threshold_usado") if isinstance(d, dict) else None
    )
    ev = _enriquecer(ev, leads)

    total = len(ev)
    qual = int(ev["passou"].sum())
    desc = total - qual

    st.markdown("### 🟣 RandomForest — decisões de qualificação")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Leads analisados", total)
        st.caption("Todos os leads entram no RF na chegada — sem triagem humana prévia.")
    with c2:
        st.metric("Qualificados ✅", qual)
        st.caption("Score acima do threshold → lead avança para Discovery com o Closer.")
    with c3:
        st.metric("Descartados ❌", desc)
        st.caption("Score abaixo do threshold → eliminado sem custo de tempo humano.")
    with c4:
        st.metric("Taxa de qualificação", f"{qual/total:.1%}" if total else "—")
        st.caption("Calibrado pelo percentil 50 dos scores — comparável ao SDR humano do Estado A (~50%).")


    st.caption(
        "💡 **Como ler a tabela:** a coluna **Score RF** é a probabilidade estimada pelo modelo de que "
        "aquele lead vai fechar negócio, com base em orçamento, urgência e tamanho da empresa. "
        "A barra de progresso mostra o valor visualmente (vazia = 0%, cheia = 100%). "
        "Leads com score acima do threshold são **qualificados**; os demais são **descartados** automaticamente, "
        "em 0,2 minuto por lead — vs. ~25 minutos do SDR humano no Estado A."
    )

    cols_tab = ["lead_id"]
    if "empresa" in ev.columns:
        cols_tab += ["empresa", "setor", "orcamento_declarado", "urgencia_declarada"]
    cols_tab += ["score_rf", "threshold", "passou"]

    tabela = ev[cols_tab].copy()
    tabela["resultado"] = tabela["passou"].map({True: "✅ Qualificado", False: "❌ Descartado"})
    tabela.drop(columns=["passou"], inplace=True)
    tabela.rename(columns={
        "lead_id": "Lead", "empresa": "Empresa", "setor": "Setor",
        "orcamento_declarado": "Orçamento (R$)", "urgencia_declarada": "Urgência",
        "score_rf": "Score RF", "threshold": "Threshold", "resultado": "Resultado",
    }, inplace=True)

    st.dataframe(
        tabela, use_container_width=True, hide_index=True,
        column_config={
            "Orçamento (R$)": st.column_config.NumberColumn(format="R$ {:,.0f}"),
            "Urgência": st.column_config.NumberColumn(format="%.3f"),
            "Score RF": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.3f",
                help="Probabilidade de fechamento estimada pelo RandomForest. Acima do threshold → qualificado.",
            ),
            "Threshold": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def _renderizar_knn(df: pd.DataFrame, leads: pd.DataFrame):
    ev = df[df["etapa"] == "match_closer_ia"].copy()
    ev["similaridade"] = ev["detalhes"].apply(
        lambda d: d.get("perfil_similaridade") if isinstance(d, dict) else None
    )
    ev = _enriquecer(ev, leads)

    total = len(ev)
    sim_media = ev["similaridade"].mean() if not ev.empty else 0
    sim_max = ev["similaridade"].max() if not ev.empty else 0
    sim_min = ev["similaridade"].min() if not ev.empty else 0

    st.markdown("### 🔵 kNN — matches de perfil realizados")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total de matches", total)
        st.caption("Um match é feito para cada lead qualificado pelo RandomForest.")
    with k2:
        st.metric("Similaridade média", f"{sim_media:.4f}")
        st.caption("1 ÷ (1 + distância euclidiana média aos 5 deals ganhos mais próximos). Mais alto = mais parecido com sucessos históricos.")
    with k3:
        st.metric("Maior similaridade", f"{sim_max:.4f}")
        st.caption("O lead mais parecido com deals ganhos do histórico do Estado A nesta execução.")
    with k4:
        st.metric("Menor similaridade", f"{sim_min:.4f}")
        st.caption("O lead menos parecido — ainda assim qualificado pelo RF e alocado ao Closer mais adequado.")

    st.caption(
        "💡 **Como ler a tabela:** a coluna **Similaridade** mostra o quão parecido o perfil do lead é "
        "com os deals ganhos do histórico do Estado A. "
        "Valores mais altos indicam leads em regiões de orçamento e urgência onde a empresa já tem histório de sucesso. "
        "Essa similaridade guia a alocação do Closer: leads com perfil parecido com sucessos passados recebem o "
        "Closer com melhor desempenho naquele tipo de conta. "
        "O processo leva **0,1 minuto** por lead — praticamente instantâneo."
    )

    cols_knn = ["lead_id"]
    if "empresa" in ev.columns:
        cols_knn += ["empresa", "setor", "orcamento_declarado", "urgencia_declarada"]
    cols_knn += ["similaridade"]

    tabela_knn = ev[cols_knn].copy()
    tabela_knn.rename(columns={
        "lead_id": "Lead", "empresa": "Empresa", "setor": "Setor",
        "orcamento_declarado": "Orçamento (R$)", "urgencia_declarada": "Urgência",
        "similaridade": "Similaridade de perfil",
    }, inplace=True)

    st.dataframe(
        tabela_knn, use_container_width=True, hide_index=True,
        column_config={
            "Orçamento (R$)": st.column_config.NumberColumn(format="R$ {:,.0f}"),
            "Urgência": st.column_config.NumberColumn(format="%.3f"),
            "Similaridade de perfil": st.column_config.ProgressColumn(
                min_value=0, max_value=0.01, format="%.4f",
                help="Score de similaridade com deals ganhos históricos. Próximo de 0.01 = muito parecido com um deal ganho.",
            ),
        },
    )


def _renderizar_propostas_c(df: pd.DataFrame, leads: pd.DataFrame):
    propostas = df[df["etapa"] == "proposta_ia"].copy()
    propostas["proposta_texto"] = propostas["detalhes"].apply(
        lambda d: d.get("proposta_texto", "") if isinstance(d, dict) else ""
    )
    propostas = _enriquecer(propostas, leads)

    total = len(propostas)
    com_texto = int((propostas["proposta_texto"].str.len() > 0).sum())
    tempo_medio = propostas["tempo_gasto_min"].mean() if total else 0

    st.markdown("### 🟠 LLM — propostas geradas")

    p1, p2, p3 = st.columns(3)
    p1.metric(
        "Total de propostas", total,
        help="Leads que passaram por qualificação (RF), match (kNN) e pela reunião de discovery.",
    )
    p2.metric(
        "Com texto gerado", com_texto,
        help="Propostas com texto real gerado com sucesso pelo Llama local.",
    )
    p3.metric(
        "Tempo médio por proposta", f"{tempo_medio:.1f} min",
        help="1,5 min = chamada nova ao LLM. 0,5 min = recuperado do cache. Compare com ~210 min do Estado A.",
    )

    st.info(
        "💡 **Diferença em relação ao Estado B:** os leads que chegam aqui já foram pré-selecionados "
        "pelo RandomForest — são leads com score acima da mediana histórica. "
        "O texto é gerado pelo mesmo LLM, mas a qualidade média da base de entrada é superior, "
        "pois o RandomForest descartou leads com baixa probabilidade de fechamento antes mesmo do discovery."
    )
    st.divider()
    _secao_propostas(propostas, "c")
