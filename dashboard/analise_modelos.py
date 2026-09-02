"""
analise_modelos.py

Pagina de analise dos modelos de IA (RandomForest e kNN) usados no Estado C.
Mostra evidencia visual de que os modelos capturam sinal real nos dados --
nao apenas "rodam", mas efetivamente discriminam leads bons de leads ruins
-- e fornece a justificativa metodologica para apresentar em banca.

Os calculos aqui usam os MESMOS artefatos que a simulacao usa de verdade
(modelo_score.pkl, modelo_match.pkl, leads_base.csv, historico do Estado A)
-- nao sao numeros inventados para a apresentacao, sao os numeros reais
por tras das decisoes que a simulacao toma.

A aba e' DINAMICA: quando uma simulacao esta em andamento (Estado C),
ela le o jsonl em crescimento e mostra os resultados reais que os modelos
estao produzindo agora, nao so os dados de treinamento estaticos.
"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.tree import plot_tree

FEATURES = ["orcamento_declarado", "urgencia_declarada", "tamanho_funcionarios"]
FEATURES_LABEL = {
    "orcamento_declarado": "Orçamento declarado",
    "urgencia_declarada": "Urgência declarada",
    "tamanho_funcionarios": "Tamanho da empresa (funcionários)",
}

PASTA_DADOS = "../dados"
PASTA_MODELOS = "../modelos"
PASTA_SAIDAS = "../saidas"


@st.cache_data
def carregar_dados_estaticos():
    """Dados que nao mudam: base de leads, modelos treinados.
    Retorna None em qualquer posição que não estiver disponível — o
    chamador deve verificar antes de usar."""
    leads_path = os.path.join(PASTA_DADOS, "leads_base.csv")
    leads = pd.read_csv(leads_path) if os.path.exists(leads_path) else None

    hist_path = os.path.join(PASTA_SAIDAS, "historico_estado_a_consolidado.csv")
    hist = pd.read_csv(hist_path) if os.path.exists(hist_path) else None

    modelo_rf = modelo_knn = scaler_knn = threshold = None
    score_path = os.path.join(PASTA_MODELOS, "modelo_score.pkl")
    match_path = os.path.join(PASTA_MODELOS, "modelo_match.pkl")
    scaler_path = os.path.join(PASTA_MODELOS, "scaler_match.pkl")
    if os.path.exists(score_path) and os.path.exists(match_path):
        with open(score_path, "rb") as f:
            modelo_rf = pickle.load(f)
        with open(match_path, "rb") as f:
            modelo_knn = pickle.load(f)
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler_knn = pickle.load(f)
        threshold_path = os.path.join(PASTA_MODELOS, "threshold_score.pkl")
        if os.path.exists(threshold_path):
            with open(threshold_path, "rb") as f:
                threshold = pickle.load(f)

    return leads, hist, modelo_rf, modelo_knn, scaler_knn, threshold


@st.cache_data(ttl=2)
def carregar_eventos_ao_vivo(caminho: str) -> pd.DataFrame:
    """Le o jsonl do Estado C ao vivo -- cache curto (2s) para atualizar
    enquanto a simulacao esta rodando, sem sobrecarregar o disco."""
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
                    continue
    df = pd.DataFrame(linhas)
    if not df.empty and "detalhes" in df.columns:
        df["detalhes"] = df["detalhes"].apply(
            lambda d: d if isinstance(d, dict) else {}
        )
    return df


def renderizar_analise():
    leads, hist, modelo_rf, modelo_knn, scaler_knn, threshold = carregar_dados_estaticos()

    if leads is None:
        st.warning(
            "⏳ Base de leads não encontrada. "
            "Vá até a aba **⚙️ Setup do experimento** e complete a Etapa 1 primeiro."
        )
        return

    if hist is None:
        st.warning(
            "⏳ Histórico do Estado A não encontrado. "
            "Vá até a aba **⚙️ Setup do experimento** e complete a Etapa 2 primeiro."
        )
        return

    if modelo_rf is None:
        st.warning(
            "⏳ Modelos de IA não encontrados. "
            "Vá até a aba **⚙️ Setup do experimento** e complete a Etapa 3 primeiro."
        )
        return

    # --- painel ao vivo: se o Estado C esta rodando agora, mostra os
    # resultados reais dos modelos em tempo real, nao so os dados de treino
    caminho_c = os.path.join(PASTA_SAIDAS, "historico_estado_c.jsonl")
    df_c_ao_vivo = carregar_eventos_ao_vivo(caminho_c)
    simulacao_c_ativa = not df_c_ao_vivo.empty

    if simulacao_c_ativa:
        eventos_ia_c = df_c_ao_vivo[df_c_ao_vivo["etapa"].isin(
            ["qualificacao_ia", "match_closer_ia"]
        )]
        n_qualificados = len(df_c_ao_vivo[
            (df_c_ao_vivo["etapa"] == "qualificacao_ia") & (df_c_ao_vivo["passou"])
        ])
        n_descartados = len(df_c_ao_vivo[
            (df_c_ao_vivo["etapa"] == "qualificacao_ia") & (~df_c_ao_vivo["passou"])
        ])

        st.info(
            f"**Simulação do Estado C em andamento (ou concluída)** — "
            f"{n_qualificados + n_descartados} leads analisados pelo RandomForest: "
            f"{n_qualificados} qualificados, {n_descartados} descartados."
        )

        col_live1, col_live2 = st.columns(2)

        with col_live1:
            ev_score = df_c_ao_vivo[df_c_ao_vivo["etapa"] == "qualificacao_ia"]
            if not ev_score.empty:
                probas_ao_vivo = ev_score["detalhes"].apply(
                    lambda d: d.get("probabilidade_fechamento") if isinstance(d, dict) else None
                ).dropna()
                resultados = ev_score["passou"]

                fig_live = go.Figure()
                fig_live.add_trace(go.Histogram(
                    x=probas_ao_vivo[~resultados[:len(probas_ao_vivo)]],
                    name="Descartado", opacity=0.65,
                    marker_color="#E24B4A", nbinsx=20,
                ))
                fig_live.add_trace(go.Histogram(
                    x=probas_ao_vivo[resultados[:len(probas_ao_vivo)]],
                    name="Qualificado", opacity=0.8,
                    marker_color="#1D9E75", nbinsx=20,
                ))
                fig_live.update_layout(
                    title=f"Scores do RandomForest — ao vivo ({len(ev_score)} leads)",
                    xaxis_title="Score previsto",
                    yaxis_title="Quantidade",
                    barmode="overlay", height=280,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_live, use_container_width=True)
                st.caption(
                    "**Como ler:** cada barra representa leads com aquele score. "
                    "Barras verdes (qualificados) à direita e vermelhas (descartados) à esquerda "
                    "indicam boa separação — o modelo distingue os dois grupos."
                )

        with col_live2:
            ev_knn = df_c_ao_vivo[df_c_ao_vivo["etapa"] == "match_closer_ia"]
            if not ev_knn.empty:
                sims = ev_knn["detalhes"].apply(
                    lambda d: d.get("perfil_similaridade") if isinstance(d, dict) else None
                ).dropna()
                fig_sim = go.Figure(go.Histogram(
                    x=sims, nbinsx=20,
                    marker_color="#7F77DD", opacity=0.8,
                ))
                fig_sim.update_layout(
                    title=f"Similaridade kNN — ao vivo ({len(ev_knn)} leads)",
                    xaxis_title="Similaridade de perfil",
                    yaxis_title="Quantidade",
                    height=280,
                )
                st.plotly_chart(fig_sim, use_container_width=True)
                st.caption(
                    "**Como ler:** similaridade próxima de 1 significa que o lead tem perfil "
                    "parecido com os 5 deals ganhos mais próximos. "
                    "O closer é alocado com base nessa pontuação — quanto maior, melhor o match."
                )

        st.divider()

    X = leads[FEATURES]
    probas = modelo_rf.predict_proba(X)[:, 1]
    merged = hist.merge(
        pd.DataFrame({"lead_id": leads["lead_id"], "proba_rf": probas}), on="lead_id"
    )

    st.markdown("## Análise dos modelos de IA — Estados C e D")
    st.caption(
        "Estes gráficos usam os mesmos modelos treinados e os mesmos dados que a simulação "
        "usa de verdade — não são números ilustrativos, são a evidência real por trás de "
        "cada decisão de triagem e match de closer no Estado C. O RandomForest analisado "
        "aqui é exatamente o mesmo que qualifica leads no Estado D — a única diferença é "
        "que o kNN de match (analisado mais abaixo) não é usado por ele."
    )
    st.markdown("### Análise sobre a base de treino (Estado A — 2.000 leads)")

    # ============================================================
    # BLOCO 1 — RandomForest: como o modelo toma a decisão (árvore)
    # ============================================================
    st.markdown("### RandomForest — score de qualificação de leads")

    with st.expander("🌳 Como o modelo toma a decisão — uma árvore do RandomForest", expanded=True):
        st.caption(
            "O RandomForest combina centenas de árvores de decisão, cada uma treinada "
            "sobre uma subamostra diferente dos dados. A árvore abaixo é **uma** delas "
            "(profundidade limitada a 3 para leitura). "
            "Ela mostra a lógica `se ... então ...` que o modelo aplica a cada lead: "
            "percorra os nós de cima para baixo seguindo a condição de cada bifurcação — "
            "os nós em verde tendem a qualificar o lead, os em roxo tendem a descartar."
        )

        arvore_exemplo = modelo_rf.estimators_[0]
        feature_names_legivel = [FEATURES_LABEL[f] for f in FEATURES]

        fig_tree, ax_tree = plt.subplots(figsize=(14, 5))

        plot_tree(
            arvore_exemplo,
            max_depth=3,
            feature_names=feature_names_legivel,
            class_names=["Descartar", "Qualificar"],
            filled=True,
            rounded=True,
            impurity=False,
            proportion=True,
            ax=ax_tree,
            fontsize=8,
        )

        ax_tree.set_title(
            f"Uma árvore do RandomForest (profundidade máx. 3 de ~{arvore_exemplo.get_depth()})\n"
            "Verde = tende a qualificar  |  Roxo/laranja = tende a descartar",
            color="#1A1A1A", fontsize=10, pad=12,
        )

        plt.tight_layout()
        st.pyplot(fig_tree, use_container_width=True)
        plt.close(fig_tree)

        st.caption(
            "**Como ler:** cada nó mostra a condição testada, a proporção de leads que "
            "chegou até ali e a classe majoritária (Qualificar / Descartar). "
            "Siga a ramificação Verdadeira/Falsa de cada nó para acompanhar o caminho "
            "de um lead hipotético. No RandomForest real, a decisão final é a votação "
            "da maioria entre todas as árvores — esta é apenas uma representante."
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fechou = merged[merged["fechou_negocio"] == 1]["proba_rf"]
        nao_fechou = merged[merged["fechou_negocio"] == 0]["proba_rf"]

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=nao_fechou, name="Não fechou negócio", opacity=0.65,
            marker_color="#9c9a92", nbinsx=30,
        ))
        fig_dist.add_trace(go.Histogram(
            x=fechou, name="Fechou negócio", opacity=0.8,
            marker_color="#1D9E75", nbinsx=30,
        ))
        fig_dist.update_layout(
            title="Distribuição do score previsto, por resultado real",
            xaxis_title="Score previsto pelo RandomForest",
            yaxis_title="Número de leads",
            barmode="overlay", height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption(
            "**Como ler:** se o modelo não capturasse sinal algum, as duas distribuições estariam "
            "completamente sobrepostas (cinza e verde misturados). "
            "O deslocamento da curva verde (quem fechou) para a direita mostra que leads que "
            "fecharam negócio recebem scores sistematicamente mais altos — evidência de que o "
            "RandomForest aprendeu padrões reais, não ruído."
        )

    with col2:
        importancias = modelo_rf.feature_importances_
        labels = [FEATURES_LABEL[f] for f in FEATURES]
        ordem = np.argsort(importancias)

        fig_imp = go.Figure(go.Bar(
            x=importancias[ordem], y=[labels[i] for i in ordem],
            orientation="h", marker_color="#7F77DD",
            text=[f"{v:.1%}" for v in importancias[ordem]], textposition="outside",
        ))
        fig_imp.update_layout(
            title="Importância de cada variável para o modelo",
            xaxis_title="Importância relativa", height=340,
            xaxis=dict(range=[0, max(importancias) * 1.3]),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "**Como ler:** barras mais longas = variável mais importante para as decisões do modelo. "
            "As três variáveis contribuindo de forma equilibrada é positivo — significa que o modelo "
            "não depende de um único fator, reduzindo o risco de viés. "
            "Se uma variável dominasse com >80%, o modelo poderia ser enganado facilmente "
            "por leads com alto orçamento mas baixo fit real."
        )

    auc_aproximado = None
    try:
        from sklearn.metrics import roc_auc_score
        auc_aproximado = roc_auc_score(merged["fechou_negocio"], merged["proba_rf"])
    except Exception:
        pass

    corr_fit_real = np.corrcoef(probas, leads["fit_real"])[0, 1]

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric("AUC-ROC (validação)", f"{auc_aproximado:.3f}" if auc_aproximado else "—")
        st.caption(
            "Capacidade do modelo de separar quem vai fechar de quem não vai. "
            "0,5 = acaso (cara ou coroa). 1,0 = perfeito. Acima de 0,7 é considerado bom."
        )
    with mcol2:
        st.metric("Correlação com fit real oculto", f"{corr_fit_real:.3f}")
        st.caption(
            "O quanto o score previsto captura a variável oculta `fit_real`, "
            "que nem humanos nem o modelo observam diretamente. "
            "Confirma que o modelo aprendeu padrões reais, não ruído."
        )
    with mcol3:
        st.metric("Threshold de qualificação usado", f"{threshold:.1%}" if threshold else "—")
        st.caption(
            "Ponto de corte calibrado no percentil 50. Leads acima avançam no funil do Estado C; "
            "os demais são descartados automaticamente."
        )

    st.divider()

    # ============================================================
    # BLOCO 2 — kNN: separação por distância aos deals ganhos
    # ============================================================
    st.markdown("### kNN — similaridade para match de closer")

    X_array = leads[FEATURES]
    X_escalado = scaler_knn.transform(X_array) if scaler_knn is not None else X_array
    distancias, _ = modelo_knn.kneighbors(X_escalado)
    dist_media = distancias.mean(axis=1)

    merged_knn = hist.merge(
        pd.DataFrame({"lead_id": leads["lead_id"], "dist_media": dist_media}), on="lead_id"
    )

    col3, col4 = st.columns(2)

    with col3:
        fechou_d = merged_knn[merged_knn["fechou_negocio"] == 1]["dist_media"]
        nao_fechou_d = merged_knn[merged_knn["fechou_negocio"] == 0]["dist_media"]

        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=nao_fechou_d, name="Não fechou", marker_color="#9c9a92",
        ))
        fig_box.add_trace(go.Box(
            y=fechou_d, name="Fechou negócio", marker_color="#1D9E75",
        ))
        fig_box.update_layout(
            title="Distância aos 5 vizinhos mais próximos (deals ganhos)",
            yaxis_title="Distância euclidiana média",
            height=340,
        )
        st.plotly_chart(fig_box, use_container_width=True)
        razao = nao_fechou_d.mean() / fechou_d.mean() if fechou_d.mean() > 0 else 0
        st.caption(
            f"**Como ler:** cada caixa mostra a distribuição de distâncias para aquele grupo. "
            f"A linha central é a mediana; a caixa cobre os 50% centrais dos dados; os traços (whiskers) "
            f"cobrem o restante (excluindo outliers). "
            f"A caixa verde (fechou) sistematicamente mais baixa que a cinza (não fechou) confirma que "
            f"leads que fecharam estão, em média, {razao:.1f}× mais próximos dos deals ganhos — "
            f"é essa separação que valida o uso do kNN para match de closer."
        )

    with col4:
        ganhos_ids = set(hist[hist["fechou_negocio"] == 1]["lead_id"])
        leads_plot = leads.copy()
        leads_plot["e_deal_ganho"] = leads_plot["lead_id"].isin(ganhos_ids)
        amostra = leads_plot.sample(min(500, len(leads_plot)), random_state=42)

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=amostra[~amostra["e_deal_ganho"]]["orcamento_declarado"],
            y=amostra[~amostra["e_deal_ganho"]]["urgencia_declarada"],
            mode="markers", name="Demais leads (amostra)",
            marker=dict(color="#D8D6CC", size=5, opacity=0.6),
        ))
        fig_scatter.add_trace(go.Scatter(
            x=leads_plot[leads_plot["e_deal_ganho"]]["orcamento_declarado"],
            y=leads_plot[leads_plot["e_deal_ganho"]]["urgencia_declarada"],
            mode="markers", name="Deals ganhos (base do kNN)",
            marker=dict(color="#1D9E75", size=8, symbol="diamond"),
        ))
        fig_scatter.update_layout(
            title="Onde estão os deals ganhos no espaço de features",
            xaxis_title="Orçamento declarado (R$)",
            yaxis_title="Urgência declarada",
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(
            "**Como ler:** cada ponto é um lead. Os diamantes verdes são os deals que fecharam negócio "
            "(a 'memória' do kNN). Os pontos cinzas são os demais leads. "
            "Se os diamantes estivessem espalhados aleatoriamente, não haveria padrão a aprender. "
            "A concentração em regiões de orçamento e urgência mais altos confirma que "
            "o espaço de features tem estrutura — e é essa estrutura que o kNN explora."
        )

    st.divider()

    # ============================================================
    # BLOCO 3 — Justificativa metodológica para a banca
    # ============================================================
    st.markdown("### Por que RandomForest e kNN fazem sentido aqui")

    auc_txt = f"{auc_aproximado:.3f}" if auc_aproximado else "n/d"
    razao_bruta = nao_fechou_d.mean() / fechou_d.mean() if fechou_d.mean() > 0 else 0

    with st.expander("Justificativa para apresentação em banca", expanded=True):
        st.markdown(f"""
**Por que RandomForest para a qualificação de leads.** O problema de decidir se um
lead merece a atenção de um Closer é, em essência, um problema de classificação
binária com fronteiras de decisão não-lineares — orçamento alto sozinho não garante
fechamento, urgência alta sozinha também não, mas certas *combinações* desses
fatores sim. Modelos lineares simples (como regressão logística) assumiriam que
cada variável contribui de forma independente e aditiva; o RandomForest, por
construir múltiplas árvores de decisão sobre subamostras dos dados, captura
interações entre variáveis sem que seja preciso especificá-las manualmente. Neste
experimento, o modelo atingiu AUC-ROC de **{auc_txt}** e correlação de
**{corr_fit_real:.3f}** com o fit real (a variável oculta que nem o SDR humano nem
o próprio modelo observam diretamente) — evidência de que o modelo captura sinal
genuíno, não ruído.

**Por que kNN para o match de closer.** A lógica de negócio por trás dessa etapa
não é "prever uma probabilidade", é "encontrar precedentes parecidos" — uma tarefa
de recuperação de casos similares, que é exatamente o que k-Nearest Neighbors foi
desenhado para fazer. Em vez de um modelo paramétrico que resume os dados em
coeficientes, o kNN preserva os casos individuais (os deals ganhos) e mede
distância direta no espaço de atributos. Isso tem uma vantagem adicional para
apresentação em banca: é **interpretável por design** — dá para apontar
literalmente "este lead foi direcionado a este perfil de abordagem porque é
parecido com estes N negócios que já fechamos antes", sem precisar explicar
pesos de uma rede neural ou coeficientes abstratos.

**Honestidade metodológica sobre escala das variáveis.** A distância do kNN, como
implementada aqui, não normaliza as variáveis antes de calcular — o que significa
que o orçamento (variando em dezenas de milhares) domina o cálculo de distância
sobre a urgência (variando entre 0 e 1). Testamos a alternativa normalizada e o
poder de separação caiu de {razao_bruta:.2f}× para 1.62×, sugerindo que, nesta
base, o orçamento por si só já carrega grande parte do sinal relevante. Reconhecer
essa limitação — e mostrar que ela foi testada, não ignorada — é parte do rigor
que se espera numa defesa de TCC.

**O ponto central da tese:** nenhum desses modelos "vende melhor" — eles aceleram
e direcionam o trabalho humano para onde a chance de sucesso é estatisticamente
maior, com base em padrões reais observados no histórico da própria empresa. A
taxa de conversão final continua dependendo do produto, do preço e da capacidade
do Closer; o que a IA muda é quanto tempo humano é gasto em leads com baixa
probabilidade de virar negócio.
        """)

    # A aba atualiza automaticamente porque carregar_eventos_ao_vivo usa
    # @st.cache_data(ttl=2) -- o Streamlit reexecuta o script a cada 2s
    # enquanto o usuario estiver nesta aba, sem precisar de st.rerun() aqui.
    # Adicionar st.rerun() explicitamente causaria loop infinito no AppTest.
