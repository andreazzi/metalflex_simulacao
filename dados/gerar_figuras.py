"""
gerar_figuras.py

Gera as figuras de comparacao entre os 4 estados DIRETO dos CSVs de
resultado -- PNG a 300 dpi e SVG (vetorial), prontas para a monografia e o
repositorio publico. Nao depende de captura de tela do dashboard (ver
ESPEC_DASHBOARD_4_ESTADOS.md).

Uso:
    python gerar_figuras.py                              # acha o CSV oficial mais recente
    python gerar_figuras.py --csv ../saidas/experimento_100x_replicas.csv

Saida:
    ../saidas/figuras/conversao_por_estado.png  (+ .svg)
    ../saidas/figuras/tempo_por_estado.png      (+ .svg)
    ../saidas/figuras/comparacao_completa.png   (+ .svg)  -- as duas lado a lado, como a Figura 17
    ../saidas/figuras/knn_separacao.png         (+ .svg)  -- separacao medida pelo kNN
    ../saidas/figuras/funil_estado_a.png        (+ .svg)  -- volume por etapa no Estado A
    ../saidas/figuras/distribuicoes_modelos.png (+ .svg)  -- escores do RF e similaridade do kNN
    ../saidas/figuras/escore_e_importancia.png  (+ .svg)  -- escore por desfecho e peso dos atributos
    ../saidas/figuras/estrutura_repositorio.png (+ .svg)  -- arvore de diretorios do repositorio

A figura do kNN reproduz, em matplotlib, os dois paineis que o dashboard
desenha em Plotly (dashboard/analise_modelos.py, BLOCO 2): as mesmas fontes
(leads_base.csv + historico_estado_a_consolidado.csv), a mesma padronizacao
(scaler_match.pkl) e a mesma amostragem do scatter (random_state=42). O
objetivo e' que a figura da monografia seja gerada por terminal, sem captura
de tela, e conferivel contra o painel.
"""

import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PASTA_SAIDAS = "../saidas"
PASTA_FIGURAS = "../saidas/figuras"
PASTA_MODELOS = "../modelos"
PASTA_DADOS = "."

# Mesmas features e cores do dashboard (dashboard/analise_modelos.py).
FEATURES_KNN = ["orcamento_declarado", "urgencia_declarada", "tamanho_funcionarios"]
COR_FECHOU = "#1D9E75"
COR_NAO_FECHOU = "#9c9a92"
COR_DEMAIS = "#D8D6CC"

# Ordem de exibicao A -> B -> D -> C (nao alfabetica): ordem argumentativa
# do capitulo -- B e D lado a lado (mesma cardinalidade de intervencao),
# C fecha como o redesenho completo. Ver ESPEC_DASHBOARD_4_ESTADOS.md.
ORDEM_ESTADOS = ["Estado A", "Estado B", "Estado D", "Estado C"]
CORES_ESTADO = {"Estado A": "#888780", "Estado B": "#D85A30", "Estado C": "#1D9E75", "Estado D": "#3A6EA5"}
NOMES_CURTOS = {"Estado A": "A", "Estado B": "B", "Estado C": "C", "Estado D": "D"}


def encontrar_csv_oficial() -> str:
    candidatos = [
        f for f in glob.glob(os.path.join(PASTA_SAIDAS, "experimento_*x_replicas.csv"))
        if "_smoke" not in f
    ]
    if not candidatos:
        raise FileNotFoundError(
            "Nenhum experimento_Nx_replicas.csv encontrado em ../saidas/. "
            "Rode experimento_30x.py --n-repeticoes N primeiro, ou passe --csv explicitamente."
        )

    def n_de(caminho):
        nome = os.path.basename(caminho)
        try:
            return int(nome.split("_")[1].rstrip("x"))
        except (IndexError, ValueError):
            return 0

    candidatos.sort(key=n_de, reverse=True)
    return candidatos[0]


def _barra(ax, df: pd.DataFrame, coluna: str, titulo: str, formato_valor):
    medias = df.groupby("estado")[coluna].mean().reindex(ORDEM_ESTADOS)
    desvios = df.groupby("estado")[coluna].std().reindex(ORDEM_ESTADOS)
    x = range(len(ORDEM_ESTADOS))
    cores = [CORES_ESTADO[e] for e in ORDEM_ESTADOS]

    ax.bar(x, medias.values, yerr=desvios.values, color=cores, capsize=4,
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([NOMES_CURTOS[e] for e in ORDEM_ESTADOS], fontsize=11)
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(medias.values):
        ax.text(i, v + desvios.values[i] * 1.05, formato_valor(v), ha="center", va="bottom", fontsize=9)
    return medias, desvios


def gerar(caminho_csv: str, n_replicas: int):
    df = pd.read_csv(caminho_csv)
    os.makedirs(PASTA_FIGURAS, exist_ok=True)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["svg.fonttype"] = "none"  # texto continua editavel no SVG, nao vira path

    # --- figura 1: conversao sozinha ---
    fig, ax = plt.subplots(figsize=(5, 4))
    _barra(ax, df, "taxa_conversao_pct", f"Taxa de conversão por estado (n = {n_replicas})",
           lambda v: f"{v:.2f}%")
    ax.set_ylabel("Taxa de conversão (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, "conversao_por_estado.png"), dpi=300)
    fig.savefig(os.path.join(PASTA_FIGURAS, "conversao_por_estado.svg"))
    plt.close(fig)

    # --- figura 2: tempo sozinho ---
    fig, ax = plt.subplots(figsize=(5, 4))
    _barra(ax, df, "tempo_total_h", f"Tempo total simulado por estado (n = {n_replicas})",
           lambda v: f"{v:.0f}h")
    ax.set_ylabel("Tempo total (horas)")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, "tempo_por_estado.png"), dpi=300)
    fig.savefig(os.path.join(PASTA_FIGURAS, "tempo_por_estado.svg"))
    plt.close(fig)

    # --- figura 3: as duas lado a lado (formato Figura 17 da monografia) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    _barra(ax1, df, "taxa_conversao_pct", "Taxa de conversão", lambda v: f"{v:.2f}%")
    ax1.set_ylabel("Taxa de conversão (%)")
    _barra(ax2, df, "tempo_total_h", "Tempo total simulado", lambda v: f"{v:.0f}h")
    ax2.set_ylabel("Tempo total (horas)")
    fig.suptitle(f"Comparação entre os quatro estados (média ± desvio-padrão, n = {n_replicas} execuções)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, "comparacao_completa.png"), dpi=300)
    fig.savefig(os.path.join(PASTA_FIGURAS, "comparacao_completa.svg"))
    plt.close(fig)

    print(f"Figuras geradas a partir de {caminho_csv} (n={n_replicas} réplicas) em {PASTA_FIGURAS}/:")
    for nome in ["conversao_por_estado", "tempo_por_estado", "comparacao_completa"]:
        print(f"  {nome}.png (300 dpi), {nome}.svg")

    # Figura 20: nao vem do CSV de replicas, e sim dos modelos e da base de
    # comparacao -- por isso e' gerada a parte.
    gerar_figura_knn()
    gerar_figura_funil()
    gerar_figuras_modelos()
    gerar_figura_arvore()


def gerar_figura_knn() -> bool:
    """Figura 20 -- separacao de perfil medida pelo kNN na base de comparacao.

    Painel esquerdo: distribuicao da distancia euclidiana media aos 5 vizinhos
    mais proximos (os deals ganhos do treino), por desfecho real.
    Painel direito: onde os deals ganhos estao no espaco orcamento x urgencia.

    Espelha o BLOCO 2 de dashboard/analise_modelos.py. Retorna False -- sem
    abortar as demais figuras -- se os modelos ou a base ainda nao existirem.
    """
    caminhos = {
        "leads": os.path.join(PASTA_DADOS, "leads_base.csv"),
        "hist": os.path.join(PASTA_SAIDAS, "historico_estado_a_consolidado.csv"),
        "knn": os.path.join(PASTA_MODELOS, "modelo_match.pkl"),
        "scaler": os.path.join(PASTA_MODELOS, "scaler_match.pkl"),
    }
    faltando = [k for k, v in caminhos.items() if not os.path.exists(v)]
    if faltando:
        print(f"  [kNN] pulada -- ausentes: {', '.join(faltando)}")
        return False

    leads = pd.read_csv(caminhos["leads"])
    hist = pd.read_csv(caminhos["hist"])
    with open(caminhos["knn"], "rb") as fh:
        modelo_knn = pickle.load(fh)
    with open(caminhos["scaler"], "rb") as fh:
        scaler = pickle.load(fh)

    distancias, _ = modelo_knn.kneighbors(scaler.transform(leads[FEATURES_KNN]))
    dist_media = distancias.mean(axis=1)
    merged = hist.merge(
        pd.DataFrame({"lead_id": leads["lead_id"], "dist_media": dist_media}),
        on="lead_id",
    )
    fechou = merged[merged["fechou_negocio"] == 1]["dist_media"]
    nao_fechou = merged[merged["fechou_negocio"] == 0]["dist_media"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # sem labels=/tick_labels= no boxplot: o nome do argumento mudou no
    # matplotlib 3.9, e os rotulos sao postos depois, o que funciona em ambas.
    bp = ax1.boxplot([nao_fechou, fechou],
                     patch_artist=True, widths=0.55, showfliers=True,
                     flierprops=dict(marker="o", markersize=2.5, alpha=0.35,
                                     markerfacecolor="#666460", markeredgecolor="none"),
                     medianprops=dict(color="white", linewidth=1.6))
    for caixa, cor in zip(bp["boxes"], [COR_NAO_FECHOU, COR_FECHOU]):
        caixa.set_facecolor(cor)
        caixa.set_edgecolor("white")
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Não fechou", "Fechou negócio"])
    ax1.set_ylabel("Distância euclidiana média")
    ax1.set_title("Distância aos cinco vizinhos mais próximos\n(negócios fechados da base de treino)",
                  fontsize=11, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    for i, serie in enumerate([nao_fechou, fechou], start=1):
        # rotulo fora da caixa (meia-largura = 0,275), para nao ficar sobre o
        # preenchimento e perder contraste
        rotulo = f"{serie.median():.3f}".replace(".", ",")
        ax1.text(i + 0.31, serie.median(), rotulo, va="center", ha="left", fontsize=9)
    ax1.set_xlim(0.5, 2.6)

    ganhos = set(hist[hist["fechou_negocio"] == 1]["lead_id"])
    leads_plot = leads.copy()
    leads_plot["e_deal_ganho"] = leads_plot["lead_id"].isin(ganhos)
    # random_state fixo: mesma amostra do dashboard, e estavel entre execucoes
    amostra = leads_plot.sample(min(500, len(leads_plot)), random_state=42)
    demais = amostra[~amostra["e_deal_ganho"]]
    ax2.scatter(demais["orcamento_declarado"], demais["urgencia_declarada"],
                s=12, c=COR_DEMAIS, alpha=0.7, linewidths=0, label="Demais leads (amostra de 500)")
    ganhos_plot = leads_plot[leads_plot["e_deal_ganho"]]
    ax2.scatter(ganhos_plot["orcamento_declarado"], ganhos_plot["urgencia_declarada"],
                s=34, c=COR_FECHOU, marker="D", linewidths=0, label="Negócios fechados")
    ax2.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
    ax2.set_xlabel("Orçamento declarado (R$)")
    ax2.set_ylabel("Urgência declarada")
    ax2.set_title("Onde estão os negócios fechados\nno espaço de atributos", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=8, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.22), ncol=2)

    # separador de milhar pt-BR aplicado so' ao numero, nao a frase inteira
    n_leads_txt = f"{len(merged):,}".replace(",", ".")
    fig.suptitle(
        f"Separação de perfil medida pelo kNN (base de comparação: "
        f"{n_leads_txt} leads, {int(merged['fechou_negocio'].sum())} fechados)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, "knn_separacao.png"), dpi=300)
    fig.savefig(os.path.join(PASTA_FIGURAS, "knn_separacao.svg"))
    plt.close(fig)

    # Estatisticas impressas para conferencia contra o texto da secao 5.8.7.
    n_f, n_nf = len(fechou), len(nao_fechou)
    s_agrupado = np.sqrt(((n_f - 1) * fechou.var(ddof=1) + (n_nf - 1) * nao_fechou.var(ddof=1))
                         / (n_f + n_nf - 2))
    d_cohen = (nao_fechou.mean() - fechou.mean()) / s_agrupado
    print("  knn_separacao.png (300 dpi), knn_separacao.svg")
    print(f"    fechados     n={n_f:4d}  mediana {fechou.median():.3f}  média {fechou.mean():.3f}")
    print(f"    não fechados n={n_nf:4d}  mediana {nao_fechou.median():.3f}  média {nao_fechou.mean():.3f}")
    print(f"    d de Cohen = {d_cohen:.2f}")
    try:
        from scipy import stats
        t, pval = stats.ttest_ind(fechou, nao_fechou, equal_var=False)
        print(f"    t de Welch = {t:.2f}  p = {pval:.3e}")
    except ImportError:
        pass
    return True


def gerar_figura_funil():
    """Volume de leads que atravessa cada estacao do Estado A.

    Le a decomposicao por etapa da rodada oficial, de modo que os numeros
    sejam os mesmos reportados na monografia.
    """
    import glob as _glob
    cands = [f for f in _glob.glob(os.path.join(PASTA_SAIDAS, "experimento_*x_decomposicao_etapas.csv"))
             if "_smoke" not in f]
    if not cands:
        print("  [funil] pulada -- decomposicao por etapa ausente")
        return False
    df = pd.read_csv(sorted(cands)[-1])
    a = df[df["estado"] == "Estado A"].copy()
    if a.empty:
        print("  [funil] pulada -- Estado A ausente na decomposicao")
        return False

    rotulos = {"geracao_leads": "Geração", "qualificacao_sdr": "Qualificação",
               "descoberta_closer": "Descoberta", "proposta_manual": "Proposta",
               "negociacao": "Negociação", "fechamento": "Fechamento",
               "pos_venda": "Pós-venda"}
    a["rotulo"] = a["etapa"].map(rotulos).fillna(a["etapa"])
    vol = a["leads_processados_na_etapa"].values

    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = range(len(a))
    ax.barh(list(y), vol, color=CORES_ESTADO["Estado A"], edgecolor="white", height=0.66)
    ax.set_yticks(list(y))
    ax.set_yticklabels(a["rotulo"])
    ax.invert_yaxis()
    ax.set_xlabel("Leads que atravessam a etapa (média de 100 réplicas)")
    ax.set_title("Funil comercial do Estado A", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(vol):
        ax.text(v + max(vol) * 0.01, i, f"{v:,.0f}".replace(",", "."),
                va="center", fontsize=9)
    ax.set_xlim(0, max(vol) * 1.12)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, "funil_estado_a.png"), dpi=300)
    fig.savefig(os.path.join(PASTA_FIGURAS, "funil_estado_a.svg"))
    plt.close(fig)
    print("  funil_estado_a.png (300 dpi), funil_estado_a.svg")
    return True


def _carregar_modelos():
    """Devolve (leads, hist, rf, knn, scaler) ou None se algo faltar."""
    caminhos = {
        "leads": os.path.join(PASTA_DADOS, "leads_base.csv"),
        "hist": os.path.join(PASTA_SAIDAS, "historico_estado_a_consolidado.csv"),
        "rf": os.path.join(PASTA_MODELOS, "modelo_score.pkl"),
        "knn": os.path.join(PASTA_MODELOS, "modelo_match.pkl"),
        "scaler": os.path.join(PASTA_MODELOS, "scaler_match.pkl"),
        "thr": os.path.join(PASTA_MODELOS, "threshold_score.pkl"),
    }
    faltando = [k for k, v in caminhos.items() if not os.path.exists(v)]
    if faltando:
        print(f"  [modelos] ausentes: {', '.join(faltando)}")
        return None
    obj = {}
    obj["leads"] = pd.read_csv(caminhos["leads"])
    obj["hist"] = pd.read_csv(caminhos["hist"])
    for k in ("rf", "knn", "scaler", "thr"):
        with open(caminhos[k], "rb") as fh:
            obj[k] = pickle.load(fh)
    return obj


def gerar_figuras_modelos():
    """Duas figuras sobre o comportamento dos modelos na base de comparacao.

    A primeira cruza a distribuicao dos escores do RandomForest com a da
    similaridade calculada pelo kNN; a segunda separa o escore por desfecho
    real e exibe a importancia relativa de cada atributo no RandomForest.
    """
    obj = _carregar_modelos()
    if obj is None:
        print("  [modelos] figuras puladas")
        return False
    leads, hist, rf, knn, scaler, thr = (obj["leads"], obj["hist"], obj["rf"],
                                         obj["knn"], obj["scaler"], obj["thr"])
    proba = rf.predict_proba(leads[FEATURES_KNN])[:, 1]
    dist, _ = knn.kneighbors(scaler.transform(leads[FEATURES_KNN]))
    simil = 1 / (1 + dist.mean(axis=1))
    df = leads[["lead_id"]].copy()
    df["proba"] = proba
    df["simil"] = simil
    df = df.merge(hist[["lead_id", "fechou_negocio"]], on="lead_id", how="left")
    df["fechou_negocio"] = df["fechou_negocio"].fillna(0).astype(int)

    # --- distribuicoes lado a lado ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.hist(df["proba"], bins=40, color=CORES_ESTADO["Estado D"], edgecolor="white")
    ax1.axvline(thr, color="#444", linestyle="--", linewidth=1.2)
    ax1.text(thr, ax1.get_ylim()[1] * 0.94, f" limiar {thr:.4f}".replace(".", ","),
             fontsize=8, va="top")
    ax1.set_xlabel("Escore previsto pelo RandomForest")
    ax1.set_ylabel("Número de leads")
    ax1.set_title("Escores do RandomForest", fontsize=11, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.hist(df["simil"], bins=40, color=COR_FECHOU, edgecolor="white")
    ax2.set_xlabel("Similaridade de perfil calculada pelo kNN")
    ax2.set_ylabel("Número de leads")
    ax2.set_title("Similaridade do kNN", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Distribuições sobre a base de comparação "
                 f"({len(df):,} leads)".replace(",", "."), fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, "distribuicoes_modelos.png"), dpi=300)
    fig.savefig(os.path.join(PASTA_FIGURAS, "distribuicoes_modelos.svg"))
    plt.close(fig)

    # --- escore por desfecho + importancia dos atributos ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    f = df[df["fechou_negocio"] == 1]["proba"]
    nf = df[df["fechou_negocio"] == 0]["proba"]
    ax1.hist([nf, f], bins=30, stacked=False, label=["Não fechou", "Fechou negócio"],
             color=[COR_NAO_FECHOU, COR_FECHOU], edgecolor="white")
    ax1.axvline(thr, color="#444", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Escore previsto pelo RandomForest")
    ax1.set_ylabel("Número de leads")
    ax1.set_title("Escore por resultado real", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    imp = getattr(rf, "feature_importances_", None)
    rotulos = {"orcamento_declarado": "Orçamento\ndeclarado",
               "urgencia_declarada": "Urgência\ndeclarada",
               "tamanho_funcionarios": "Tamanho\n(funcionários)"}
    nomes = [rotulos.get(c, c) for c in FEATURES_KNN]
    ordem = sorted(range(len(imp)), key=lambda k: imp[k])
    ax2.barh([nomes[k] for k in ordem], [imp[k] for k in ordem],
             color=CORES_ESTADO["Estado D"], edgecolor="white", height=0.6)
    ax2.set_xlabel("Importância relativa")
    ax2.set_title("Peso de cada atributo no modelo", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    for k, pos in enumerate(ordem):
        ax2.text(imp[pos] + 0.01, k, f"{100*imp[pos]:.1f}%".replace(".", ","),
                 va="center", fontsize=9)
    ax2.set_xlim(0, max(imp) * 1.25)

    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, "escore_e_importancia.png"), dpi=300)
    fig.savefig(os.path.join(PASTA_FIGURAS, "escore_e_importancia.svg"))
    plt.close(fig)

    print("  distribuicoes_modelos.png (300 dpi), distribuicoes_modelos.svg")
    print("  escore_e_importancia.png (300 dpi), escore_e_importancia.svg")
    print(f"    importâncias: " + ", ".join(
        f"{c} {100*v:.1f}%" for c, v in zip(FEATURES_KNN, imp)))
    return True


def gerar_figura_arvore():
    """Arvore de diretorios do repositorio, lida do proprio repositorio.

    A fonte e' `git ls-files`, de modo que a figura mostra exatamente o que
    esta' versionado -- nem arquivos locais nao publicados, nem pastas
    ignoradas. Sem git disponivel, cai para a varredura do sistema de
    arquivos aplicando as mesmas exclusoes do .gitignore.

    Grupos repetitivos sao colapsados com a contagem, para que a figura
    caiba em uma folha sem omitir a estrutura.
    """
    import subprocess
    from collections import defaultdict

    raiz = os.path.abspath(os.path.join(PASTA_DADOS, ".."))
    try:
        saida = subprocess.run(["git", "-C", raiz, "ls-files"],
                               capture_output=True, text=True, timeout=30)
        arquivos = [l for l in saida.stdout.splitlines() if l.strip()]
        origem = "git ls-files"
    except Exception:
        arquivos, origem = [], "varredura"
    if not arquivos:
        ignorar = ("_arquivo_", "_backup_", "_scratch_", "__pycache__", ".git/",
                   ".claude/", ".DS_Store")
        for base, dirs, fs in os.walk(raiz):
            rel = os.path.relpath(base, raiz)
            if any(x in rel for x in ignorar):
                continue
            for f in fs:
                caminho = os.path.normpath(os.path.join(rel, f))
                if not any(x in caminho for x in ignorar):
                    arquivos.append(caminho.replace(os.sep, "/"))
        origem = "varredura"

    # colapsa os grupos repetitivos, preservando a contagem
    COLAPSAR = {
        "saidas/figuras": "as sete figuras de dados, em PNG e SVG",
        "saidas/sensibilidade": "os cinco cenários da análise de sensibilidade",
    }
    por_dir = defaultdict(list)
    for a in arquivos:
        por_dir["/".join(a.split("/")[:-1])].append(a.split("/")[-1])

    linhas = []           # (texto, tipo) com tipo em {dir, arq, nota}
    def montar(pai, prefixo):
        subdirs = sorted({d for d in por_dir
                          if d and "/".join(d.split("/")[:-1]) == pai and d != pai})
        arqs = sorted(por_dir.get(pai, []))
        itens = [(d, True) for d in subdirs] + [(a, False) for a in arqs]
        for k, (nome, e_dir) in enumerate(itens):
            ult = k == len(itens) - 1
            ramo = "└── " if ult else "├── "
            if e_dir:
                curto = nome.split("/")[-1]
                linhas.append((prefixo + ramo + curto + "/", "dir"))
                if nome in COLAPSAR:
                    n = len(por_dir.get(nome, []))
                    linhas.append((prefixo + ("    " if ult else "│   ")
                                   + f"└── ({n} arquivos — {COLAPSAR[nome]})", "nota"))
                else:
                    montar(nome, prefixo + ("    " if ult else "│   "))
            else:
                linhas.append((prefixo + ramo + nome, "arq"))

    montar("", "")
    n_dir = len({d for d in por_dir if d})
    n_arq = len(arquivos)

    # Duas colunas: 70 linhas numa coluna so' obrigariam um corpo ilegivel na
    # largura da folha. O corte e' feito num limite de diretorio de primeiro
    # nivel, para nao partir uma subarvore ao meio.
    inicios = [k for k, (t, tp) in enumerate(linhas)
               if tp == "dir" and not t.startswith(("│", " "))]
    meio = len(linhas) / 2
    corte = min(inicios, key=lambda k: abs(k - meio)) if inicios else len(linhas) // 2
    col1, col2 = linhas[:corte], linhas[corte:]
    n_lin = max(len(col1), len(col2))

    altura = max(5.0, 0.21 * (n_lin + 4))
    fig, ax = plt.subplots(figsize=(9.2, altura))
    ax.axis("off")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, n_lin + 3)

    def desenhar(coluna, x, titulo=None):
        y = n_lin + 2
        if titulo:
            ax.text(x, y, titulo, family="monospace", fontsize=10,
                    fontweight="bold", color=CORES_ESTADO["Estado C"], va="top")
        y -= 1
        for texto, tipo in coluna:
            cor = {"dir": CORES_ESTADO["Estado C"], "nota": "#6b6b6b"}.get(tipo, "#1a1a1a")
            ax.text(x, y, texto, family="monospace", fontsize=8.8, color=cor,
                    fontweight="bold" if tipo == "dir" else "normal",
                    style="italic" if tipo == "nota" else "normal", va="top")
            y -= 1

    desenhar(col1, 0.01, "metalflex_simulacao/")
    desenhar(col2, 1.02, "(continuação)")
    ax.text(0.01, 0.6, f"{n_dir} diretórios, {n_arq} arquivos versionados",
            family="monospace", fontsize=9, color="#3a3a3a", va="top")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(PASTA_FIGURAS, "estrutura_repositorio.png"), dpi=300)
    fig.savefig(os.path.join(PASTA_FIGURAS, "estrutura_repositorio.svg"))
    plt.close(fig)
    print("  estrutura_repositorio.png (300 dpi), estrutura_repositorio.svg")
    print(f"    fonte: {origem} — {n_dir} diretórios, {n_arq} arquivos, "
          f"{len(linhas)} linhas")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None, help="Caminho do CSV de réplicas (default: acha o oficial mais recente)")
    args = parser.parse_args()

    caminho = args.csv or encontrar_csv_oficial()
    df_temp = pd.read_csv(caminho)
    n_replicas = df_temp["repeticao"].nunique()
    gerar(caminho, n_replicas)
