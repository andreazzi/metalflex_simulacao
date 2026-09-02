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
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

PASTA_SAIDAS = "../saidas"
PASTA_FIGURAS = "../saidas/figuras"

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None, help="Caminho do CSV de réplicas (default: acha o oficial mais recente)")
    args = parser.parse_args()

    caminho = args.csv or encontrar_csv_oficial()
    df_temp = pd.read_csv(caminho)
    n_replicas = df_temp["repeticao"].nunique()
    gerar(caminho, n_replicas)
