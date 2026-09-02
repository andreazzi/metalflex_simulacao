"""
experimento_30x.py

Executa os quatro estados (A, B, C, D) N vezes cada sobre a mesma base fixa
de leads, registrando apenas o resumo agregado de cada execucao (nao o log
evento-a-evento completo, que seria redundante e pesado em centenas de
execucoes).

Objetivo: substituir a comparacao de uma unica execucao por uma comparacao
estatistica (media +/- desvio-padrao) entre os quatro estados.

Estado D e' um teste de ablacao (achado E-15): isola a qualificacao por
RandomForest do restante das intervencoes do Estado C (sem kNN, sem LLM),
com a MESMA cardinalidade de intervencao do Estado B (uma etapa), mas em
local diferente -- a restricao do processo, nao a etapa mais cara em tempo.

Desenho pareado: a replica i de cada estado compartilha, etapa a etapa, a
mesma semente aleatoria (ver pipeline.executar_estado, parametro repeticao).
Isso implementa numeros aleatorios comuns -- a replica i do Estado A e a
replica i do Estado C diferem apenas pelo tratamento, nao pelo ruido de
fundo -- o que permite teste t PAREADO entre estados (maior poder
estatistico que amostras independentes) e reduz a variancia da diferenca
medida.

Nomeacao dos arquivos de saida (por numero de repeticoes, nao fixo): evita
a colisao que ja' custou proveniencia confusa em rodadas anteriores -- um
teste rapido com N=3 nunca sobrescreve a rodada oficial de N=100, porque os
nomes dos arquivos sao literalmente diferentes. --smoke acrescenta um sufixo
extra, para quando o teste rapido usa o MESMO N da rodada oficial (ou
proximo dela) e a diferenca de nome por N sozinha nao bastaria.

Uso:
    python experimento_30x.py                    # 30 repeticoes (default)
    python experimento_30x.py --n-repeticoes 100  # rodada oficial de 100
    python experimento_30x.py --n-repeticoes 3 --smoke   # teste rapido, nunca colide

Saida (N = numero de repeticoes efetivamente usado):
    ../saidas/experimento_{N}x_replicas.csv              -- uma linha por execucao
    ../saidas/experimento_{N}x_decomposicao_etapas.csv   -- horas medias e leads
                                                             processados por etapa,
                                                             por estado
    ../saidas/experimento_{N}x_diagnostico_handoff.csv   -- distribuicao de
                                                             qualidade_handoff por
                                                             estado (achado F-11)
    (com --smoke, cada nome acima ganha o sufixo _smoke antes da extensao)
"""

import argparse
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from estados import ESTADOS
from pipeline import executar_estado

CAMINHO_LEADS = "leads_base.csv"

CAMINHO_MODELO_SCORE = "../modelos/modelo_score.pkl"
CAMINHO_THRESHOLD = "../modelos/threshold_score.pkl"
CAMINHO_MODELO_MATCH = "../modelos/modelo_match.pkl"
CAMINHO_SCALER_MATCH = "../modelos/scaler_match.pkl"


def carregar_contexto_c():
    with open(CAMINHO_MODELO_SCORE, "rb") as f:
        modelo_score = pickle.load(f)
    with open(CAMINHO_MODELO_MATCH, "rb") as f:
        modelo_match = pickle.load(f)
    with open(CAMINHO_SCALER_MATCH, "rb") as f:
        scaler_match = pickle.load(f)
    with open(CAMINHO_THRESHOLD, "rb") as f:
        threshold = pickle.load(f)
    return {
        "modelo_score": modelo_score,
        "modelo_match": modelo_match,
        "scaler_match": scaler_match,
        "threshold_score": threshold,
    }


def carregar_contexto_d():
    # Deliberadamente SEM modelo_match/scaler_match: o Estado D nao usa kNN
    # de match de closer -- so a qualificacao recebe IA (ver estados.py).
    # Carregar apenas o que D de fato usa torna qualquer uso acidental do
    # kNN em D um KeyError imediato, em vez de um bug silencioso.
    with open(CAMINHO_MODELO_SCORE, "rb") as f:
        modelo_score = pickle.load(f)
    with open(CAMINHO_THRESHOLD, "rb") as f:
        threshold = pickle.load(f)
    return {
        "modelo_score": modelo_score,
        "threshold_score": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-repeticoes", type=int, default=30, help="Numero de replicas por estado (default: 30)")
    parser.add_argument("--smoke", action="store_true",
                         help="Acrescenta sufixo _smoke aos arquivos de saida -- para testes rapidos "
                              "que nao devem ser confundidos com uma rodada oficial")
    args = parser.parse_args()
    n_repeticoes = args.n_repeticoes
    sufixo_smoke = "_smoke" if args.smoke else ""

    caminho_scratch_jsonl = f"../saidas/_scratch_experimento_{n_repeticoes}x{sufixo_smoke}.jsonl"
    caminho_replicas_csv = f"../saidas/experimento_{n_repeticoes}x_replicas{sufixo_smoke}.csv"
    caminho_decomposicao_csv = f"../saidas/experimento_{n_repeticoes}x_decomposicao_etapas{sufixo_smoke}.csv"
    caminho_handoff_csv = f"../saidas/experimento_{n_repeticoes}x_diagnostico_handoff{sufixo_smoke}.csv"

    leads_df = pd.read_csv(CAMINHO_LEADS)
    contexto_c = carregar_contexto_c()
    contexto_d = carregar_contexto_d()

    colunas = ["repeticao", "estado", "leads_processados", "negocios_fechados",
               "taxa_conversao_pct", "tempo_total_min", "tempo_total_h", "duracao_execucao_s"]
    pd.DataFrame(columns=colunas).to_csv(caminho_replicas_csv, index=False)

    soma_tempo_por_etapa = defaultdict(lambda: defaultdict(float))
    soma_leads_por_etapa = defaultdict(lambda: defaultdict(int))
    handoff_valores_por_estado = defaultdict(list)

    total_inicio = time.time()

    estados_ordem = [
        ("A", "Estado A", None),
        ("B", "Estado B", None),
        ("C", "Estado C", contexto_c),
        ("D", "Estado D", contexto_d),
    ]

    print(f"Rodada: N={n_repeticoes} repeticoes x {len(estados_ordem)} estados "
          f"{'[SMOKE -- arquivos de teste, nao oficiais]' if args.smoke else '[oficial]'}", flush=True)
    print(f"Saida: {caminho_replicas_csv}", flush=True)

    for rep in range(1, n_repeticoes + 1):
        for chave_estado, nome_exibicao, contexto_extra in estados_ordem:
            config = ESTADOS[chave_estado]

            t0 = time.time()
            resumo = executar_estado(
                config, leads_df, caminho_scratch_jsonl,
                contexto_extra=contexto_extra, pausa_entre_leads=0.0,
                repeticao=rep,
            )
            duracao = time.time() - t0

            taxa_conversao = 100.0 * resumo["fechados"] / resumo["total_leads"]
            linha = {
                "repeticao": rep,
                "estado": nome_exibicao,
                "leads_processados": resumo["total_leads"],
                "negocios_fechados": resumo["fechados"],
                "taxa_conversao_pct": round(taxa_conversao, 2),
                "tempo_total_min": round(resumo["tempo_total_min"], 1),
                "tempo_total_h": round(resumo["tempo_total_min"] / 60, 2),
                "duracao_execucao_s": round(duracao, 1),
            }
            pd.DataFrame([linha]).to_csv(caminho_replicas_csv, mode="a", header=False, index=False)

            for etapa in config.etapas:
                soma_tempo_por_etapa[nome_exibicao][etapa] += resumo["tempo_por_etapa_min"][etapa]
                soma_leads_por_etapa[nome_exibicao][etapa] += resumo["leads_processados_por_etapa"][etapa]
            handoff_valores_por_estado[nome_exibicao].extend(resumo["qualidade_handoff_valores"])

            print(f"[rep {rep:02d}/{n_repeticoes}] {nome_exibicao}: "
                  f"{resumo['fechados']} fechados ({taxa_conversao:.1f}%), "
                  f"{resumo['tempo_total_min']/60:.1f}h simuladas, "
                  f"{duracao:.1f}s de execucao real", flush=True)

    if os.path.exists(caminho_scratch_jsonl):
        os.remove(caminho_scratch_jsonl)

    linhas_decomposicao = []
    for chave_estado, nome_exibicao, _ in estados_ordem:
        config = ESTADOS[chave_estado]
        for etapa in config.etapas:
            linhas_decomposicao.append({
                "estado": nome_exibicao,
                "etapa": etapa,
                "horas_media": round(soma_tempo_por_etapa[nome_exibicao][etapa] / n_repeticoes / 60, 3),
                "leads_processados_na_etapa": round(soma_leads_por_etapa[nome_exibicao][etapa] / n_repeticoes, 1),
            })
    pd.DataFrame(linhas_decomposicao).to_csv(caminho_decomposicao_csv, index=False)

    # Diagnostico de comparabilidade de escala (F-11): distribuicao de
    # qualidade_handoff recebida por descoberta_closer, so' para leads
    # aprovados na qualificacao (unica populacao em que o campo existe).
    linhas_handoff = []
    for chave_estado, nome_exibicao, _ in estados_ordem:
        config = ESTADOS[chave_estado]
        valores = np.array(handoff_valores_por_estado[nome_exibicao])
        idx_desc = config.etapas.index("descoberta_closer")
        proxima_etapa = config.etapas[idx_desc + 1]
        leads_descoberta = soma_leads_por_etapa[nome_exibicao]["descoberta_closer"]
        leads_proxima = soma_leads_por_etapa[nome_exibicao][proxima_etapa]
        taxa_reuniao_util = leads_proxima / leads_descoberta if leads_descoberta else float("nan")
        linhas_handoff.append({
            "estado": nome_exibicao,
            "n_leads_aprovados": len(valores),
            "handoff_medio": round(float(valores.mean()), 4),
            "handoff_mediana": round(float(np.median(valores)), 4),
            "handoff_p10": round(float(np.percentile(valores, 10)), 4),
            "handoff_p90": round(float(np.percentile(valores, 90)), 4),
            "taxa_reuniao_util": round(taxa_reuniao_util, 4),
        })
    pd.DataFrame(linhas_handoff).to_csv(caminho_handoff_csv, index=False)

    total_duracao = time.time() - total_inicio
    print(f"\nConcluido: {n_repeticoes} repeticoes x {len(estados_ordem)} estados em {total_duracao/60:.1f} min.")
    print(f"Replicas salvas em {caminho_replicas_csv}")
    print(f"Decomposicao por etapa salva em {caminho_decomposicao_csv}")
    print(f"Diagnostico de handoff salvo em {caminho_handoff_csv}")


if __name__ == "__main__":
    main()
