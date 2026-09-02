"""
rodar_estado_c.py

Executa o Estado C (orquestracao sistemica: fluxo redesenhado com IA
treinada na entrada do funil + match de closer + LLM na proposta).

Pre-requisitos:
1. ja ter rodado rodar_estado_a.py (gera o historico para treino)
2. ja ter rodado treinar_modelos.py (gera modelo_score.pkl e modelo_match.pkl)
3. Ollama rodando localmente com llama3.1:8b

Uso:
    python rodar_estado_c.py
"""

import pickle

import pandas as pd

from estados import ESTADO_C
from ia import carregar_modelo_match, carregar_modelo_score
from pipeline import executar_estado

CAMINHO_LEADS = "leads_base.csv"
CAMINHO_JSONL = "../saidas/historico_estado_c.jsonl"
CAMINHO_MODELO_SCORE = "../modelos/modelo_score.pkl"
CAMINHO_THRESHOLD = "../modelos/threshold_score.pkl"
CAMINHO_MODELO_MATCH = "../modelos/modelo_match.pkl"
CAMINHO_SCALER_MATCH = "../modelos/scaler_match.pkl"

if __name__ == "__main__":
    leads_df = pd.read_csv(CAMINHO_LEADS)

    modelo_score = carregar_modelo_score(CAMINHO_MODELO_SCORE)
    modelo_match = carregar_modelo_match(CAMINHO_MODELO_MATCH)
    with open(CAMINHO_SCALER_MATCH, "rb") as f:
        scaler_match = pickle.load(f)
    with open(CAMINHO_THRESHOLD, "rb") as f:
        threshold_score = pickle.load(f)

    contexto_extra = {
        "modelo_score": modelo_score,
        "modelo_match": modelo_match,
        "scaler_match": scaler_match,
        "threshold_score": threshold_score,
    }

    print(f"Rodando Estado C sobre {len(leads_df)} leads...")
    print("(usa RandomForest + kNN + LLM local -- fluxo redesenhado, 8 etapas)\n")

    resumo = executar_estado(ESTADO_C, leads_df, CAMINHO_JSONL, contexto_extra=contexto_extra)

    print("\n=== Resumo Estado C ===")
    print(f"Total de leads processados: {resumo['total_leads']}")
    print(f"Negocios fechados: {resumo['fechados']}")
    print(f"Taxa de conversao: {resumo['fechados'] / resumo['total_leads']:.1%}")
    print(f"Tempo total acumulado: {resumo['tempo_total_min']:.0f} min "
          f"({resumo['tempo_total_min']/60:.0f} horas)")
    print("\nPerdidos por etapa:")
    for etapa, n in resumo["perdidos_por_etapa"].items():
        print(f"  {etapa}: {n}")
