"""
rodar_estado_d.py

Executa o Estado D (teste de ablacao: so a qualificacao usa IA -- mesmo
RandomForest do Estado C -- o resto do fluxo e' identico ao Estado A,
incluindo a proposta manual). Sem kNN, sem LLM: cardinalidade de
intervencao 1, igual ao Estado B, mas em local diferente (a restricao do
processo em vez da etapa mais cara em tempo).

Pre-requisitos:
1. ja ter rodado rodar_estado_a.py (gera o historico para treino)
2. ja ter rodado treinar_modelos.py (gera modelo_score.pkl e threshold_score.pkl)

NAO precisa de modelo_match.pkl, scaler_match.pkl nem do Ollama rodando --
essa e' exatamente a diferenca proposital em relacao ao Estado C.

Uso:
    python rodar_estado_d.py
"""

import pickle

import pandas as pd

from estados import ESTADO_D
from ia import carregar_modelo_score
from pipeline import executar_estado

CAMINHO_LEADS = "leads_base.csv"
CAMINHO_JSONL = "../saidas/historico_estado_d.jsonl"
CAMINHO_MODELO_SCORE = "../modelos/modelo_score.pkl"
CAMINHO_THRESHOLD = "../modelos/threshold_score.pkl"

if __name__ == "__main__":
    leads_df = pd.read_csv(CAMINHO_LEADS)

    modelo_score = carregar_modelo_score(CAMINHO_MODELO_SCORE)
    with open(CAMINHO_THRESHOLD, "rb") as f:
        threshold_score = pickle.load(f)

    contexto_extra = {
        "modelo_score": modelo_score,
        "threshold_score": threshold_score,
    }

    print(f"Rodando Estado D sobre {len(leads_df)} leads...")
    print("(usa APENAS RandomForest na qualificacao -- sem kNN, sem LLM, 7 etapas)\n")

    resumo = executar_estado(ESTADO_D, leads_df, CAMINHO_JSONL, contexto_extra=contexto_extra)

    print("\n=== Resumo Estado D ===")
    print(f"Total de leads processados: {resumo['total_leads']}")
    print(f"Negocios fechados: {resumo['fechados']}")
    print(f"Taxa de conversao: {resumo['fechados'] / resumo['total_leads']:.1%}")
    print(f"Tempo total acumulado: {resumo['tempo_total_min']:.0f} min "
          f"({resumo['tempo_total_min']/60:.0f} horas)")
    print("\nPerdidos por etapa:")
    for etapa, n in resumo["perdidos_por_etapa"].items():
        print(f"  {etapa}: {n}")
