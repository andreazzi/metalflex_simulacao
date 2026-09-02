"""
rodar_estado_a.py

Executa o Estado A (baseline) sobre toda a base de leads, gera o log
detalhado (historico_estado_a.jsonl) E um CSV consolidado por lead
(historico_estado_a_consolidado.csv) que sera usado para treinar os
modelos de IA dos Estados B e C.

Uso:
    python rodar_estado_a.py
"""

import json

import pandas as pd

from estados import ESTADO_A
from pipeline import executar_estado

CAMINHO_LEADS = "leads_base.csv"
# arquivo separado do historico ao vivo (historico_estado_a.jsonl) para
# que rodar o Estado A pelo setup nao sobrescreva uma demo em andamento.
CAMINHO_JSONL = "../saidas/historico_estado_a_treino.jsonl"
CAMINHO_CONSOLIDADO = "../saidas/historico_estado_a_consolidado.csv"


def consolidar_por_lead(caminho_jsonl: str, leads_df: pd.DataFrame) -> pd.DataFrame:
    """Le o jsonl e produz uma linha por lead com o resultado final (fechou ou nao)."""
    fechou_por_lead = {}
    with open(caminho_jsonl, "r", encoding="utf-8") as f:
        for linha in f:
            evento = json.loads(linha)
            if evento["etapa"] == "fechamento":
                fechou_por_lead[evento["lead_id"]] = bool(evento["passou"])

    leads_df = leads_df.copy()
    leads_df["fechou_negocio"] = leads_df["lead_id"].map(fechou_por_lead).fillna(False).astype(int)
    return leads_df


if __name__ == "__main__":
    leads_df = pd.read_csv(CAMINHO_LEADS)
    print(f"Rodando Estado A sobre {len(leads_df)} leads...")

    resumo = executar_estado(ESTADO_A, leads_df, CAMINHO_JSONL)

    print("\n=== Resumo Estado A ===")
    print(f"Total de leads processados: {resumo['total_leads']}")
    print(f"Negocios fechados: {resumo['fechados']}")
    print(f"Taxa de conversao: {resumo['fechados'] / resumo['total_leads']:.1%}")
    print(f"Tempo total acumulado: {resumo['tempo_total_min']:.0f} min "
          f"({resumo['tempo_total_min']/60:.0f} horas)")
    print("\nPerdidos por etapa:")
    for etapa, n in resumo["perdidos_por_etapa"].items():
        print(f"  {etapa}: {n}")

    consolidado = consolidar_por_lead(CAMINHO_JSONL, leads_df)
    consolidado.to_csv(CAMINHO_CONSOLIDADO, index=False)
    print(f"\nHistorico consolidado salvo em {CAMINHO_CONSOLIDADO}")
