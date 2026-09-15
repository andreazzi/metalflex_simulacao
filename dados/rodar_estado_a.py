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
# Mesmo nome adotado por rodar_estado_b/c/d.py, que o dashboard procura em
# ARQUIVOS["A"]. Ate aqui este script gravava com sufixo "_treino" -- unico
# dos quatro a fazer isso --, e por isso o Estado A nunca aparecia nas visoes
# por estado do painel, mesmo apos rodar o script.
CAMINHO_JSONL = "../saidas/historico_estado_a.jsonl"
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


# Repeticao usada para semear esta execucao avulsa. O experimento oficial
# (experimento_30x.py) numera suas replicas de 1 a N e deriva a semente de
# cada etapa da tripla (repeticao, lead, etapa) -- ver pipeline._seed_pareada.
# Ao adotar aqui repeticao=1, esta passada reproduz exatamente a REPLICA 1 da
# rodada oficial, e seu resultado pode ser conferido contra a linha
# repeticao=1 de saidas/experimento_100x_replicas.csv.
#
# Antes desta correcao a chamada omitia o parametro, caindo no estado global
# do modulo random: cada execucao produzia um historico diferente, e com ele
# graficos do painel irreprodutiveis. Mesmo defeito ja corrigido em
# rodar_estado_a_para_treino_modelos.py, que usa repeticao=SEED_TREINO.
REPETICAO_REFERENCIA = 1


if __name__ == "__main__":
    leads_df = pd.read_csv(CAMINHO_LEADS)
    print(f"Rodando Estado A sobre {len(leads_df)} leads "
          f"(semente da replica {REPETICAO_REFERENCIA})...")

    resumo = executar_estado(ESTADO_A, leads_df, CAMINHO_JSONL,
                             repeticao=REPETICAO_REFERENCIA)

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
