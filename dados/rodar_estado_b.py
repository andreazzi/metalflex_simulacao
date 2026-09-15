"""
rodar_estado_b.py

Executa o Estado B (melhoria pontual: IA generativa isolada na etapa de
proposta) sobre a MESMA base de leads usada no Estado A. Nao depende dos
modelos treinados (RandomForest/kNN) -- so usa o LLM local via Ollama.

Pre-requisito: Ollama rodando localmente (`ollama serve`) com o modelo
llama3.1:8b disponivel (`ollama pull llama3.1:8b`).

Uso:
    python rodar_estado_b.py
"""

import pandas as pd

from estados import ESTADO_B
from pipeline import executar_estado

CAMINHO_LEADS = "leads_base.csv"
CAMINHO_JSONL = "../saidas/historico_estado_b.jsonl"

# Repeticao usada para semear esta execucao avulsa, igual a de
# rodar_estado_a.py: a semente de cada etapa vem da tripla
# (repeticao, lead, etapa) -- ver pipeline._seed_pareada. Com repeticao=1 os
# quatro scripts reproduzem a REPLICA 1 da rodada oficial, o que preserva as
# duas identidades exatas do desenho (A == B e C == D) tambem fora do
# experimento pareado. Sem o parametro, cada execucao caia no estado global
# do modulo random e as identidades se quebravam no painel.
REPETICAO_REFERENCIA = 1


if __name__ == "__main__":
    leads_df = pd.read_csv(CAMINHO_LEADS)
    print(f"Rodando Estado B sobre {len(leads_df)} leads...")
    print("(usa LLM local via Ollama na etapa de proposta -- pode ser lento)\n")

    resumo = executar_estado(ESTADO_B, leads_df, CAMINHO_JSONL,
                             repeticao=REPETICAO_REFERENCIA)

    print("\n=== Resumo Estado B ===")
    print(f"Total de leads processados: {resumo['total_leads']}")
    print(f"Negocios fechados: {resumo['fechados']}")
    print(f"Taxa de conversao: {resumo['fechados'] / resumo['total_leads']:.1%}")
    print(f"Tempo total acumulado: {resumo['tempo_total_min']:.0f} min "
          f"({resumo['tempo_total_min']/60:.0f} horas)")
    print("\nPerdidos por etapa:")
    for etapa, n in resumo["perdidos_por_etapa"].items():
        print(f"  {etapa}: {n}")
