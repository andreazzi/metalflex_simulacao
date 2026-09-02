"""
rodar_estado_a_para_treino_modelos.py

Executa o Estado A (baseline, sem IA) sobre a base de TREINO (leads_treino.csv,
gerada por gerar_base_treino.py) para produzir o historico consolidado que
sera usado para treinar o RandomForest e o kNN do Estado C.

Diferente de rodar_estado_a.py (que roda sobre leads_base.csv, a base usada
nas 30+ repeticoes de comparacao entre os Estados A, B, C e D), este script
roda sobre uma base DISJUNTA -- isso garante que o treino dos modelos nunca
veja os leads sobre os quais o Estado C sera avaliado, eliminando o risco de
vazamento de dados (data leakage) apontado na revisao do orientador.

Semente: a geracao dos LEADS de treino ja' era determinística (seed=142 em
gerar_base_treino.py), mas a SIMULACAO ESTOCASTICA do comportamento humano
sobre esses leads nao era -- rodava no estado global do modulo random do
Python, sem reseed. Duas execucoes desta etapa produziam historicos de
treino diferentes (e portanto modelos diferentes), quebrando a cadeia de
reprodutibilidade "do zero" apesar do RandomForest ter random_state fixo.

Corrigido reaproveitando o mesmo mecanismo de sementes pareadas usado na
comparacao entre estados (pipeline._seed_pareada), com repeticao=SEED_TREINO
(142) -- o mesmo numero ja' documentado para esta etapa, agora tambem
cobrindo o sorteio estocastico, nao so' a selecao dos leads. Por construcao,
142 nunca coincide com as sementes de avaliacao (repeticao em 1..N do
experimento oficial), preservando a disjuncao treino/teste.

Uso:
    python rodar_estado_a_para_treino_modelos.py
"""

import pandas as pd

from estados import ESTADO_A
from gerar_base_treino import SEED_TREINO
from pipeline import executar_estado
from rodar_estado_a import consolidar_por_lead

CAMINHO_LEADS_TREINO = "leads_treino.csv"
CAMINHO_JSONL_TREINO = "../saidas/historico_estado_a_consolidado_treino.jsonl"
CAMINHO_CONSOLIDADO_TREINO = "../saidas/historico_treino_para_modelos.csv"


if __name__ == "__main__":
    leads_df = pd.read_csv(CAMINHO_LEADS_TREINO)
    print(f"Rodando Estado A sobre a base de TREINO ({len(leads_df)} leads, "
          f"disjunta da base de comparacao), com semente {SEED_TREINO}...")

    resumo = executar_estado(ESTADO_A, leads_df, CAMINHO_JSONL_TREINO, repeticao=SEED_TREINO)

    print("\n=== Resumo Estado A (base de treino) ===")
    print(f"Total de leads processados: {resumo['total_leads']}")
    print(f"Negocios fechados: {resumo['fechados']}")
    print(f"Taxa de conversao: {resumo['fechados'] / resumo['total_leads']:.1%}")

    consolidado = consolidar_por_lead(CAMINHO_JSONL_TREINO, leads_df)
    consolidado.to_csv(CAMINHO_CONSOLIDADO_TREINO, index=False)
    print(f"\nHistorico de treino consolidado salvo em {CAMINHO_CONSOLIDADO_TREINO}")
