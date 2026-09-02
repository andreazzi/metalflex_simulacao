"""
gerar_base_treino.py

Gera uma base de leads SEPARADA, usada exclusivamente para treinar o
RandomForest e o kNN do Estado C. Usa a mesma funcao geradora e a mesma
formula de fit_real de gerar_base.py, mas com uma semente diferente --
garantindo que nenhum lead aqui coincida com os 2.000 de leads_base.csv
(a base usada nas 30 repeticoes de comparacao entre Estados A, B e C).

Isso resolve o risco de vazamento de dados (data leakage): o modelo passa
a ser avaliado, no Estado C, sobre leads que ele nunca viu durante o treino.

Uso:
    python gerar_base_treino.py
"""

from gerar_base import gerar_base_leads

SEED_TREINO = 142
N_LEADS_TREINO = 10000


def gerar_base_treino(n_leads: int = N_LEADS_TREINO, seed: int = SEED_TREINO):
    df = gerar_base_leads(n_leads=n_leads, seed=seed)
    # Prefixo/offset distintos de leads_base.csv (Empresa_0000..1999) para deixar
    # explicito, mesmo visualmente, que estes leads sao de treino, nao de comparacao.
    df["lead_id"] = df["lead_id"] + 10_000
    df["empresa"] = [f"Treino_{i:04d}" for i in range(len(df))]
    return df


if __name__ == "__main__":
    df = gerar_base_treino()
    saida = "leads_treino.csv"
    df.to_csv(saida, index=False)
    print(f"Gerados {len(df)} leads de treino (seed={SEED_TREINO}) -> {saida}")
    print(df.head())

    # Verificacao de disjuncao com a base de comparacao
    import pandas as pd
    base_comparacao = pd.read_csv("leads_base.csv")
    overlap = set(df["empresa"]) & set(base_comparacao["empresa"])
    print(f"\nInterseccao de empresas com leads_base.csv: {len(overlap)} (deve ser 0)")
