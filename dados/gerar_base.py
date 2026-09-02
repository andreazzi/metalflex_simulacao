"""
gerar_base.py

Gera a base de leads fixa que sera usada por TODOS os estados (A, B, C).
Roda-se UMA UNICA VEZ. O arquivo leads_base.csv resultante e' versionado
junto ao codigo e nunca mais regenerado -- isso garante que qualquer pessoa
que reproduza o experimento parte exatamente dos mesmos dados.

Uso:
    python gerar_base.py
"""

import numpy as np
import pandas as pd

SEED = 42
N_LEADS = 2000


def gerar_base_leads(n_leads: int = N_LEADS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    setores = ["industria", "varejo", "servicos", "tecnologia", "saude"]
    canais = ["inbound", "outbound", "indicacao"]

    # --- Features VISIVEIS: o que SDR humano ou IA conseguem observar ---
    orcamento_declarado = rng.normal(50_000, 20_000, n_leads).clip(min=5_000)
    urgencia_declarada = rng.uniform(0, 1, n_leads)
    tamanho_funcionarios = rng.integers(5, 500, n_leads)
    setor = rng.choice(setores, n_leads)
    canal_origem = rng.choice(canais, n_leads, p=[0.55, 0.30, 0.15])

    # Dia de chegada do lead, distribuido ao longo de ~18 meses uteis
    dia_chegada = rng.integers(0, 540, n_leads)

    # --- Feature OCULTA: a "verdade" que nenhum humano ou IA ve diretamente ---
    # IMPORTANTE: fit_real e' parcialmente determinado pelas features visiveis
    # (orcamento, urgencia, tamanho) + ruido idiossincratico. Isso reflete a
    # realidade: orcamento e urgencia SAO sinais reais de fit, so que imperfeitos.
    # Sem essa correlacao, nem um humano perfeito nem uma IA perfeita poderiam
    # prever fechamento -- o que invalidaria a tese de que IA bem aplicada ajuda.
    orcamento_norm = (orcamento_declarado - orcamento_declarado.min()) / (
        orcamento_declarado.max() - orcamento_declarado.min()
    )
    tamanho_norm = (tamanho_funcionarios - tamanho_funcionarios.min()) / (
        tamanho_funcionarios.max() - tamanho_funcionarios.min()
    )

    sinal_real = (
        0.40 * orcamento_norm
        + 0.35 * urgencia_declarada
        + 0.25 * tamanho_norm
    )
    ruido_idiossincratico = rng.normal(0, 0.18, n_leads)
    fit_real = np.clip(sinal_real + ruido_idiossincratico, 0.01, 0.99)

    df = pd.DataFrame({
        "lead_id": np.arange(n_leads),
        "empresa": [f"Empresa_{i:04d}" for i in range(n_leads)],
        "setor": setor,
        "canal_origem": canal_origem,
        "dia_chegada": dia_chegada,
        "orcamento_declarado": orcamento_declarado.round(2),
        "urgencia_declarada": urgencia_declarada.round(3),
        "tamanho_funcionarios": tamanho_funcionarios,
        "fit_real": fit_real.round(4),
    })

    df = df.sort_values("dia_chegada").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = gerar_base_leads()
    saida = "leads_base.csv"
    df.to_csv(saida, index=False)
    print(f"Gerados {len(df)} leads -> {saida}")
    print(df.head())
    print("\nResumo estatistico:")
    print(df[["orcamento_declarado", "urgencia_declarada", "tamanho_funcionarios", "fit_real"]].describe())
