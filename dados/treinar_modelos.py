"""
treinar_modelos.py

Treina os modelos de IA REAIS usados nos Estados B e C, a partir do
historico consolidado do Estado A rodado sobre a base de TREINO
(historico_treino_para_modelos.csv, gerada por rodar_estado_a_para_treino_modelos.py
a partir de leads_treino.csv).

IMPORTANTE (protocolo out-of-sample): a base de treino e' gerada com uma
semente diferente (gerar_base_treino.py, seed=142) da base usada nas 30
repeticoes de comparacao entre Estados A, B e C (leads_base.csv, seed=42).
As duas bases sao inteiramente disjuntas (nenhum lead_id ou empresa em
comum) -- por isso, quando o Estado C aplica o modelo aos 2.000 leads de
comparacao, ele esta avaliando genuinamente fora da amostra de treino,
eliminando o vazamento de dados (data leakage) que existiria se a mesma
base fosse usada para treinar e para avaliar.

Isso e' executado APENAS UMA VEZ, antes de rodar os Estados B e C.

Uso:
    python treinar_modelos.py
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

CAMINHO_HISTORICO = "../saidas/historico_treino_para_modelos.csv"
CAMINHO_BASE_COMPARACAO = "../dados/leads_base.csv"
CAMINHO_MODELO_SCORE = "../modelos/modelo_score.pkl"
CAMINHO_THRESHOLD = "../modelos/threshold_score.pkl"
CAMINHO_MODELO_MATCH = "../modelos/modelo_match.pkl"
CAMINHO_SCALER_MATCH = "../modelos/scaler_match.pkl"

FEATURES = ["orcamento_declarado", "urgencia_declarada", "tamanho_funcionarios"]


def treinar_modelo_score(df: pd.DataFrame):
    x = df[FEATURES]
    y = df["fechou_negocio"]

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # class_weight="balanced" compensa o forte desbalanceamento (so ~4.5%
    # de fechamentos) -- sem isso, o modelo aprende a sempre prever "nao fecha".
    modelo = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
    )
    modelo.fit(x_treino, y_treino)

    proba_teste = modelo.predict_proba(x_teste)[:, 1]
    auc = roc_auc_score(y_teste, proba_teste)
    print(f"RandomForest (score de fit) -- AUC no teste: {auc:.3f} "
          f"(acuracia simples nao e' informativa com classe tao desbalanceada)")
    print(f"  importancia das features: "
          f"{dict(zip(FEATURES, modelo.feature_importances_.round(3)))}")

    # Threshold calibrado sobre a base de COMPARACAO, e nao sobre a de treino.
    # Motivo: o limiar nao e' um parametro aprendido -- e' o ponto de operacao
    # que fixa a COTA de leads qualificados. Como o SDR humano qualifica 25%
    # dos leads da base de comparacao (limiar 51,798 em humanos.py), o modelo
    # precisa qualificar os mesmos 25% DELA para que a comparacao entre estados
    # isole a QUALIDADE da decisao, e nao um corte de volume.
    # Nao ha' vazamento: o percentil usa apenas as probabilidades previstas
    # pelo proprio modelo a partir dos atributos visiveis -- nenhum rotulo,
    # nenhum desfecho e nenhum fit_real participa do calculo. O modelo em si
    # continua treinado exclusivamente sobre a base de treino, disjunta.
    TAXA_QUALIFICACAO = 0.25
    base_comparacao = pd.read_csv(CAMINHO_BASE_COMPARACAO)
    proba_base_comp = modelo.predict_proba(base_comparacao[FEATURES])[:, 1]
    threshold = float(pd.Series(proba_base_comp).quantile(1 - TAXA_QUALIFICACAO))
    taxa_obtida = float((proba_base_comp > threshold).mean())
    print(f"  threshold calibrado (percentil {100*(1-TAXA_QUALIFICACAO):.0f} "
          f"da base de comparacao): {threshold:.4f}")
    print(f"  taxa de qualificacao resultante: {taxa_obtida:.2%} "
          f"(alvo {TAXA_QUALIFICACAO:.0%})")

    return modelo, threshold


def treinar_modelo_match(df: pd.DataFrame):
    """kNN e' baseado em distancia euclidiana, que so e' comparavel entre
    atributos quando eles estao na mesma escala. Sem padronizacao, orcamento
    (dezenas de milhares) domina a distancia e urgencia (0 a 1) -- o
    atributo que o proprio RandomForest aponta como mais importante --
    torna-se irrelevante para o kNN. O StandardScaler e' ajustado (fit)
    apenas sobre os deals ganhos do treino, e a MESMA transformacao deve
    ser aplicada a qualquer lead consultado depois (ver ia.match_closer_ia)."""
    deals_ganhos = df[df["fechou_negocio"] == 1]
    print(f"kNN (match de perfil) -- treinado sobre {len(deals_ganhos)} deals ganhos")

    scaler = StandardScaler()
    x_escalado = scaler.fit_transform(deals_ganhos[FEATURES])
    print(f"  medias do treino (escala original): "
          f"{dict(zip(FEATURES, deals_ganhos[FEATURES].mean().round(2)))}")
    print(f"  desvios-padrao do treino (escala original): "
          f"{dict(zip(FEATURES, deals_ganhos[FEATURES].std().round(2)))}")

    modelo = NearestNeighbors(n_neighbors=5)
    modelo.fit(x_escalado)
    return modelo, scaler


if __name__ == "__main__":
    os.makedirs(os.path.dirname(CAMINHO_MODELO_SCORE), exist_ok=True)
    df = pd.read_csv(CAMINHO_HISTORICO)
    print(f"Historico carregado: {len(df)} leads, "
          f"{df['fechou_negocio'].sum()} fechamentos ({df['fechou_negocio'].mean():.1%})\n")

    modelo_score, threshold = treinar_modelo_score(df)
    with open(CAMINHO_MODELO_SCORE, "wb") as f:
        pickle.dump(modelo_score, f)
    with open(CAMINHO_THRESHOLD, "wb") as f:
        pickle.dump(threshold, f)
    print(f"Salvo em {CAMINHO_MODELO_SCORE} (threshold em {CAMINHO_THRESHOLD})\n")

    modelo_match, scaler_match = treinar_modelo_match(df)
    with open(CAMINHO_MODELO_MATCH, "wb") as f:
        pickle.dump(modelo_match, f)
    with open(CAMINHO_SCALER_MATCH, "wb") as f:
        pickle.dump(scaler_match, f)
    print(f"Salvo em {CAMINHO_MODELO_MATCH} (scaler em {CAMINHO_SCALER_MATCH})")
