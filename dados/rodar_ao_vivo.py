"""
rodar_ao_vivo.py

Roda um estado (A, B ou C) em modo ESCALONADO: varios leads chegam e
avancam pelo funil de forma intercalada (nao um por vez do inicio ao fim),
criando filas reais e visiveis entre etapas -- pensado para o modo "ao
vivo" do dashboard. Lancado como subprocesso pelo dashboard Streamlit
(dashboard/app.py), mas funciona tambem direto no terminal.

NOTA: para gerar os dados OFICIAIS de comparacao entre estados (usados
nos KPIs finais do TCC), use sempre rodar_estado_a.py / rodar_estado_b.py
/ rodar_estado_c.py, que processam em lote sequencial -- mais simples e
ja validado. Este script aqui e' so para a demonstracao visual ao vivo.

Uso:
    python rodar_ao_vivo.py --estado A --velocidade normal
    python rodar_ao_vivo.py --estado C --velocidade rapida --n-leads 300
    python rodar_ao_vivo.py --estado B --repeticao 7        # reproduz a replica 7 do lote
    python rodar_ao_vivo.py --estado B --exploratorio       # sementes livres, NAO oficial

Velocidades (intervalo de chegada de um novo lead no sistema):
    lenta   -> 1.2s por lead novo
    normal  -> 0.5s por lead novo
    rapida  -> 0.15s por lead novo

Reproducao de replica (ver ADENDO_PARIDADE_AO_VIVO.md): por padrao, este
script reproduz a replica --repeticao (default 1) EXATAMENTE como ela
ocorreu no lote (experimento_30x.py) -- mesmos sorteios estocasticos, ja
que os dois motores agora compartilham o mesmo nucleo de execucao
(pipeline._processar_etapa). Use --exploratorio para sortear livremente
em vez de reproduzir uma replica oficial -- o resumo final avisa
claramente quando esse modo esta' ativo.
"""

import argparse
import pickle
import sys

# O urllib3 usa recursão para retries/redirects e o matplotlib tem pipelines
# de rendering recursivos. Com 2000 leads e muitas chamadas ao LLM, a pilha
# pode ultrapassar o limite padrão (1000) em Python 3.14.
sys.setrecursionlimit(3000)

import pandas as pd

from estados import ESTADOS
from ia import carregar_modelo_match, carregar_modelo_score
from pipeline import executar_estado_escalonado

VELOCIDADES = {
    "lenta": 1.2,
    "normal": 0.5,
    "rapida": 0.15,
}

CAMINHO_LEADS = "leads_base.csv"
CAMINHO_MODELO_SCORE = "../modelos/modelo_score.pkl"
CAMINHO_THRESHOLD = "../modelos/threshold_score.pkl"
CAMINHO_MODELO_MATCH = "../modelos/modelo_match.pkl"
CAMINHO_SCALER_MATCH = "../modelos/scaler_match.pkl"


def main():
    parser = argparse.ArgumentParser(description="Roda um estado da simulacao em tempo real, com filas visiveis.")
    parser.add_argument("--estado", choices=["A", "B", "C", "D"], required=True)
    parser.add_argument("--velocidade", choices=list(VELOCIDADES.keys()), default="normal")
    parser.add_argument("--n-leads", type=int, default=None,
                         help="Processar so os N primeiros leads (util para demos curtas)")
    parser.add_argument("--repeticao", type=int, default=1,
                         help="Qual replica do lote reproduzir exatamente (default: 1)")
    parser.add_argument("--exploratorio", action="store_true",
                         help="Sementes livres (nao reproduz nenhuma replica oficial)")
    args = parser.parse_args()

    intervalo = VELOCIDADES[args.velocidade]
    config = ESTADOS[args.estado]
    repeticao = None if args.exploratorio else args.repeticao

    leads_df = pd.read_csv(CAMINHO_LEADS)
    if args.n_leads:
        leads_df = leads_df.head(args.n_leads)

    caminho_jsonl = f"../saidas/historico_estado_{args.estado.lower()}.jsonl"

    contexto_extra = {}
    if config.usa_modelos_ia:
        contexto_extra["modelo_score"] = carregar_modelo_score(CAMINHO_MODELO_SCORE)
        with open(CAMINHO_THRESHOLD, "rb") as f:
            contexto_extra["threshold_score"] = pickle.load(f)
        # Estado D nao usa kNN de match de closer (ablacao -- ver estados.py):
        # carregar modelo_match/scaler_match so' para quem de fato os usa.
        if "match_closer_ia" in config.etapas:
            contexto_extra["modelo_match"] = carregar_modelo_match(CAMINHO_MODELO_MATCH)
            with open(CAMINHO_SCALER_MATCH, "rb") as f:
                contexto_extra["scaler_match"] = pickle.load(f)

    modo_str = "EXPLORATORIO (sementes livres, nao oficial)" if repeticao is None else f"replica {repeticao} (identica ao lote)"
    print(f"[rodar_ao_vivo] Estado {args.estado} | {len(leads_df)} leads | "
          f"intervalo={intervalo}s/lead novo | modo={modo_str} | saida={caminho_jsonl}", flush=True)

    resumo = executar_estado_escalonado(
        config, leads_df, caminho_jsonl,
        contexto_extra=contexto_extra, intervalo_chegada=intervalo,
        repeticao=repeticao,
    )

    print(f"[rodar_ao_vivo] Concluido ({modo_str}). Fechados: {resumo['fechados']}/{resumo['total_leads']}", flush=True)


if __name__ == "__main__":
    main()
