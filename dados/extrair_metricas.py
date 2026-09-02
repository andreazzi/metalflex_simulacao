"""
extrair_metricas.py

Le um historico_estado_X.jsonl ja gerado pelo dashboard (via "Rodar ao vivo")
e calcula o resumo (fechados, taxa de conversao, tempo total) -- usado pelo
loop de coleta manual de 30 repeticoes, para nao precisar reimplementar essa
logica a cada rodada.

Tambem calcula o detalhamento de tempo por etapa do funil (qualificacao,
descoberta, proposta, negociacao, fechamento, pos-venda), para permitir a
comparacao fina entre estados ao final das 30 repeticoes.

Uso:
    python extrair_metricas.py <caminho_jsonl>
    python extrair_metricas.py <caminho_jsonl> --etapas   # inclui detalhamento por etapa
"""

import json
import sys
from collections import defaultdict
from datetime import datetime


def extrair(caminho_jsonl):
    ids = set()
    fechados = 0
    tempo_total_min = 0.0
    primeiro_timestamp = None
    ultimo_timestamp = None

    with open(caminho_jsonl, "r", encoding="utf-8") as f:
        for linha in f:
            evento = json.loads(linha)
            ids.add(evento["lead_id"])
            tempo_total_min += evento["tempo_gasto_min"]
            if evento["etapa"] == "fechamento" and evento["passou"]:
                fechados += 1
            if primeiro_timestamp is None:
                primeiro_timestamp = evento["timestamp"]
            ultimo_timestamp = evento["timestamp"]

    total_leads = len(ids)
    taxa_conversao = 100.0 * fechados / total_leads if total_leads else 0.0

    duracao_real_min = 0.0
    if primeiro_timestamp and ultimo_timestamp:
        t0 = datetime.fromisoformat(primeiro_timestamp)
        t1 = datetime.fromisoformat(ultimo_timestamp)
        duracao_real_min = (t1 - t0).total_seconds() / 60

    return {
        "total_leads": total_leads,
        "fechados": fechados,
        "taxa_conversao_pct": round(taxa_conversao, 2),
        "tempo_total_min": round(tempo_total_min, 1),
        "tempo_total_h": round(tempo_total_min / 60, 2),
        "duracao_real_min": round(duracao_real_min, 1),
    }


def extrair_por_etapa(caminho_jsonl):
    """Retorna lista de dicts: uma linha por etapa, com tempo total, contagem
    de execucoes e tempo medio por execucao."""
    tempo_por_etapa = defaultdict(float)
    contagem_por_etapa = defaultdict(int)

    with open(caminho_jsonl, "r", encoding="utf-8") as f:
        for linha in f:
            e = json.loads(linha)
            tempo_por_etapa[e["etapa"]] += e["tempo_gasto_min"]
            contagem_por_etapa[e["etapa"]] += 1

    linhas = []
    for etapa, tempo in tempo_por_etapa.items():
        n = contagem_por_etapa[etapa]
        linhas.append({
            "etapa": etapa,
            "tempo_total_min": round(tempo, 1),
            "execucoes": n,
            "tempo_medio_min": round(tempo / n, 2) if n else 0.0,
        })
    return sorted(linhas, key=lambda x: -x["tempo_total_min"])


if __name__ == "__main__":
    caminho = sys.argv[1]
    resultado = extrair(caminho)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))

    if "--etapas" in sys.argv:
        print()
        print(json.dumps(extrair_por_etapa(caminho), ensure_ascii=False, indent=2))
