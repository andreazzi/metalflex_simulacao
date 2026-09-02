"""
registrar_rodada.py

Extrai metricas (resumo + por etapa) de um historico_estado_X.jsonl e grava
diretamente nos dois CSVs de acompanhamento das 30 repeticoes:
  - ../saidas/registro_30_execucoes.csv          (resumo por rodada/estado)
  - ../saidas/registro_30_execucoes_por_etapa.csv (tempo por etapa)

Uso:
    python registrar_rodada.py <repeticao> <estado_letra> <caminho_jsonl>
    ex: python registrar_rodada.py 3 A ../saidas/historico_estado_a.jsonl
"""

import csv
import sys

from extrair_metricas import extrair, extrair_por_etapa

CSV_RESUMO = "../saidas/registro_30_execucoes.csv"
CSV_ETAPAS = "../saidas/registro_30_execucoes_por_etapa.csv"


def main():
    repeticao = sys.argv[1]
    estado = sys.argv[2]
    caminho_jsonl = sys.argv[3]

    resumo = extrair(caminho_jsonl)
    with open(CSV_RESUMO, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            repeticao, estado, resumo["fechados"], resumo["taxa_conversao_pct"],
            resumo["tempo_total_h"], resumo["duracao_real_min"],
        ])

    etapas = extrair_por_etapa(caminho_jsonl)
    with open(CSV_ETAPAS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for linha in etapas:
            w.writerow([
                repeticao, estado, linha["etapa"], linha["tempo_total_min"],
                linha["execucoes"], linha["tempo_medio_min"],
            ])

    print(f"Registrado: repeticao={repeticao} estado={estado}")
    print(f"  Resumo: {resumo}")
    print(f"  Etapas: {len(etapas)} linhas gravadas")


if __name__ == "__main__":
    main()
