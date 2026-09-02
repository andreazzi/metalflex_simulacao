#!/usr/bin/env bash
#
# replicar.sh — reproduz integralmente os resultados reportados no Capítulo 5
#
#   Andreazzi, A. Redesenho sistêmico de processos como condição para a adoção
#   de inteligência artificial: um experimento simulado em um funil comercial B2B.
#   MBA em Inteligência Artificial e Big Data — ICMC/USP, 2026.
#
# Uso:
#   ./replicar.sh              # rodada oficial: 100 réplicas por estado
#   ./replicar.sh 30           # rodada reduzida, para verificação rápida
#   ./replicar.sh 3 --smoke    # teste de fumaça; nunca sobrescreve a rodada oficial
#
# A ordem das etapas é obrigatória por encadeamento de dependências:
#   gerar_base_treino  exclui da base de treino todo lead da base de avaliação;
#   rodar_estado_a_para_treino_modelos  produz o histórico rotulado;
#   treinar_modelos  ajusta RandomForest e kNN sobre esse histórico;
#   experimento_30x  consome os modelos já treinados.
# Inverter qualquer etapa interrompe a execução ou introduz vazamento de dados.

set -euo pipefail

N="${1:-100}"
SMOKE="${2:-}"

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RAIZ"

echo "=============================================================="
echo " Replicação do experimento Metalflex"
echo " Réplicas por estado: $N ${SMOKE:+(modo smoke)}"
echo " Início: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# ---------------------------------------------------------------------------
# Etapa 0 — arquivar artefatos da rodada anterior (arquiva, não apaga)
# ---------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
ARQ="saidas/_arquivo_replicar_${TS}"
mkdir -p "$ARQ/dados" "$ARQ/modelos" "$ARQ/saidas"

mv dados/leads_base.csv dados/leads_treino.csv "$ARQ/dados/" 2>/dev/null || true
mv modelos/*.pkl "$ARQ/modelos/" 2>/dev/null || true
find saidas -maxdepth 1 -type f \
     \( -name "historico_*" -o -name "experimento_*" -o -name "registro_*" \) \
     -exec mv {} "$ARQ/saidas/" \; 2>/dev/null || true

echo "[0/4] Artefatos anteriores arquivados em $ARQ"

cd dados

# ---------------------------------------------------------------------------
# Etapa 1 — bases de leads, mutuamente disjuntas
# ---------------------------------------------------------------------------
echo "[1/4] Gerando bases de leads..."
python3 gerar_base.py           # seed=42  -> leads_base.csv   (avaliação)
python3 gerar_base_treino.py    # seed=142 -> leads_treino.csv (treino, disjunta)

# ---------------------------------------------------------------------------
# Etapa 2 — histórico rotulado de treino
# ---------------------------------------------------------------------------
echo "[2/4] Gerando histórico de treino (Estado A sobre leads_treino.csv)..."
python3 rodar_estado_a_para_treino_modelos.py   # -> historico_treino_para_modelos.csv

# ---------------------------------------------------------------------------
# Etapa 3 — ajuste dos modelos
# ---------------------------------------------------------------------------
echo "[3/4] Treinando RandomForest e kNN..."
python3 treinar_modelos.py      # -> modelo_score.pkl, threshold_score.pkl,
                                #    modelo_match.pkl, scaler_match.pkl

# ---------------------------------------------------------------------------
# Etapa 4 — rodada oficial: N réplicas pareadas dos quatro estados
# ---------------------------------------------------------------------------
echo "[4/4] Executando $N réplicas pareadas dos Estados A, B, C e D..."
python3 experimento_30x.py --n-repeticoes "$N" ${SMOKE:+--smoke}

cd "$RAIZ"

# ---------------------------------------------------------------------------
# Verificação de integridade
# ---------------------------------------------------------------------------
echo
echo "Verificando as identidades exatas do desenho pareado..."
python3 - "$N" <<'PYEOF'
import sys, glob, pandas as pd

n = sys.argv[1]
alvos = glob.glob(f"saidas/experimento_{n}x_replicas*.csv")
if not alvos:
    sys.exit(f"ERRO: nenhum arquivo de réplicas encontrado para N={n}")

r = pd.read_csv(sorted(alvos)[0])
p = r.pivot(index="repeticao", columns="estado", values="negocios_fechados")

ok = True
for x, y in [("Estado A", "Estado B"), ("Estado C", "Estado D")]:
    if x in p.columns and y in p.columns:
        iguais = (p[x] == p[y]).all()
        marca = "OK " if iguais else "FALHA"
        print(f"  [{marca}] {x} == {y} em todas as {len(p)} réplicas")
        ok &= bool(iguais)

t = r.pivot(index="repeticao", columns="estado", values="tempo_total_h").mean()
print("\n  Tempo médio por estado (h):")
for e in sorted(t.index):
    print(f"    {e}: {t[e]:10.1f}")

c = r.groupby("estado").taxa_conversao_pct.mean()
print("\n  Conversão média por estado (%):")
for e in sorted(c.index):
    print(f"    {e}: {c[e]:8.4f}")

if not ok:
    sys.exit("\nERRO: as identidades exatas nao se verificaram — execucao invalida.")
print("\n  Todas as verificacoes passaram.")
PYEOF

echo
echo "=============================================================="
echo " Concluído: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Resultados em saidas/experimento_${N}x_*.csv"
echo "=============================================================="
