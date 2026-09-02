"""
sementes.py — inventário único e verificável das fontes de aleatoriedade
do experimento Metalflex.

    Andreazzi, A. Redesenho sistêmico de processos como condição para a adoção
    de inteligência artificial: um experimento simulado em um funil comercial B2B.
    MBA em Inteligência Artificial e Big Data — ICMC/USP, 2026.

Este módulo tem duas finalidades.

1. DOCUMENTAR. Reúne, em um só lugar, todas as sementes e parâmetros
   determinísticos do experimento, com a origem de cada um no código-fonte.
   Sem isso, a informação fica espalhada por sete arquivos e a alegação de
   replicabilidade não é auditável.

2. VERIFICAR. Executado diretamente (python3 sementes.py), roda um conjunto
   de checagens que falham com código de saída diferente de zero se qualquer
   premissa de reprodutibilidade tiver sido quebrada.

Uso:
    python3 sementes.py              # imprime o inventário e roda as verificações
    python3 sementes.py --tabela     # só o inventário, em formato de tabela
    python3 sementes.py --hashes     # confere os hashes MD5 das bases geradas

Como módulo:
    from sementes import SEED_AVALIACAO, SEED_TREINO, derivar_semente
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constantes — as sementes propriamente ditas
# ---------------------------------------------------------------------------

#: Base de leads sobre a qual os quatro estados são comparados (gerar_base.py)
SEED_AVALIACAO = 42

#: Base de leads usada exclusivamente para treinar os modelos, disjunta da
#: base de avaliação (gerar_base_treino.py). Também é usada como número de
#: repetição na geração do histórico de treino, em
#: rodar_estado_a_para_treino_modelos.py.
SEED_TREINO = 142

#: random_state do RandomForest e da divisão treino/teste (treinar_modelos.py)
SEED_MODELO = 42

#: seed e temperature na chamada ao Ollama (ia.py)
SEED_LLM = 42
TEMPERATURE_LLM = 0

#: Número de leads em cada base
N_LEADS = 2000
N_LEADS_TREINO = 2000

#: Número de réplicas da rodada oficial (experimento_30x.py --n-repeticoes)
N_REPLICAS_OFICIAL = 100

#: Hashes MD5 das bases geradas pela rodada oficial. Servem como impressão
#: digital: se gerar_base.py for executado em outra máquina e produzir um
#: hash diferente, a reprodutibilidade da geração foi quebrada.
MD5_ESPERADO = {
    "leads_base.csv": "c289175b43b056927abe2c3e87d41259",
    "leads_treino.csv": "049d7cb18beb6e758579606278d0a80c",
}

#: Etapas que consomem sorteios aleatórios, por estado (estados.py).
#: A semente é aplicada por etapa, e não por réplica inteira — ver
#: derivar_semente() e a justificativa em pipeline._seed_pareada.
ETAPAS_POR_ESTADO = {
    "A": ["geracao_leads", "qualificacao_sdr", "descoberta_closer",
          "proposta_manual", "negociacao", "fechamento", "pos_venda"],
    "B": ["geracao_leads", "qualificacao_sdr", "descoberta_closer",
          "proposta_ia", "negociacao", "fechamento", "pos_venda"],
    "C": ["geracao_leads", "qualificacao_ia", "match_closer_ia",
          "descoberta_closer", "proposta_ia", "negociacao", "fechamento",
          "pos_venda"],
    "D": ["geracao_leads", "qualificacao_ia", "descoberta_closer",
          "proposta_manual", "negociacao", "fechamento", "pos_venda"],
}


# ---------------------------------------------------------------------------
# Derivação da semente por etapa
# ---------------------------------------------------------------------------

def derivar_semente(repeticao: int, lead_id: int, nome_etapa: str) -> int:
    """Reimplementação exata de pipeline._seed_pareada.

    A semente de cada etapa estocástica é derivada da tripla
    (repetição, lead, etapa) por MD5. Duas propriedades importam:

    - MD5 em vez de hash() embutido: desde o Python 3.3, PYTHONHASHSEED é
      aleatorizado por padrão, de modo que hash("negociacao") muda entre
      execuções do interpretador. MD5 é estável entre processos e máquinas.

    - Semente por etapa, e não por réplica: os estados percorrem sequências
      de etapas de comprimentos diferentes (C e D não executam
      qualificacao_sdr, que consome sorteios em A e B). Uma única semente no
      início da réplica desalinharia todos os sorteios seguintes. Semeando por
      etapa, "negociação do lead X na réplica N" consome os mesmos números em
      A, B, C e D — que é o que torna as comparações pareadas.
    """
    chave = f"{repeticao}|{lead_id}|{nome_etapa}"
    return int(hashlib.md5(chave.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Inventário
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Semente:
    elemento: str
    valor: str
    origem: str
    efeito: str


INVENTARIO: list[Semente] = [
    Semente(
        "Base de avaliação",
        f"seed={SEED_AVALIACAO}",
        "gerar_base.py",
        f"Fixa os {N_LEADS} leads sobre os quais os quatro estados são comparados",
    ),
    Semente(
        "Base de treino",
        f"seed={SEED_TREINO}",
        "gerar_base_treino.py",
        "Fixa a base de treino e garante disjunção em relação à base de avaliação",
    ),
    Semente(
        "Histórico de treino",
        f"repeticao={SEED_TREINO}",
        "rodar_estado_a_para_treino_modelos.py",
        "Torna determinística a execução do Estado A que rotula os dados de treino",
    ),
    Semente(
        "Divisão treino/teste",
        f"random_state={SEED_MODELO}",
        "treinar_modelos.py",
        "Fixa a partição 80/20 estratificada usada para medir a AUC",
    ),
    Semente(
        "RandomForest",
        f"random_state={SEED_MODELO}",
        "treinar_modelos.py",
        "Fixa a amostragem bootstrap e a seleção de atributos das 200 árvores",
    ),
    Semente(
        "Modelo de linguagem",
        f"seed={SEED_LLM}, temperature={TEMPERATURE_LLM}",
        "ia.py",
        "Elimina a amostragem estocástica do Llama 3.2 na geração das propostas",
    ),
    Semente(
        "Comportamento humano",
        "MD5(repeticao|lead_id|etapa)[:8]",
        "pipeline._seed_pareada",
        "Pareia as réplicas entre os quatro estados e torna cada uma reproduzível",
    ),
    Semente(
        "Réplicas da rodada oficial",
        f"repeticao = 1..{N_REPLICAS_OFICIAL}",
        "experimento_30x.py",
        "Amostra a distribuição dos resultados; nunca colide com SEED_TREINO",
    ),
]


def imprimir_inventario() -> None:
    larg = [max(len(s.elemento) for s in INVENTARIO),
            max(len(s.valor) for s in INVENTARIO),
            max(len(s.origem) for s in INVENTARIO)]
    cab = ("Elemento", "Valor", "Origem", "Efeito")
    print(f"{cab[0]:<{larg[0]}}  {cab[1]:<{larg[1]}}  {cab[2]:<{larg[2]}}  {cab[3]}")
    print("-" * (sum(larg) + 6 + 40))
    for s in INVENTARIO:
        print(f"{s.elemento:<{larg[0]}}  {s.valor:<{larg[1]}}  {s.origem:<{larg[2]}}  {s.efeito}")


# ---------------------------------------------------------------------------
# Verificações
# ---------------------------------------------------------------------------

def _ok(cond: bool, descricao: str, detalhe: str = "") -> bool:
    print(f"  [{'OK  ' if cond else 'FALHA'}] {descricao}")
    if detalhe and not cond:
        print(f"          {detalhe}")
    return bool(cond)


def verificar_estabilidade_md5() -> bool:
    """A semente derivada não pode depender do processo nem da máquina."""
    esperado = {
        (1, 946, "qualificacao_sdr"): 945057383,
        (1, 946, "negociacao"): 1273454881,
        (142, 946, "qualificacao_sdr"): 2872523827,
        (100, 1077, "fechamento"): 3777996755,
    }
    divergencias = [
        f"{k} -> {derivar_semente(*k)}, esperado {v}"
        for k, v in esperado.items() if derivar_semente(*k) != v
    ]
    return _ok(not divergencias,
               "Sementes derivadas são estáveis entre execuções (MD5, não hash())",
               "; ".join(divergencias))


def verificar_paridade_com_pipeline() -> bool:
    """derivar_semente() deve coincidir com a implementação usada na simulação."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pipeline import _seed_pareada
    except Exception as e:  # pragma: no cover
        return _ok(False, "derivar_semente() coincide com pipeline._seed_pareada",
                   f"não foi possível importar pipeline: {e}")
    casos = [(r, l, e) for r in (1, 7, 100, SEED_TREINO)
             for l in (1, 946, 1077, 2000)
             for e in ("qualificacao_sdr", "negociacao", "fechamento")]
    difs = [c for c in casos if derivar_semente(*c) != _seed_pareada(*c)]
    return _ok(not difs,
               f"derivar_semente() coincide com pipeline._seed_pareada ({len(casos)} casos)",
               f"{len(difs)} divergências")


def verificar_pareamento_entre_estados() -> bool:
    """A semente de uma etapa não pode depender do estado que a executa.

    É essa propriedade que garante que a réplica i do Estado A e a réplica i
    do Estado C difiram apenas pelo tratamento, e não pelo ruído de fundo.
    """
    compartilhadas = set.intersection(*(set(v) for v in ETAPAS_POR_ESTADO.values()))
    falhas = []
    for etapa in sorted(compartilhadas):
        for rep in (1, 50, 100):
            for lead in (1, 946, 2000):
                valores = {derivar_semente(rep, lead, etapa) for _ in ETAPAS_POR_ESTADO}
                if len(valores) != 1:
                    falhas.append((etapa, rep, lead))
    return _ok(not falhas,
               f"Etapas compartilhadas ({len(compartilhadas)}) recebem a mesma "
               f"semente nos quatro estados")


def verificar_ausencia_de_colisao() -> bool:
    """A semente do treino não pode coincidir com nenhuma réplica de avaliação."""
    colide = SEED_TREINO in range(1, N_REPLICAS_OFICIAL + 1)
    return _ok(not colide,
               f"SEED_TREINO ({SEED_TREINO}) não colide com as réplicas "
               f"1..{N_REPLICAS_OFICIAL}",
               "o histórico de treino compartilharia sorteios com uma réplica de avaliação")


def verificar_dispersao() -> bool:
    """Sementes de leads e réplicas vizinhos não podem ficar correlacionadas."""
    amostra = [derivar_semente(r, l, "negociacao")
               for r in range(1, 21) for l in range(1, 101)]
    unicas = len(set(amostra))
    return _ok(unicas == len(amostra),
               f"Sem colisão entre sementes vizinhas ({unicas}/{len(amostra)} distintas)")


def verificar_hashes(raiz: str | None = None) -> bool:
    """Confere o MD5 das bases geradas contra o valor documentado."""
    raiz = raiz or os.path.dirname(os.path.abspath(__file__))
    todos_ok = True
    for arquivo, esperado in MD5_ESPERADO.items():
        caminho = os.path.join(raiz, arquivo)
        if not os.path.exists(caminho):
            todos_ok &= _ok(False, f"{arquivo}: presente", "arquivo não encontrado")
            continue
        with open(caminho, "rb") as f:
            obtido = hashlib.md5(f.read()).hexdigest()
        todos_ok &= _ok(obtido == esperado, f"{arquivo}: MD5 confere",
                        f"obtido {obtido}, esperado {esperado}")
    return todos_ok


def verificar_disjuncao(raiz: str | None = None) -> bool:
    """As duas bases não podem ter lead_id em comum — é o controle contra
    vazamento de dados."""
    raiz = raiz or os.path.dirname(os.path.abspath(__file__))
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return _ok(False, "Bases de treino e avaliação são disjuntas",
                   "pandas não disponível")
    caminhos = [os.path.join(raiz, n) for n in ("leads_base.csv", "leads_treino.csv")]
    if not all(os.path.exists(c) for c in caminhos):
        return _ok(False, "Bases de treino e avaliação são disjuntas",
                   "bases ainda não geradas — rode ../replicar.sh")
    a, b = (pd.read_csv(c) for c in caminhos)
    inter = set(a.lead_id) & set(b.lead_id)
    return _ok(not inter, "Bases de treino e avaliação são disjuntas (lead_id)",
               f"{len(inter)} identificadores em comum")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabela", action="store_true", help="imprime apenas o inventário")
    ap.add_argument("--hashes", action="store_true", help="confere apenas os hashes MD5")
    args = ap.parse_args()

    if args.tabela:
        imprimir_inventario()
        return 0

    if args.hashes:
        print("Hashes das bases geradas:")
        return 0 if verificar_hashes() else 1

    print("=" * 78)
    print(" Inventário de sementes — experimento Metalflex")
    print("=" * 78)
    imprimir_inventario()

    print()
    print("Verificações de reprodutibilidade:")
    resultados = [
        verificar_estabilidade_md5(),
        verificar_paridade_com_pipeline(),
        verificar_pareamento_entre_estados(),
        verificar_ausencia_de_colisao(),
        verificar_dispersao(),
    ]

    print()
    print("Verificações que dependem das bases já geradas:")
    dependentes = [verificar_hashes(), verificar_disjuncao()]

    print()
    if all(resultados) and all(dependentes):
        print("Todas as verificações passaram.")
        return 0
    if all(resultados):
        print("As verificações estruturais passaram; as dependentes de arquivo "
              "falharam.\nSe as bases ainda não foram geradas, rode ../replicar.sh.")
        return 1
    print("ERRO: há premissas de reprodutibilidade quebradas — ver acima.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
